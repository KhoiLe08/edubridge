"""Read-only multilingual retrieval for the EduBridge vector index."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from vector_store import COLLECTION_NAME, EMBEDDING_MODEL_NAME


PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "chroma_db"

RESULT_METADATA_FIELDS = (
    "source_id",
    "topic",
    "display_topic",
    "title",
    "author",
    "publisher",
    "source_url",
    "license",
    "chunk_index",
)


class VectorIndexUnavailableError(RuntimeError):
    """Raised when the prebuilt EduBridge vector index cannot be used."""


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load the ingestion-compatible multilingual model only once."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@lru_cache(maxsize=1)
def _get_client() -> Any:
    """Open one process-wide client for the committed Chroma database."""
    if not DATABASE_PATH.is_dir() or not any(DATABASE_PATH.iterdir()):
        raise VectorIndexUnavailableError(
            "The EduBridge vector index is missing. Ensure the prebuilt "
            "chroma_db directory is included in the deployed repository."
        )
    return chromadb.PersistentClient(path=str(DATABASE_PATH))


@lru_cache(maxsize=1)
def _get_collection() -> Any:
    """Open the existing persistent Chroma collection without modifying it."""
    try:
        # get_collection is intentionally used instead of get_or_create_collection
        # so retrieval can never create or replace the index.
        collection = _get_client().get_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
        )
    except VectorIndexUnavailableError:
        raise
    except Exception as exc:
        raise VectorIndexUnavailableError(
            "The prebuilt Chroma database does not contain the required "
            f"'{COLLECTION_NAME}' collection. Rebuild it locally with ingest.py "
            "and commit the complete chroma_db directory."
        ) from exc

    if collection.count() == 0:
        raise VectorIndexUnavailableError(
            f"The '{COLLECTION_NAME}' Chroma collection is empty. Rebuild the "
            "index locally with ingest.py and commit the populated chroma_db directory."
        )
    return collection


def _encode_query(query: str) -> list[float]:
    """Encode one query with the same normalization used during ingestion."""
    model = _get_model()
    encode_query = getattr(model, "encode_query", None)
    encode_method = encode_query if callable(encode_query) else model.encode
    embedding = encode_method(
        query,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()

    # SentenceTransformer returns one flat vector for a single string. Handle
    # a one-row matrix defensively in case an older release returns that form.
    if embedding and isinstance(embedding[0], (list, tuple)):
        embedding = embedding[0]
    vector = [float(value) for value in embedding]
    if not vector:
        raise ValueError("The embedding model returned an empty query embedding.")
    return vector


def _first_result_row(value: Any) -> list[Any]:
    """Extract the first query's row from a nested Chroma query field."""
    if not value:
        return []
    first_row = value[0]
    return list(first_row) if first_row is not None else []


def _format_results(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Chroma's column-oriented query response into clean records."""
    ids = _first_result_row(response.get("ids"))
    documents = _first_result_row(response.get("documents"))
    distances = _first_result_row(response.get("distances"))
    metadatas = _first_result_row(response.get("metadatas"))

    results: list[dict[str, Any]] = []
    for index, chunk_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else None
        metadata = metadata if isinstance(metadata, dict) else {}
        distance = distances[index] if index < len(distances) else None
        document = documents[index] if index < len(documents) else ""

        result: dict[str, Any] = {
            "chunk_id": str(chunk_id),
            "text": "" if document is None else str(document),
            "distance": None if distance is None else float(distance),
        }
        for field in RESULT_METADATA_FIELDS:
            result[field] = metadata.get(field)
        results.append(result)

    return results


def retrieve(
    query: str,
    top_k: int = 4,
    topic: str | None = None,
) -> list[dict[str, Any]]:
    """Return nearest chunks, optionally restricted to one metadata topic."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if topic is not None and (not isinstance(topic, str) or not topic.strip()):
        raise ValueError("topic must be None or a non-empty string")

    collection = _get_collection()
    collection_count = collection.count()
    if collection_count == 0:
        return []

    query_embedding = _encode_query(query.strip())
    query_options: dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, collection_count),
        "include": ["documents", "metadatas", "distances"],
    }
    if topic is not None:
        # Apply the constraint inside Chroma so similarity ranking operates
        # only over vectors belonging to the teacher-selected topic.
        query_options["where"] = {"topic": {"$eq": topic.strip()}}

    response = collection.query(**query_options)
    return _format_results(response)


def _print_query_results(
    query: str,
    results: list[dict[str, Any]],
    topic: str | None,
) -> None:
    """Print readable details for one CLI retrieval test."""
    print("\n" + "=" * 50)
    print(f"QUERY: {query}")
    print(f"TOPIC FILTER: {topic if topic is not None else 'None (all topics)'}")

    if not results:
        print("\nNo results found.")
        return

    for rank, result in enumerate(results, start=1):
        source = f"{result.get('source_id', '')} — {result.get('title', '')}"
        distance = result.get("distance")
        distance_text = "N/A" if distance is None else f"{distance:.6f}"

        print(f"\nRESULT {rank}")
        print(f"topic: {result.get('topic', '')}")
        print(f"source: {source}")
        print(f"distance: {distance_text}")
        print(f"chunk: {result.get('chunk_id', '')}")
        print("first 500 characters of text:")
        print(result.get("text", "")[:500])


def _print_retrieval_summary(
    query_results: list[tuple[str, str | None, list[dict[str, Any]]]],
    top_k: int = 4,
) -> None:
    """Print a compact topic-ranking table for all test queries."""
    topic_headers = " | ".join(f"Top-{rank} topic" for rank in range(1, top_k + 1))
    print("\n=== RETRIEVAL SUMMARY ===")
    print(f"Query | {topic_headers}")

    for query, topic, results in query_results:
        topics = [str(result.get("topic") or "-") for result in results[:top_k]]
        topics.extend(["-"] * (top_k - len(topics)))
        query_label = query if topic is not None else f"{query} [all topics]"
        print(f"{query_label} | " + " | ".join(topics))


def main() -> None:
    """Run filtered and unrestricted Vietnamese retrieval smoke tests."""
    test_queries: list[tuple[str, str | None]] = [
        (
            "Hệ số góc của hàm số bậc nhất có ý nghĩa gì?",
            "linear_functions",
        ),
        (
            (
                "Tam giác vuông có hai cạnh góc vuông dài 3 và 4, "
                "cạnh huyền bằng bao nhiêu?"
            ),
            "pythagorean_theorem",
        ),
        (
            "Tung đồng xu hai lần thì không gian mẫu gồm những kết quả nào?",
            "basic_probability",
        ),
        (
            "Hệ số góc, cạnh huyền và xác suất liên quan đến chủ đề nào?",
            None,
        ),
    ]

    query_results: list[tuple[str, str | None, list[dict[str, Any]]]] = []
    for query, topic in test_queries:
        try:
            results = retrieve(query, top_k=4, topic=topic)
        except Exception as exc:
            print("\n" + "=" * 50)
            print(f"QUERY: {query}")
            print(f"TOPIC FILTER: {topic if topic is not None else 'None (all topics)'}")
            print(f"RETRIEVAL ERROR: {exc}")
            results = []
        else:
            _print_query_results(query, results, topic)
        query_results.append((query, topic, results))

    _print_retrieval_summary(query_results)


if __name__ == "__main__":
    main()
