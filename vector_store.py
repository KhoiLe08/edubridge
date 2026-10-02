"""Embedding and persistent Chroma indexing helpers for EduBridge."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
COLLECTION_NAME = "edubridge"
DATABASE_PATH = PROJECT_ROOT / "chroma_db"
EMBEDDING_BATCH_SIZE = 32
CHROMA_UPSERT_BATCH_SIZE = 100

CHROMA_METADATA_FIELDS = (
    "topic",
    "display_topic",
    "title",
    "author",
    "publisher",
    "source_url",
    "license",
    "license_url",
    "language",
)


@dataclass(frozen=True)
class VectorIndexSummary:
    """Final statistics for one vector-indexing run."""

    model_name: str
    total_chunks_indexed: int
    collection_count: int | str
    database_path: Path


def load_embedding_model(
    model_name: str = EMBEDDING_MODEL_NAME,
) -> SentenceTransformer:
    """Load the multilingual embedding model exactly once per indexing run."""
    print(f"\nLoading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def _as_float_vectors(embeddings: Any) -> list[list[float]]:
    """Convert NumPy, tensor, or nested-list embeddings to plain float lists."""
    if hasattr(embeddings, "tolist"):
        embeddings = embeddings.tolist()
    return [[float(value) for value in vector] for vector in embeddings]


def _validate_chunk_ids(chunks: Sequence[dict[str, Any]]) -> list[str]:
    """Return chunk IDs after verifying that each is present and unique."""
    chunk_ids = [str(chunk.get("chunk_id", "")).strip() for chunk in chunks]
    if any(not chunk_id for chunk_id in chunk_ids):
        raise ValueError("Every chunk must have a non-empty chunk_id.")

    duplicate_ids = sorted(
        chunk_id for chunk_id in set(chunk_ids) if chunk_ids.count(chunk_id) > 1
    )
    if duplicate_ids:
        preview = ", ".join(duplicate_ids[:5])
        raise ValueError(f"Duplicate chunk IDs detected: {preview}")
    return chunk_ids


def _validate_embeddings(
    embeddings: Iterable[Iterable[float]], expected_count: int
) -> int:
    """Verify embedding count, non-empty vectors, finite values, and dimensions."""
    vectors = list(embeddings)
    if len(vectors) != expected_count:
        raise ValueError(
            f"Expected {expected_count} embeddings, received {len(vectors)}."
        )

    dimensions: set[int] = set()
    for index, vector in enumerate(vectors):
        values = list(vector)
        if not values:
            raise ValueError(f"Embedding {index} is empty.")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"Embedding {index} contains a non-finite value.")
        dimensions.add(len(values))

    if len(dimensions) != 1:
        raise ValueError(f"Embedding dimensions are inconsistent: {dimensions}")
    return dimensions.pop()


def encode_chunks(
    model: SentenceTransformer,
    chunks: Sequence[dict[str, Any]],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    """Encode all chunk texts in batches and return normalized vectors."""
    texts = [str(chunk.get("text", "")).strip() for chunk in chunks]
    if any(not text for text in texts):
        raise ValueError("Every chunk must contain non-empty text.")

    encode_document = getattr(model, "encode_document", None)
    encode_method = encode_document if callable(encode_document) else model.encode
    embeddings = encode_method(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    vectors = _as_float_vectors(embeddings)
    _validate_embeddings(vectors, len(chunks))
    return vectors


def _create_collection(database_path: Path, collection_name: str) -> Any:
    """Open the persistent database and get a cosine-distance collection."""
    database_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(database_path))

    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )
    except TypeError:
        # Compatibility path for older Chroma releases that configured HNSW
        # through collection metadata rather than ``configuration``.
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
    return collection


def _chunk_metadata(chunk: dict[str, Any]) -> dict[str, str | int]:
    """Build the exact metadata payload stored for a Chroma document."""
    metadata: dict[str, str | int] = {
        "source_id": str(chunk.get("source_id", chunk.get("id", ""))),
        "chunk_index": int(chunk.get("chunk_index", 0)),
    }
    for field in CHROMA_METADATA_FIELDS:
        value = chunk.get(field, "")
        metadata[field] = "" if value is None else str(value)
    return metadata


def _batched_indexes(total: int, batch_size: int) -> Iterable[tuple[int, int]]:
    """Yield half-open index ranges for Chroma upsert batches."""
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def index_chunks(
    chunks: Sequence[dict[str, Any]],
    database_path: Path = DATABASE_PATH,
    collection_name: str = COLLECTION_NAME,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> VectorIndexSummary:
    """Embed, upsert, synchronize, and validate the persistent Chroma index."""
    if not chunks:
        raise ValueError("Cannot build the vector index because no chunks were created.")

    chunk_ids = _validate_chunk_ids(chunks)
    documents = [str(chunk["text"]) for chunk in chunks]
    metadatas = [_chunk_metadata(chunk) for chunk in chunks]

    model = load_embedding_model(model_name)
    embeddings = encode_chunks(model, chunks)
    expected_dimension = _validate_embeddings(embeddings, len(chunks))

    collection = _create_collection(database_path, collection_name)
    for start, end in _batched_indexes(len(chunks), CHROMA_UPSERT_BATCH_SIZE):
        collection.upsert(
            ids=chunk_ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    # Upsert prevents duplicates. Removing IDs absent from this complete run
    # also prevents old chunks from lingering after source or chunk-size edits.
    existing = collection.get()
    stale_ids = sorted(set(existing.get("ids", [])) - set(chunk_ids))
    if stale_ids:
        collection.delete(ids=stale_ids)

    collection_count = collection.count()
    if collection_count != len(chunks):
        raise ValueError(
            "Chroma collection count does not match the chunks created: "
            f"{collection_count} != {len(chunks)}."
        )

    stored = collection.get(ids=chunk_ids, include=["embeddings"])
    stored_ids = stored.get("ids", [])
    if len(stored_ids) != len(set(stored_ids)):
        raise ValueError("Duplicate chunk IDs were returned from Chroma.")
    if set(stored_ids) != set(chunk_ids):
        raise ValueError("Chroma is missing one or more indexed chunk IDs.")

    stored_embeddings = stored.get("embeddings")
    if stored_embeddings is None:
        raise ValueError("Chroma did not return stored embeddings for validation.")
    stored_dimension = _validate_embeddings(stored_embeddings, len(chunks))
    if stored_dimension != expected_dimension:
        raise ValueError(
            "Stored embedding dimensions differ from generated dimensions: "
            f"{stored_dimension} != {expected_dimension}."
        )

    return VectorIndexSummary(
        model_name=model_name,
        total_chunks_indexed=len(chunks),
        collection_count=collection_count,
        database_path=database_path.resolve(),
    )


def print_vector_index_summary(summary: VectorIndexSummary) -> None:
    """Print the required final vector-index report."""
    print("\n=== VECTOR INDEX SUMMARY ===")
    print(f"Embedding model: {summary.model_name}")
    print(f"Total chunks indexed: {summary.total_chunks_indexed}")
    print(f"Chroma collection count: {summary.collection_count}")
    print(f"Database path: {summary.database_path}")
