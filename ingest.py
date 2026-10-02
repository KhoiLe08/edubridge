"""Download and clean openly licensed educational sources for EduBridge.

This module downloads, cleans, validates, and chunks plain text. Embeddings,
vector storage, and LLM calls belong to later stages of the RAG pipeline.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from vector_store import (
    DATABASE_PATH,
    EMBEDDING_MODEL_NAME,
    VectorIndexSummary,
    index_chunks,
    print_vector_index_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parent
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.json"
REQUEST_TIMEOUT = (10, 30)  # Separate connection and response-read timeouts.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/129.0.0.0 Safari/537.36 EduBridge/1.0"
)

# Only these Creative Commons families and versions are approved. Public
# Domain remains allowed under the project's existing licensing policy.
ALLOWED_CC_FAMILIES = ("CC0", "CC BY", "CC BY-SA", "CC BY-NC", "CC BY-NC-SA")
ALLOWED_CC_VERSIONS = {"1.0", "2.0", "2.5", "3.0", "4.0"}
CC_FAMILY_PATTERN = re.compile(
    r"\b(?:CC0|CC\s+BY(?:\s*-\s*NC\s*-\s*SA|\s*-\s*NC|\s*-\s*SA)?)(?!\s*-)",
)


def load_sources(metadata_path: Path = METADATA_PATH) -> list[dict[str, Any]]:
    """Load source records from the metadata JSON file.

    The preferred format is a JSON array. A top-level object containing a
    ``sources`` array is also accepted for convenience.
    """
    try:
        raw_metadata = metadata_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read metadata file: {exc}") from exc

    if not raw_metadata.strip():
        raise ValueError(f"Metadata file is empty: {metadata_path}")

    try:
        data = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Metadata file contains invalid JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc

    if isinstance(data, dict):
        data = data.get("sources")

    if not isinstance(data, list):
        raise ValueError(
            "Metadata must be a JSON array or an object with a 'sources' array."
        )

    sources: list[dict[str, Any]] = []
    for index, source in enumerate(data, start=1):
        if not isinstance(source, dict):
            print(f"WARNING: Skipping metadata item {index}; it is not an object.")
            continue
        sources.append(source)

    return sources


def validate_license(
    source: dict[str, Any], errors: list[str] | None = None
) -> bool:
    """Return True only when a source declares an approved reusable license.

    License text is normalized before its family is detected, so normal
    version and jurisdiction suffixes do not cause otherwise valid Creative
    Commons licenses to be rejected.
    """
    source_id = source.get("id", "<missing id>")
    license_name = source.get("license")

    if not isinstance(license_name, str) or not license_name.strip():
        reason = "license is missing"
        if errors is not None:
            errors.append(reason)
        print(f"WARNING: Skipping source {source_id}: {reason}.")
        return False

    normalized_license = " ".join(license_name.upper().split())

    # Explicitly disallow restrictive or conflicting license declarations.
    if (
        "ND" in normalized_license
        or "NODERIVATIVES" in normalized_license
        or "ALL RIGHTS RESERVED" in normalized_license
    ):
        reason = f"license '{license_name}' does not permit the required reuse"
        if errors is not None:
            errors.append(reason)
        print(f"WARNING: Skipping source {source_id}: {reason}.")
        return False

    if normalized_license == "PUBLIC DOMAIN":
        return True

    family_match = CC_FAMILY_PATTERN.search(normalized_license)
    if family_match is None:
        reason = f"license '{license_name}' is unknown or not allowed"
        if errors is not None:
            errors.append(reason)
        print(
            f"WARNING: Skipping source {source_id}: {reason}. "
            f"Allowed families: {', '.join(ALLOWED_CC_FAMILIES)} and Public Domain."
        )
        return False

    # When a numeric version is supplied after the family, it must be one of
    # the established CC versions supported by this project.
    suffix = normalized_license[family_match.end() :]
    version_match = re.search(r"\b\d+(?:\.\d+)?\b", suffix)
    if version_match and version_match.group() not in ALLOWED_CC_VERSIONS:
        reason = f"license version '{version_match.group()}' is not allowed"
        if errors is not None:
            errors.append(reason)
        print(f"WARNING: Skipping source {source_id}: {reason}.")
        return False

    return True


def _is_http_url(value: Any) -> bool:
    """Check that a value is a complete HTTP or HTTPS URL."""
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def download_html(source_url: str, errors: list[str] | None = None) -> str | None:
    """Download one HTML page, returning None when the request fails."""
    try:
        response = requests.get(
            source_url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        return response.text
    except requests.RequestException as exc:
        if errors is not None:
            errors.append(f"download failed: {exc}")
        print(f"WARNING: Could not download {source_url}: {exc}")
        return None


def clean_html(html: str) -> str:
    """Remove page chrome and convert the main document content to plain text."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove executable, decorative, and navigational page elements.
    for element in soup.find_all(
        ["script", "style", "nav", "header", "footer", "aside"]
    ):
        element.decompose()

    # Prefer semantic content containers, with progressively broader fallbacks.
    content = soup.find("main") or soup.find("article") or soup.body
    if content is None:
        content = soup

    # Collapse spaces within each line and remove empty lines while retaining
    # line boundaries between headings, paragraphs, and list items.
    lines = []
    for line in content.get_text(separator="\n").splitlines():
        cleaned_line = " ".join(line.split())
        if cleaned_line:
            lines.append(cleaned_line)
    return "\n".join(lines)


def assess_content_quality(text: str) -> str:
    """Classify extracted text by its character count."""
    character_count = len(text)
    if character_count >= 2500:
        return "GOOD"
    if character_count >= 1000:
        return "SHORT"
    return "INVALID"


def looks_like_index_page(text: str) -> bool:
    """Detect common signs of chapter indexes or summary/listing pages."""
    normalized_text = text.casefold()
    if normalized_text.count("this page covers") >= 3:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_lines = [line for line in lines if len(line) <= 120]
    numbered_heading_pattern = re.compile(
        r"^(?:chapter\s+)?\d+(?:\.\d+)*(?:[.)\-:]|\s)\s*\S+",
        re.IGNORECASE,
    )
    numbered_headings = [
        line
        for line in short_lines
        if numbered_heading_pattern.match(line) and len(line.split()) <= 14
    ]
    if len(numbered_headings) >= 5 and len(numbered_headings) >= len(lines) / 2:
        return True

    # Repeated card descriptions or headings are another common index signal.
    repeated_lines = Counter(
        " ".join(line.casefold().split()) for line in lines if len(line) >= 20
    )
    return sum(count - 1 for count in repeated_lines.values() if count >= 3) >= 3


def _print_debug_summary(
    source: dict[str, Any], text: str, quality: str
) -> None:
    """Print source metadata, content quality, and a short text preview."""
    print("\n" + "=" * 80)
    print(f"Source ID: {source.get('id', '')}")
    print(f"Title: {source.get('title', '')}")
    print(f"License: {source.get('license', '')}")
    print(f"Extracted characters: {len(text)}")
    print(f"Content quality: {quality}")
    print("Text preview (first 500 characters):")
    print(text[:500])


def process_source(
    source: dict[str, Any],
    failures: list[tuple[str, str]] | None = None,
    warnings: list[tuple[str, str]] | None = None,
) -> dict[str, Any] | None:
    """Validate, download, and clean one source while preserving its metadata."""
    source_id = str(source.get("id", "<missing id>"))

    def record_failure(reason: str) -> None:
        if failures is not None:
            failures.append((source_id, reason))

    def record_warning(warning: str) -> None:
        if warnings is not None:
            warnings.append((source_id, warning))

    # License validation deliberately happens before any download attempt.
    license_errors: list[str] = []
    if not validate_license(source, license_errors):
        record_failure(license_errors[-1])
        return None

    source_url = source.get("source_url")
    if not _is_http_url(source_url):
        reason = "source_url is missing or is not a valid HTTP(S) URL"
        print(
            f"WARNING: Skipping source {source_id}: {reason}."
        )
        record_failure(reason)
        return None

    download_errors: list[str] = []
    html = download_html(source_url.strip(), download_errors)
    if html is None:
        record_failure(download_errors[-1] if download_errors else "download failed")
        return None

    text = clean_html(html)
    quality = assess_content_quality(text)
    suspected_index = looks_like_index_page(text)

    _print_debug_summary(source, text, quality)

    if suspected_index:
        warning = "source may be an index/summary page"
        print(f"WARNING: {warning}")
        record_warning(warning)

    if quality == "SHORT":
        record_warning("content quality is SHORT")
    elif quality == "INVALID":
        reason = "content quality is INVALID (fewer than 1000 characters)"
        print(f"WARNING: Skipping source {source_id}: {reason}.")
        record_failure(reason)
        return None

    # Copy every original field, including attribution and license metadata,
    # then attach the cleaned text for use by a later pipeline stage.
    processed_source = dict(source)
    processed_source["text"] = text
    processed_source["quality"] = quality
    return processed_source


def chunk_source(
    source: dict[str, Any],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Split a processed source into overlapping, metadata-rich text chunks.

    Chunk boundaries prefer paragraph, line, sentence, and word boundaries in
    that order. Every chunk retains the original source metadata so later RAG
    stages can provide attribution without looking the source up again.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between zero and chunk_size")

    text = source.get("text", "")
    if not isinstance(text, str) or not text.strip():
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    text_length = len(text)
    minimum_boundary = max(chunk_size // 2, 1)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # Avoid cutting through a natural text unit when a useful boundary is
        # available in the latter half of the proposed chunk.
        if end < text_length:
            search_start = start + minimum_boundary
            boundary_candidates = (
                text.rfind("\n\n", search_start, end),
                text.rfind("\n", search_start, end),
                text.rfind(". ", search_start, end),
                text.rfind(" ", search_start, end),
            )
            boundary = max(boundary_candidates)
            if boundary >= search_start:
                end = boundary + (2 if text[boundary : boundary + 2] == ". " else 1)

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk = {key: value for key, value in source.items() if key != "text"}
            chunk_index = len(chunks)
            chunk["chunk_id"] = f"{source.get('id', 'source')}-chunk-{chunk_index + 1:04d}"
            chunk["chunk_index"] = chunk_index
            chunk["text"] = chunk_text
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(end - chunk_overlap, start + 1)
        while next_start < end and text[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks


def _print_chunk_statistics(
    source: dict[str, Any], chunks: list[dict[str, Any]]
) -> None:
    """Print chunk counts, lengths, and previews for one processed source."""
    chunk_lengths = [len(chunk["text"]) for chunk in chunks]
    average_length = sum(chunk_lengths) / len(chunk_lengths) if chunks else 0
    minimum_length = min(chunk_lengths, default=0)
    maximum_length = max(chunk_lengths, default=0)

    print("\n--- CHUNK STATISTICS ---")
    print(f"Source ID: {source.get('id', '')}")
    print(f"Original character count: {len(source.get('text', ''))}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Average chunk length: {average_length:.1f}")
    print(f"Minimum chunk length: {minimum_length}")
    print(f"Maximum chunk length: {maximum_length}")

    for preview_index, chunk in enumerate(chunks[:2], start=1):
        print(f"First 300 characters of chunk {preview_index}:")
        print(chunk["text"][:300])


def _print_ingestion_summary(
    successful: list[dict[str, Any]],
    failed: list[tuple[str, str]],
    warnings: list[tuple[str, str]],
) -> None:
    """Print a compact report of successes, failures, and warnings."""
    print("\n=== INGESTION SUMMARY ===")

    print("\nSuccessful sources:")
    print("ID | topic | character count | quality")
    if successful:
        for source in successful:
            print(
                f"{source.get('id', '')} | {source.get('topic', '')} | "
                f"{len(source['text'])} | {source['quality']}"
            )
    else:
        print("(none)")

    print("\nFailed sources:")
    print("ID | reason")
    if failed:
        for source_id, reason in failed:
            print(f"{source_id} | {reason}")
    else:
        print("(none)")

    print("\nWarnings:")
    print("ID | warning")
    if warnings:
        for source_id, warning in warnings:
            print(f"{source_id} | {warning}")
    else:
        print("(none)")


def _print_chunking_summary(
    successful_sources: list[dict[str, Any]],
    all_chunks: list[dict[str, Any]],
) -> None:
    """Print overall chunk totals grouped by source topic."""
    source_counts: Counter[str] = Counter(
        str(source.get("topic", "")) for source in successful_sources
    )
    chunk_counts: Counter[str] = Counter(
        str(chunk.get("topic", "")) for chunk in all_chunks
    )
    topics = sorted(source_counts.keys() | chunk_counts.keys())

    print("\n=== CHUNKING SUMMARY ===")
    print(f"Total sources processed: {len(successful_sources)}")
    print(f"Total chunks created: {len(all_chunks)}")
    print("\ntopic | sources | chunks")
    if topics:
        for topic in topics:
            print(f"{topic} | {source_counts[topic]} | {chunk_counts[topic]}")
    else:
        print("(none)")


def main() -> list[dict[str, Any]]:
    """Process sources, create chunks, and persist their vector index."""
    try:
        sources = load_sources()
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}")
        _print_ingestion_summary([], [("metadata", str(exc))], [])
        _print_chunking_summary([], [])
        print_vector_index_summary(
            VectorIndexSummary(
                model_name=EMBEDDING_MODEL_NAME,
                total_chunks_indexed=0,
                collection_count="not available",
                database_path=DATABASE_PATH,
            )
        )
        return []

    processed_sources: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] = []
    failed_sources: list[tuple[str, str]] = []
    ingestion_warnings: list[tuple[str, str]] = []
    for source in sources:
        try:
            processed = process_source(
                source,
                failures=failed_sources,
                warnings=ingestion_warnings,
            )
            if processed is None:
                continue

            # Chunking is part of the normal ingestion flow, not merely a
            # helper defined for callers elsewhere in the application.
            source_chunks = chunk_source(processed)
            _print_chunk_statistics(processed, source_chunks)
            all_chunks.extend(source_chunks)
            processed_sources.append(processed)
        except Exception as exc:
            # A malformed page or unexpected per-source error must not prevent
            # the remaining metadata records from being processed.
            source_id = str(source.get("id", "<missing id>"))
            reason = f"unexpected processing error: {exc}"
            print(f"WARNING: Skipping source {source_id}: {reason}")
            failed_sources.append((source_id, reason))
            continue

    try:
        vector_summary = index_chunks(all_chunks)
    except Exception as exc:
        print(f"ERROR: Vector indexing failed: {exc}")
        vector_summary = VectorIndexSummary(
            model_name=EMBEDDING_MODEL_NAME,
            total_chunks_indexed=0,
            collection_count="not available",
            database_path=DATABASE_PATH.resolve(),
        )

    _print_ingestion_summary(
        processed_sources, failed_sources, ingestion_warnings
    )
    _print_chunking_summary(processed_sources, all_chunks)
    print_vector_index_summary(vector_summary)
    return processed_sources


if __name__ == "__main__":
    main()
