"""
Loader: read report markdowns into LangChain Documents and split into chunks.

Reports live under data/reports/{LOCALE}/*.md. Each file has YAML frontmatter
(title, report_type, domain, period, route, etc.). The body becomes
Document.page_content; frontmatter populates Document.metadata after
Chroma-friendly normalization.

Public API:
    load_documents(reports_dir) -> list[Document]
    split_documents(docs, chunk_size, chunk_overlap) -> list[Document]
"""
from pathlib import Path
from typing import Any
import logging

import frontmatter
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.cargo_demand_agent.i18n import LOCALE

logger = logging.getLogger(__name__)

# Project root resolved from this file's location, not CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORTS_ROOT = _PROJECT_ROOT / "data" / "reports"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100


def _normalize_metadata(
    metadata: dict[str, Any],
) -> dict[str, str | int | float | bool]:
    """
    Coerce metadata values to scalar types Chroma accepts.

    Chroma rejects list/dict/None metadata values. Lists become '|'-joined
    strings; None is dropped; other non-scalar types fall back to str().
    """
    normalized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, list):
            normalized[key] = "|".join(str(item) for item in value)
        elif isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        else:
            normalized[key] = str(value)
    return normalized


def load_documents(reports_dir: Path | None = None) -> list[Document]:
    """
    Load every .md file from the locale-specific reports directory.

    Two metadata keys are injected on top of the frontmatter:
        - "source":   absolute file path (for citation traces in the final answer)
        - "language": active LOCALE (for filtering across multilingual indices)

    Args:
        reports_dir: Override the default. If None, uses
                     DEFAULT_REPORTS_ROOT / LOCALE.

    Raises:
        FileNotFoundError: directory missing or contains no .md files.
    """
    if reports_dir is None:
        reports_dir = DEFAULT_REPORTS_ROOT / LOCALE

    if not reports_dir.is_dir():
        raise FileNotFoundError(f"Reports directory not found: {reports_dir}")

    documents: list[Document] = []
    for md_file in sorted(reports_dir.glob("*.md")):
        with md_file.open(encoding="utf-8") as f:
            post = frontmatter.load(f)
        metadata = _normalize_metadata(post.metadata)
        metadata["source"] = str(md_file)
        metadata["language"] = LOCALE
        documents.append(
            Document(page_content=post.content, metadata=metadata)
        )
        logger.info("Loaded %s (%d chars)", md_file.name, len(post.content))

    if not documents:
        raise FileNotFoundError(f"No .md files found under {reports_dir}")

    return documents


def split_documents(
    docs: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split Documents into smaller chunks for embedding.

    Metadata (including 'source' and 'language') propagates to every chunk,
    so citation traces survive splitting.

    chunk_size is measured in CHARACTERS, not tokens (LangChain's default).
    800 chars roughly equals one paragraph in our Korean reports.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    logger.info(
        "Split %d documents -> %d chunks (size=%d, overlap=%d)",
        len(docs),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


if __name__ == "__main__":
    # CLI entry point: rebuild the Chroma index from scratch.
    # Usage: python -m src.cargo_demand_agent.loader
    from src.cargo_demand_agent.retriever import build_index

    logging.basicConfig(
        level=logging.INFO,
        format="[%(name)s] %(message)s",
    )
    docs = load_documents()
    chunks = split_documents(docs)
    build_index(chunks)
    logger.info("Indexing complete.")
