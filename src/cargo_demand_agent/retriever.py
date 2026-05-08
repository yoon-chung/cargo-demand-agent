"""
Retriever: hybrid (BGE-M3 dense / Chroma + BM25 sparse / Kiwi), fused via RRF.

Dense covers paraphrased queries; sparse rescues short keyword queries (PMI,
FBX) where dense distance is washed out by surrounding context. The Chroma
index lives at data/chroma_db_{LOCALE}/, one per language (ko, en).
"""
from functools import lru_cache
from pathlib import Path
import logging
import shutil

import torch
from kiwipiepy import Kiwi
from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.cargo_demand_agent.i18n import LOCALE
from src.cargo_demand_agent.loader import load_documents, split_documents

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
# Project root resolved from this file's location, not CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHROMA_ROOT = _PROJECT_ROOT / "data"
DEFAULT_K = 4
DEFAULT_FETCH_K = 20
DEFAULT_DENSE_WEIGHT = 0.5
DEFAULT_SPARSE_WEIGHT = 0.5


@lru_cache(maxsize=1)
def _get_kiwi() -> Kiwi:
    """Lazy-init Kiwi morphological analyzer (loaded once per process)."""
    return Kiwi()


def _korean_tokenize(text: str) -> list[str]:
    """
    Tokenize Korean text into morphemes using Kiwi.

    Used as BM25Retriever's preprocess_func so that both indexed chunks and
    user queries are tokenized identically. Strips Korean particles
    (조사) so 'PMI 수요에' matches 'PMI 수요' in target chunks.
    """
    return [token.form for token in _get_kiwi().tokenize(text)]


def _default_persist_directory() -> Path:
    """Locale-scoped Chroma directory: data/chroma_db_{ko|en}/."""
    return DEFAULT_CHROMA_ROOT / f"chroma_db_{LOCALE}"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Load (and cache) the BGE-M3 embedding model.

    First call downloads ~2GB to the HuggingFace cache. Subsequent calls
    in the same process return the cached instance instantly.

    Uses CUDA if available, else CPU.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Loading embedding model %s on %s", EMBEDDING_MODEL_NAME, device
    )
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_index(
    chunks: list[Document],
    persist_directory: Path | None = None,
) -> Chroma:
    """
    Build a fresh Chroma index from chunks.

    If `persist_directory` already exists, it is wiped and rebuilt. This
    keeps the index in lockstep with the source documents (no stale chunks
    from earlier loader runs).
    """
    if persist_directory is None:
        persist_directory = _default_persist_directory()

    if persist_directory.exists():
        logger.info("Removing existing index at %s", persist_directory)
        shutil.rmtree(persist_directory)

    persist_directory.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Building index at %s with %d chunks", persist_directory, len(chunks)
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(persist_directory),
    )
    logger.info("Index built. count=%d", vectorstore._collection.count())
    return vectorstore


def get_retriever(
    persist_directory: Path | None = None,
    k: int = DEFAULT_K,
    fetch_k: int = DEFAULT_FETCH_K,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    sparse_weight: float = DEFAULT_SPARSE_WEIGHT,
) -> EnsembleRetriever:
    """
    Hybrid retriever: BGE-M3 dense (Chroma + MMR) + BM25 sparse, fused via RRF.
    Dense handles paraphrased queries; sparse rescues short keyword queries
    (PMI, FBX) where dense distance is washed out by surrounding context.
    """
    if persist_directory is None:
        persist_directory = _default_persist_directory()

    if not persist_directory.exists():
        raise FileNotFoundError(
            f"Chroma index not found at {persist_directory}. "
            "Build it first via loader.py (`python -m src.cargo_demand_agent.loader`)."
        )

    # Dense branch: BGE-M3 -> Chroma -> MMR
    vectorstore = Chroma(
        persist_directory=str(persist_directory),
        embedding_function=get_embeddings(),
    )
    dense_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k},
    )

    # Sparse branch: BM25 over the same chunks, rebuilt in-memory.
    # Kiwi morpheme tokenizer applied to both docs and queries so that
    # Korean particles (조사) don't break keyword matches.
    chunks = split_documents(load_documents())
    bm25_retriever = BM25Retriever.from_documents(
        chunks,
        preprocess_func=_korean_tokenize,
    )
    bm25_retriever.k = k

    return EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[dense_weight, sparse_weight],
    )
