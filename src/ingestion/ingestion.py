import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# Configuration models
# =========================

@dataclass
class InputConfig:
    pdf_dir: Path
    glob_pattern: str = "*.pdf"
    recursive: bool = False


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] | None = None
    keep_separator: bool = True


@dataclass
class OutputConfig:
    output_dir: Path
    chunks_file: str = "chunks.jsonl"


@dataclass
class IngestionConfig:
    input: InputConfig
    chunking: ChunkingConfig
    output: OutputConfig


# =========================
# Config loading
# =========================

def load_yaml_config(config_path: str | Path) -> IngestionConfig:
    """
    Load ingestion configuration from a YAML file.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    IngestionConfig
        Parsed ingestion configuration.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg: dict[str, Any] = yaml.safe_load(f)

    input_cfg = raw_cfg.get("input", {})
    chunking_cfg = raw_cfg.get("chunking", {})
    output_cfg = raw_cfg.get("output", {})

    return IngestionConfig(
        input=InputConfig(
            pdf_dir=Path(input_cfg["pdf_dir"]),
            glob_pattern=input_cfg.get("glob_pattern", "*.pdf"),
            recursive=input_cfg.get("recursive", False),
        ),
        chunking=ChunkingConfig(
            chunk_size=chunking_cfg.get("chunk_size", 1000),
            chunk_overlap=chunking_cfg.get("chunk_overlap", 200),
            separators=chunking_cfg.get("separators"),
            keep_separator=chunking_cfg.get("keep_separator", True),
        ),
        output=OutputConfig(
            output_dir=Path(output_cfg["output_dir"]),
            chunks_file=output_cfg.get("chunks_file", "chunks.jsonl"),
        ),
    )


# =========================
# PDF discovery
# =========================

def discover_pdf_files(pdf_dir: Path, glob_pattern: str = "*.pdf", recursive: bool = False) -> list[Path]:
    """
    Discover PDF files inside a directory.

    Parameters
    ----------
    pdf_dir : Path
        Directory containing PDF files.
    glob_pattern : str
        Glob pattern used to match files.
    recursive : bool
        If True, scan subdirectories recursively.

    Returns
    -------
    list[Path]
        Sorted list of PDF file paths.
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    if not pdf_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {pdf_dir}")

    files = pdf_dir.rglob(glob_pattern) if recursive else pdf_dir.glob(glob_pattern)
    pdf_files = sorted([p for p in files if p.is_file()])

    return pdf_files


# =========================
# PDF loading
# =========================

def load_single_pdf(pdf_path: Path) -> list[Document]:
    """
    Load a single PDF into LangChain Document objects.

    Each page is typically returned as a separate Document.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.

    Returns
    -------
    list[Document]
        Documents extracted from the PDF.
    """
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    # Enrich metadata with a normalized source path
    for doc in documents:
        doc.metadata["source_file"] = pdf_path.name
        doc.metadata["source_path"] = str(pdf_path.resolve())

    return documents


def load_all_pdfs(pdf_files: list[Path]) -> list[Document]:
    """
    Load all PDFs from a list of file paths.

    Parameters
    ----------
    pdf_files : list[Path]
        PDF file paths.

    Returns
    -------
    list[Document]
        Aggregated list of loaded documents.
    """
    all_documents: list[Document] = []

    for pdf_file in pdf_files:
        docs = load_single_pdf(pdf_file)
        all_documents.extend(docs)

    return all_documents


# =========================
# Chunking
# =========================

def build_text_splitter(chunk_cfg: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    """
    Build a RecursiveCharacterTextSplitter from configuration.

    Parameters
    ----------
    chunk_cfg : ChunkingConfig
        Chunking configuration.

    Returns
    -------
    RecursiveCharacterTextSplitter
        Configured text splitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg.chunk_size,
        chunk_overlap=chunk_cfg.chunk_overlap,
        separators=chunk_cfg.separators,
        keep_separator=chunk_cfg.keep_separator,
    )


def chunk_documents(
    documents: list[Document],
    chunk_cfg: ChunkingConfig,
) -> list[Document]:
    """
    Split loaded documents into chunks.

    Parameters
    ----------
    documents : list[Document]
        Original documents loaded from PDFs.
    chunk_cfg : ChunkingConfig
        Chunking configuration.

    Returns
    -------
    list[Document]
        Chunked documents.
    """
    splitter = build_text_splitter(chunk_cfg)
    chunks = splitter.split_documents(documents)

    # Add a stable chunk id per output chunk
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    return chunks


# =========================
# Output serialization
# =========================

def save_chunks_to_jsonl(chunks: list[Document], output_path: Path) -> None:
    """
    Save chunked documents to JSONL format.

    Parameters
    ----------
    chunks : list[Document]
        Chunked documents.
    output_path : Path
        Target JSONL file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            row = {
                "text": chunk.page_content,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# =========================
# Pipeline
# =========================

def run_ingestion(config_path: str | Path = "config.ingestion.yaml") -> list[Document]:
    """
    Execute the full ingestion pipeline:
    1. Read config
    2. Discover PDFs
    3. Load documents
    4. Chunk documents
    5. Save chunks to disk

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    list[Document]
        Final chunked documents.
    """
    config = load_yaml_config(config_path)

    pdf_files = discover_pdf_files(
        pdf_dir=config.input.pdf_dir,
        glob_pattern=config.input.glob_pattern,
        recursive=config.input.recursive,
    )

    if not pdf_files:
        raise ValueError(f"No PDF files found in: {config.input.pdf_dir}")

    documents = load_all_pdfs(pdf_files)
    chunks = chunk_documents(documents, config.chunking)

    output_path = config.output.output_dir / config.output.chunks_file
    save_chunks_to_jsonl(chunks, output_path)

    print(f"Discovered PDF files: {len(pdf_files)}")
    print(f"Loaded documents/pages: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")
    print(f"Chunks saved to: {output_path}")

    return chunks


if __name__ == "__main__":
    actual_dir = Path(__file__).parent
    run_ingestion(actual_dir / "config.ingestion.yaml")