import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple
import re
from copy import deepcopy
from collections import Counter, defaultdict
import yaml
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# Configuration models
# =========================

@dataclass
class InputConfig:
    pdf_dir: Path = Path("data/raw")
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
    output_dir: Path = Path("data/processed")
    version: str = "v1"
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
            version=output_cfg.get("version", "v1"),
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
# Docs cleaning
# =========================

def normalize_whitespace(text: str) -> str:
    """
    Normalize spaces and line endings.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[ \u00A0]+", " ", text)
    return text


def is_separator_line(line: str) -> bool:
    """
    Return True if the line is mostly made of separators or punctuation.
    """
    stripped = line.strip()

    if not stripped:
        return True

    if re.fullmatch(r"[.\-_=•·\s]{3,}", stripped):
        return True

    return False


def is_page_number_line(line: str) -> bool:
    """
    Return True if the line looks like a page number.
    """
    stripped = line.strip()

    patterns = [
        r"^\d+$",
        r"^page\s+\d+$",
        r"^pagina\s+\d+$",
        r"^\d+\s*/\s*\d+$",
        r"^-\s*\d+\s*-$",
    ]

    return any(re.fullmatch(pattern, stripped, flags=re.IGNORECASE) for pattern in patterns)


def is_toc_line(line: str) -> bool:
    """
    Return True if the line looks like a table-of-contents line,
    e.g. 'Introduction ................ 12'
    """
    stripped = line.strip()

    patterns = [
        r"^.{1,200}\.{4,}\s*\d+\s*$",
        r"^.{1,200}\s{2,}\d+\s*$",
    ]

    return any(re.fullmatch(pattern, stripped) for pattern in patterns)


def line_alpha_ratio(line: str) -> float:
    """
    Ratio of alphabetic characters over non-space characters.
    """
    non_space = [c for c in line if not c.isspace()]
    if not non_space:
        return 0.0

    alpha = sum(c.isalpha() for c in non_space)
    return alpha / len(non_space)


def is_low_information_line(line: str) -> bool:
    """
    Heuristic to remove very poor lines.
    """
    stripped = line.strip()

    if not stripped:
        return True

    if len(stripped) <= 2:
        return True

    # Remove lines with no letters and very short content
    if sum(c.isalpha() for c in stripped) == 0 and len(stripped) < 20:
        return True

    # Very low alphabetic density often means noise
    if len(stripped) < 12 and line_alpha_ratio(stripped) < 0.3:
        return True

    return False


def clean_line(line: str) -> str:
    """
    Clean a single line.
    """
    line = normalize_whitespace(line)
    line = line.strip()

    # Remove spaces before punctuation
    line = re.sub(r"\s+([,.;:!?])", r"\1", line)

    # Collapse multiple spaces again after punctuation fixes
    line = re.sub(r"\s{2,}", " ", line)

    return line


def detect_repeated_header_footer(
    pages_lines: List[List[str]],
    max_candidates_per_page: int = 3,
    min_repetition_ratio: float = 0.4,
) -> Tuple[set, set]:
    """
    Detect repeated header/footer lines across pages.

    Strategy:
    - collect first N non-empty lines of each page as header candidates
    - collect last N non-empty lines of each page as footer candidates
    - lines repeated on many pages are treated as headers/footers
    """
    header_candidates = []
    footer_candidates = []

    total_pages = len(pages_lines)
    if total_pages == 0:
        return set(), set()

    for lines in pages_lines:
        non_empty = [clean_line(line) for line in lines if clean_line(line)]

        header_candidates.extend(non_empty[:max_candidates_per_page])
        footer_candidates.extend(non_empty[-max_candidates_per_page:])

    header_counts = Counter(header_candidates)
    footer_counts = Counter(footer_candidates)

    min_count = max(2, int(total_pages * min_repetition_ratio))

    headers = {
        line for line, count in header_counts.items()
        if count >= min_count and len(line) > 2
    }

    footers = {
        line for line, count in footer_counts.items()
        if count >= min_count and len(line) > 2
    }

    return headers, footers


def preprocess_page_lines(page_text: str) -> List[str]:
    """
    Split page text into raw lines and normalize them.
    """
    lines = page_text.split("\n")
    lines = [normalize_whitespace(line).strip() for line in lines]
    return lines


def should_drop_line(line: str, headers: set, footers: set) -> bool:
    """
    Decide whether a line should be removed.
    """
    cleaned = clean_line(line)

    if not cleaned:
        return True

    if cleaned in headers or cleaned in footers:
        return True

    if is_separator_line(cleaned):
        return True

    if is_page_number_line(cleaned):
        return True

    if is_toc_line(cleaned):
        return True

    if is_low_information_line(cleaned):
        return True

    return False


def merge_lines_into_paragraphs(lines: List[str]) -> str:
    """
    Merge cleaned lines into paragraphs.

    Heuristics:
    - keep blank lines as paragraph boundaries
    - join broken lines inside a paragraph
    - preserve likely headings as standalone lines
    """
    paragraphs = []
    current = []

    def flush_current() -> None:
        nonlocal current
        if current:
            paragraph = " ".join(current)
            paragraph = re.sub(r"\s{2,}", " ", paragraph).strip()
            if paragraph:
                paragraphs.append(paragraph)
            current = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            flush_current()
            continue

        # Likely heading: short line, many uppercase letters, or ends without sentence punctuation
        is_heading = (
            len(stripped) < 80
            and (
                stripped.isupper()
                or re.fullmatch(r"^\d+(\.\d+)*\s+.+$", stripped) is not None
            )
        )

        if is_heading:
            flush_current()
            paragraphs.append(stripped)
            continue

        if current:
            prev = current[-1]

            # If previous line ends with hyphen, merge without space
            if prev.endswith("-"):
                current[-1] = prev[:-1] + stripped
            else:
                current.append(stripped)
        else:
            current.append(stripped)

    flush_current()

    return "\n\n".join(paragraphs)

def clean_pdf_pages(raw_pages: List[Document]) -> List[Document]:
    """
    Clean PDF-extracted text while preserving metadata.

    Cleaning is performed per source document, so repeated headers/footers
    are detected within each PDF only.
    """
    pages_by_source: dict[str, list[Document]] = defaultdict(list)

    for doc in raw_pages:
        source_key = doc.metadata.get("source_path", doc.metadata.get("source_file", "unknown"))
        pages_by_source[source_key].append(doc)

    cleaned_documents: List[Document] = []

    for _, source_pages in pages_by_source.items():
        source_pages = sorted(source_pages, key=lambda d: d.metadata.get("page", 0))

        pages_lines: List[List[str]] = [
            preprocess_page_lines(doc.page_content) for doc in source_pages
        ]

        headers, footers = detect_repeated_header_footer(pages_lines)

        for doc, page_lines in zip(source_pages, pages_lines):
            cleaned_lines = []

            for line in page_lines:
                if should_drop_line(line, headers, footers):
                    continue

                cleaned = clean_line(line)
                if cleaned:
                    cleaned_lines.append(cleaned)

            page_text = merge_lines_into_paragraphs(cleaned_lines).strip()

            if not page_text:
                continue

            new_metadata = deepcopy(doc.metadata)
            new_metadata["is_cleaned"] = True
            new_metadata["original_char_count"] = len(doc.page_content)
            new_metadata["cleaned_char_count"] = len(page_text)

            cleaned_documents.append(
                Document(
                    page_content=page_text,
                    metadata=new_metadata,
                )
            )

    return cleaned_documents

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

def find_page_for_offset(page_spans: list[dict[str, int]], offset: int) -> int | None:
    """
    Return the page number containing the given character offset.
    """
    for span in page_spans:
        if span["start_char"] <= offset < span["end_char"]:
            return span["page"]

    if page_spans and offset == page_spans[-1]["end_char"]:
        return page_spans[-1]["page"]

    return None

def chunk_merged_documents(
    documents: list[Document],
    chunk_cfg: ChunkingConfig,
) -> list[Document]:
    """
    Split merged document-level texts into chunks and enrich each chunk
    with page provenance metadata (start_page, end_page, pages).
    """
    splitter = build_text_splitter(chunk_cfg)
    all_chunks: list[Document] = []
    global_chunk_id = 0

    for doc in documents:
        source_text = doc.page_content
        source_metadata = deepcopy(doc.metadata)
        page_spans = source_metadata.get("page_spans", [])

        split_texts = splitter.split_text(source_text)

        search_start = 0

        for chunk_text in split_texts:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            start_offset = source_text.find(chunk_text, search_start)

            if start_offset == -1:
                # Fallback: search from beginning
                start_offset = source_text.find(chunk_text)

            if start_offset == -1:
                # Last-resort fallback if exact text cannot be found
                start_offset = search_start

            end_offset = start_offset + len(chunk_text)

            start_page = find_page_for_offset(page_spans, start_offset)
            end_page = find_page_for_offset(page_spans, max(start_offset, end_offset - 1))

            pages = []
            if start_page is not None and end_page is not None:
                pages = list(range(start_page, end_page + 1))

            chunk_metadata = deepcopy(source_metadata)
            chunk_metadata["chunk_id"] = global_chunk_id
            chunk_metadata["chunk_start_char"] = start_offset
            chunk_metadata["chunk_end_char"] = end_offset
            chunk_metadata["start_page"] = start_page
            chunk_metadata["end_page"] = end_page
            chunk_metadata["pages"] = pages

            all_chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata,
                )
            )

            global_chunk_id += 1
            search_start = max(start_offset + 1, end_offset - chunk_cfg.chunk_overlap)

    return all_chunks


def merge_cleaned_pages_by_document(cleaned_pages: list[Document]) -> list[Document]:
    """
    Merge cleaned page-level documents into one document per source file,
    while preserving a page-to-character-span mapping.

    Output metadata includes:
    - source_file
    - source_path
    - total_pages
    - page_spans: list of dicts with page/start_char/end_char
    """
    pages_by_source: dict[str, list[Document]] = defaultdict(list)

    for doc in cleaned_pages:
        source_key = doc.metadata.get("source_path", doc.metadata.get("source_file", "unknown"))
        pages_by_source[source_key].append(doc)

    merged_documents: list[Document] = []

    for _, pages in pages_by_source.items():
        # Sort pages by original page index if available
        pages = sorted(pages, key=lambda d: d.metadata.get("page", 0))

        merged_parts: list[str] = []
        page_spans: list[dict[str, int]] = []
        cursor = 0

        for idx, page_doc in enumerate(pages):
            page_text = page_doc.page_content.strip()
            if not page_text:
                continue

            if merged_parts:
                separator = "\n\n"
                cursor += len(separator)
                merged_parts.append(separator)

            start_char = cursor
            merged_parts.append(page_text)
            cursor += len(page_text)
            end_char = cursor

            page_spans.append(
                {
                    "page": int(page_doc.metadata.get("page", idx)),
                    "start_char": start_char,
                    "end_char": end_char,
                }
            )

        merged_text = "".join(merged_parts).strip()

        if not merged_text:
            continue

        first_meta = deepcopy(pages[0].metadata)

        merged_metadata = {
            "source_file": first_meta.get("source_file"),
            "source_path": first_meta.get("source_path"),
            "is_cleaned": True,
            "is_merged_document": True,
            "merged_page_count": len(page_spans),
            "original_total_pages": max(
                int(p.metadata.get("page", 0)) for p in pages
            ) + 1 if pages else 0,
            "page_spans": page_spans,
        }

        merged_documents.append(
            Document(
                page_content=merged_text,
                metadata=merged_metadata,
            )
        )

    return merged_documents


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
    4. Clean page-level text
    5. Merge cleaned pages by source document
    6. Chunk merged documents
    7. Save chunks to disk

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

    raw_pages = load_all_pdfs(pdf_files)
    clean_pages = clean_pdf_pages(raw_pages)
    merged_documents = merge_cleaned_pages_by_document(clean_pages)
    chunks = chunk_merged_documents(merged_documents, config.chunking)

    output_path = config.output.output_dir / config.output.version / config.output.chunks_file
    save_chunks_to_jsonl(chunks, output_path)

    print(f"Discovered PDF files: {len(pdf_files)}")
    print(f"Loaded raw pages: {len(raw_pages)}")
    print(f"Cleaned pages: {len(clean_pages)}")
    print(f"Merged documents: {len(merged_documents)}")
    print(f"Generated chunks: {len(chunks)}")
    print(f"Chunks saved to: {output_path}")

    return chunks


if __name__ == "__main__":
    actual_dir = Path(__file__).parent
    run_ingestion(actual_dir / "config.ingestion.yaml")