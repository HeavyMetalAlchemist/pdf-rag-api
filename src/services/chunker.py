import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from opentelemetry import trace

from src.core.config import settings

tracer = trace.get_tracer(__name__)

MIN_CHARS_PER_PAGE = 50


def extract_pages(pdf_bytes):
    """
    Extracts text from each page individually.
    Returns a list of dicts with page number and text.
    Preserving page numbers enables meaningful citations later.

    Raises ValueError if PDF appears to be image-based.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = doc.page_count

    if page_count == 0:
        raise ValueError("PDF has no pages")

    pages = []
    total_chars = 0

    for i, page in enumerate(doc):
        page_text = page.get_text().strip()
        total_chars += len(page_text)
        pages.append(
            {
                "page_number": i + 1,
                "text": page_text,
            }
        )

    doc.close()

    avg_chars = total_chars / page_count
    if avg_chars < MIN_CHARS_PER_PAGE:
        raise ValueError(
            f"PDF appears to be image-based. "
            f"Average {avg_chars:.0f} chars per page, "
            f"minimum required is {MIN_CHARS_PER_PAGE}. "
            f"Only text-based PDFs are currently supported."
        )

    logger.info(
        f"Extracted {page_count} pages, "
        f"{total_chars} total chars, "
        f"{avg_chars:.0f} avg chars per page"
    )

    return pages


def split_pages(pages):
    """
    Splits each page's text into chunks, preserving the page number
    each chunk came from.

    Returns a list of dicts:
        {
            "text": "chunk text here",
            "page_number": 3,
            "chunk_index": 7   ← global index across all pages
        }

    Splitting per page rather than across the whole document means
    chunks never span two pages — a chunk always belongs to one page.
    This makes citations accurate.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = []
    global_index = 0

    for page in pages:
        if not page["text"]:
            # skip empty pages — blank pages, image-only pages
            continue

        page_chunks = splitter.split_text(page["text"])

        for chunk_text in page_chunks:
            chunk_text = chunk_text.strip()
            if len(chunk_text) <= 20:
                # skip artifacts — lone page numbers, headers, punctuation
                continue

            all_chunks.append(
                {
                    "text": chunk_text,
                    "page_number": page["page_number"],
                    "chunk_index": global_index,
                }
            )
            global_index += 1

    logger.info(f"Split into {len(all_chunks)} chunks across {len(pages)} pages")
    return all_chunks


def process_pdf(pdf_bytes):
    """
    Full PDF processing pipeline.
    Returns list of chunk dicts with text and page_number.
    This is the only function the ingest route needs to call.
    """
    with tracer.start_as_current_span("process_pdf") as span:
        span.set_attribute("pdf_size_bytes", len(pdf_bytes))

        pages = extract_pages(pdf_bytes)
        span.set_attribute("page_count", len(pages))

        chunks = split_pages(pages)
        span.set_attribute("chunk_count", len(chunks))

    return chunks
