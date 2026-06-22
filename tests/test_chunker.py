import pytest
from src.services.chunker import split_pages


def make_page(page_number, text):
    """Helper to create a page dict for testing."""
    return {"page_number": page_number, "text": text}


def test_split_pages_basic():
    """
    split_pages should return chunks with text, page_number,
    and chunk_index for normal input.
    """
    pages = [
        make_page(1, "This is a test paragraph. " * 30),
        make_page(2, "Another paragraph here. " * 30),
    ]
    chunks = split_pages(pages)

    assert len(chunks) > 0
    for chunk in chunks:
        assert "text" in chunk
        assert "page_number" in chunk
        assert "chunk_index" in chunk
        assert len(chunk["text"]) > 20


def test_split_pages_empty_pages_skipped():
    """
    split_pages should skip empty pages without raising.
    Image-only pages produce empty text — should be handled cleanly.
    """
    pages = [
        make_page(1, ""),  # empty — should skip
        make_page(2, "Real content here. " * 30),  # has content
    ]
    chunks = split_pages(pages)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["page_number"] == 2


def test_split_pages_preserves_page_numbers():
    """
    Chunks should carry the correct page number from their source page.
    """
    pages = [
        make_page(5, "Content from page five. " * 30),
        make_page(9, "Content from page nine. " * 30),
    ]
    chunks = split_pages(pages)

    page_numbers = {chunk["page_number"] for chunk in chunks}
    assert 5 in page_numbers
    assert 9 in page_numbers


def test_split_pages_chunk_index_sequential():
    """
    chunk_index should be sequential across all pages,
    not reset per page.
    """
    pages = [
        make_page(1, "First page content. " * 30),
        make_page(2, "Second page content. " * 30),
    ]
    chunks = split_pages(pages)

    indices = [chunk["chunk_index"] for chunk in chunks]
    assert indices == list(range(len(chunks)))


def test_split_pages_filters_short_chunks():
    """
    Chunks shorter than 20 characters should be filtered out.
    """
    pages = [
        make_page(1, "Hi.\n\n" + "Real content here. " * 30),
    ]
    chunks = split_pages(pages)

    for chunk in chunks:
        assert len(chunk["text"].strip()) > 20
