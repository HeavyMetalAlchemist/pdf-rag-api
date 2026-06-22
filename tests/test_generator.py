import pytest
from src.services.generator import (
    extract_citations,
    build_context_block,
    parse_sources,
)


# ── extract_citations ────────────────────────────────────────────────────────


def test_extract_citations_single():
    """Single citation on its own line."""
    text = "Some answer.\n\nSources:\n[Source 1 — doc.pdf, page 3]"
    citations = extract_citations(text)
    assert len(citations) == 1
    assert "Source 1" in citations[0]


def test_extract_citations_multiple_same_line():
    """Multiple citations on one line — Claude's inconsistent format."""
    text = (
        "Answer here.\n\nSources: "
        "[Source 1 — doc.pdf, page 3] "
        "[Source 2 — doc.pdf, page 7]"
    )
    citations = extract_citations(text)
    assert len(citations) == 2


def test_extract_citations_multiple_separate_lines():
    """Multiple citations each on their own line."""
    text = (
        "Answer here.\n\nSources:\n"
        "[Source 1 — doc.pdf, page 3]\n"
        "[Source 2 — doc.pdf, page 7]\n"
        "[Source 3 — doc.pdf, page 12]"
    )
    citations = extract_citations(text)
    assert len(citations) == 3


def test_extract_citations_ignores_non_source_brackets():
    """
    Brackets that don't start with Source should not be extracted.
    Handles cases where Claude uses brackets for other purposes.
    """
    text = "Answer [important note] here.\n\nSources:\n[Source 1 — doc.pdf, page 3]"
    citations = extract_citations(text)
    assert len(citations) == 1
    assert citations[0].startswith("Source")


def test_extract_citations_empty_text():
    """Empty input should return empty list without raising."""
    assert extract_citations("") == []
    assert extract_citations(None) == []


def test_extract_citations_no_citations():
    """Answer with no citations should return empty list."""
    text = "I cannot find the answer to this question in the provided documents."
    citations = extract_citations(text)
    assert citations == []


# ── build_context_block ──────────────────────────────────────────────────────


def make_chunk(filename, page_number, text, distance=0.5):
    """Helper to create a retrieved chunk dict for testing."""
    return {
        "metadata": {
            "filename": filename,
            "page_number": str(page_number),
            "text": text,
        },
        "distance": distance,
    }


def test_build_context_block_basic():
    """Context block should contain source labels and text."""
    chunks = [
        make_chunk("doc.pdf", 3, "Content from page three."),
        make_chunk("doc.pdf", 7, "Content from page seven."),
    ]
    block = build_context_block(chunks)

    assert "[Source 1 — doc.pdf, page 3]" in block
    assert "[Source 2 — doc.pdf, page 7]" in block
    assert "Content from page three." in block
    assert "Content from page seven." in block


def test_build_context_block_empty():
    """Empty chunks should return no context available message."""
    block = build_context_block([])
    assert block == "No context available."


def test_build_context_block_skips_empty_text():
    """Chunks with empty text should be skipped gracefully."""
    chunks = [
        make_chunk("doc.pdf", 1, ""),  # empty — should skip
        make_chunk("doc.pdf", 2, "Real content here."),
    ]
    block = build_context_block(chunks)

    assert "[Source 1 — doc.pdf, page 1]" not in block
    assert "Real content here." in block


# ── parse_sources ────────────────────────────────────────────────────────────


def test_parse_sources_cited_flag():
    """
    Sources that Claude cited should have cited=True.
    Sources not cited should have cited=False.
    """
    chunks = [
        make_chunk("doc.pdf", 3, "Content A."),
        make_chunk("doc.pdf", 7, "Content B."),
    ]
    cited_labels = ["Source 1 — doc.pdf, page 3"]

    sources = parse_sources(chunks, cited_labels)

    assert sources[0]["cited"] is True
    assert sources[1]["cited"] is False


def test_parse_sources_excerpt_truncated():
    """Excerpts longer than 200 chars should be truncated with ellipsis."""
    long_text = "A" * 300
    chunks = [make_chunk("doc.pdf", 1, long_text)]
    sources = parse_sources(chunks, [])

    assert len(sources[0]["excerpt"]) <= 204
    assert sources[0]["excerpt"].endswith("...")


def test_parse_sources_short_excerpt_not_truncated():
    """Short excerpts should not be truncated."""
    short_text = "Short content."
    chunks = [make_chunk("doc.pdf", 1, short_text)]
    sources = parse_sources(chunks, [])

    assert sources[0]["excerpt"] == short_text
    assert not sources[0]["excerpt"].endswith("...")


def test_parse_sources_empty_chunks():
    """Empty chunk list should return empty sources list."""
    sources = parse_sources([], [])
    assert sources == []


def test_parse_sources_structure():
    """Each source should have all required fields."""
    chunks = [make_chunk("doc.pdf", 5, "Some content.")]
    sources = parse_sources(chunks, [])

    assert len(sources) == 1
    source = sources[0]
    assert "source_number" in source
    assert "filename" in source
    assert "page_number" in source
    assert "excerpt" in source
    assert "distance" in source
    assert "cited" in source
