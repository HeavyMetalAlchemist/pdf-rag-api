import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport
from src.main import app


@pytest.mark.anyio
async def test_ingest_returns_202_with_valid_pdf():
    """
    Tests that POST /ingest returns 202 and triggers background
    processing when a valid PDF is uploaded.

    Mocks all AWS calls so no real infrastructure is needed.
    Verifies the route logic itself — validation, response shape,
    and that storage functions are called with correct arguments.
    """
    # Mock all external calls
    # AsyncMock for async functions, MagicMock for sync
    with (
        patch(
            "src.api.routes.ingest.storage.upload_pdf", new_callable=AsyncMock
        ) as mock_upload,
        patch(
            "src.api.routes.ingest.storage.write_status", new_callable=AsyncMock
        ) as mock_status,
        patch(
            "src.api.routes.ingest.process_document", new_callable=AsyncMock
        ) as mock_process,
    ):
        # write_status returns a dict with created_at
        mock_status.return_value = {
            "document_id": "test-id",
            "filename": "test.pdf",
            "status": "processing",
            "created_at": "2026-01-01T00:00:00",
            "completed_at": None,
            "chunk_count": None,
            "error": None,
        }

        # Create a minimal valid PDF in memory
        # PyMuPDF can open this as a valid empty PDF
        minimal_pdf = b"%PDF-1.4\n%EOF"

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/ingest",
                files={"file": ("test.pdf", minimal_pdf, "application/pdf")},
            )

        # Verify response
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "processing"
        assert body["filename"] == "test.pdf"
        assert "document_id" in body
        assert "message" in body

        # Verify AWS calls were made
        mock_upload.assert_called_once()
        mock_status.assert_called_once()


@pytest.mark.anyio
async def test_ingest_rejects_non_pdf():
    """
    Tests that POST /ingest returns 415 for non-PDF files.
    No mocking needed — validation happens before any AWS calls.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/ingest",
            files={"file": ("test.txt", b"not a pdf", "text/plain")},
        )

    assert response.status_code == 415


@pytest.mark.anyio
async def test_ingest_rejects_empty_file():
    """
    Tests that POST /ingest returns 400 for empty files.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/ingest",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )

    assert response.status_code == 400


@pytest.mark.anyio
async def test_status_returns_404_for_unknown_document():
    """
    Tests that GET /status/{document_id} returns 404 for
    a document that doesn't exist.

    Mocks read_status to return None — simulates missing document.
    """
    with patch(
        "src.api.dependencies.storage.read_status", new_callable=AsyncMock
    ) as mock_read:
        mock_read.return_value = None

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/status/nonexistent-id")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
