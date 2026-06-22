from fastapi import Depends, HTTPException, status
from loguru import logger

from src.core.config import get_settings
from src.services import storage


async def get_document_or_404(document_id):
    """
    Dependency that verifies a document exists in S3 before
    the route runs. If the document status file does not exist,
    raises a 404 immediately and the route never executes.

    Used by /status and /query routes to avoid duplicating
    the existence check in both places.

    FastAPI injects document_id from the path parameter
    automatically when this is used as a dependency.
    """
    status_data = await storage.read_status(document_id)

    if status_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document {document_id} not found",
        )

    return status_data


async def get_ready_document_or_422(document_id):
    """
    Dependency that verifies a document exists AND is ready
    to be queried. Raises appropriate errors if not.

    Used by /query route only — querying a document that is
    still processing or failed ingestion should be rejected
    with a clear error message, not silently return empty results.

    Status transitions:
        processing → document uploaded, ingestion in progress
        ready      → ingestion complete, safe to query
        failed     → ingestion failed, cannot query
    """
    status_data = await get_document_or_404(document_id)
    doc_status = status_data.get("status")

    if doc_status == "processing":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Document {document_id} is still being processed. "
                f"Check /status/{document_id} and retry when status is 'ready'."
            ),
        )

    if doc_status == "failed":
        error = status_data.get("error", "unknown error")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Document {document_id} failed ingestion: {error}. "
                f"Please re-upload the document."
            ),
        )

    if doc_status != "ready":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Document {document_id} has unexpected status: {doc_status}",
        )

    return status_data
