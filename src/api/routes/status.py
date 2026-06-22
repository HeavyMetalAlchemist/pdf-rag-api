from fastapi import APIRouter, Depends
from opentelemetry import trace
from src.api.dependencies import get_document_or_404

tracer = trace.get_tracer(__name__)
router = APIRouter()


@router.get(
    "/status/{document_id}",
    summary="Check document ingestion status",
    description=(
        "Returns the current processing status of an uploaded document. "
        "Poll this endpoint after POST /ingest until status is 'ready'."
    ),
)
async def get_status(
    document_id,
    status_data=Depends(get_document_or_404),
):
    """
    Returns status JSON stored in S3 for the given document_id.
    get_document_or_404 dependency handles the S3 read and raises
    404 if the document does not exist — this function only runs
    if the document exists.
    """
    with tracer.start_as_current_span("get_status") as span:
        span.set_attribute("document_id", document_id)
        span.set_attribute("status", status_data.get("status", "unknown"))
    status_data.pop("vector_keys", None)  # unnecessary to show in the UI
    return status_data
