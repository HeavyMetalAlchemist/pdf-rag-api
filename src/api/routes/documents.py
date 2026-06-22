from fastapi import APIRouter, HTTPException, Depends, status
from loguru import logger
from opentelemetry import trace

from src.api.dependencies import get_document_or_404
from src.services import storage, vectors

tracer = trace.get_tracer(__name__)

router = APIRouter()


@router.get(
    "/documents",
    summary="List all ingested documents",
    description="Returns all documents that have been uploaded and their current status.",
)
async def list_documents():
    """
    Lists all documents by reading status files from S3.
    Returns filename, status, chunk_count and timestamps for each.
    """
    with tracer.start_as_current_span("list_documents"):
        try:
            documents = await storage.list_documents()
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to retrieve document list. Please try again.",
            )

    return {
        "documents": documents,
        "count": len(documents),
    }


@router.delete(
    "/documents/{document_id}",
    summary="Delete a document",
    description=(
        "Deletes a document and all associated data: "
        "vectors from S3 Vectors, original PDF from S3, "
        "and status file from S3."
    ),
)
async def delete_document(
    document_id,
    status_data=Depends(get_document_or_404),
):
    """
    Full document deletion:
        1. Delete vectors from S3 Vectors using stored keys
        2. Delete original PDF from S3
        3. Delete status file from S3

    Uses vector_keys stored in status JSON at ingestion time
    to avoid needing to query S3 Vectors for keys.
    """
    with tracer.start_as_current_span("delete_document") as span:
        span.set_attribute("document_id", document_id)

        filename = status_data.get("filename", "unknown")
        vector_keys = status_data.get("vector_keys", [])
        doc_status = status_data.get("status")

        span.set_attribute("vector_key_count", len(vector_keys))
        span.set_attribute("filename", filename)

        errors = []

        # Step 1 — delete vectors from S3 Vectors
        # Only attempt if document was successfully ingested
        # A failed or processing document may have no vectors
        if doc_status == "ready" and vector_keys:
            try:
                deleted_count = await vectors.delete_vectors(vector_keys)
                logger.info(
                    f"Deleted {deleted_count} vectors for document_id={document_id}"
                )
            except Exception as e:
                logger.error(f"Vector deletion failed: {e}")
                errors.append(f"Vector deletion failed: {str(e)}")
        else:
            logger.info(
                f"Skipping vector deletion for document_id={document_id} "
                f"status={doc_status} keys={len(vector_keys)}"
            )

        # Step 2 — delete original PDF from S3
        try:
            await storage.delete_pdf(document_id, filename)
            logger.info(f"Deleted PDF for document_id={document_id}")
        except Exception as e:
            logger.error(f"PDF deletion failed: {e}")
            errors.append(f"PDF deletion failed: {str(e)}")

        # Step 3 — delete status file from S3
        # Do this last so if earlier steps fail the document
        # is still discoverable and deletable on retry
        try:
            await storage.delete_status(document_id)
            logger.info(f"Deleted status for document_id={document_id}")
        except Exception as e:
            logger.error(f"Status deletion failed: {e}")
            errors.append(f"Status deletion failed: {str(e)}")

        if errors:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "message": "Document partially deleted with errors",
                    "errors": errors,
                },
            )

    logger.info(f"Document fully deleted: document_id={document_id}")

    return {
        "document_id": document_id,
        "filename": filename,
        "deleted": True,
        "vectors_deleted": len(vector_keys) if doc_status == "ready" else 0,
    }
