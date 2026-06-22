from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, status
from loguru import logger
from opentelemetry import trace

from src.services import storage, chunker, embeddings, vectors
from src.services.storage import generate_document_id

tracer = trace.get_tracer(__name__)

router = APIRouter()

# Maximum file size — 20MB
# Large PDFs slow down ingestion significantly
# and risk hitting Bedrock token limits per chunk
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024


async def process_document(document_id, filename, pdf_bytes, created_at):
    """
    Background task that runs after /ingest returns.
    Orchestrates the full ingestion pipeline:
        chunk → embed → store → update status

    Any failure updates status to failed with the error message
    so the client can see what went wrong via /status endpoint.
    """
    logger.info(f"Background ingestion started: document_id={document_id}")

    try:
        chunks = chunker.process_pdf(pdf_bytes)

        if not chunks:
            raise ValueError("PDF produced no chunks after processing")

        logger.info(f"document_id={document_id} produced {len(chunks)} chunks")

        chunk_embeddings = await embeddings.embed_chunks(chunks)

        vector_keys = await vectors.store_chunks(
            document_id=document_id,
            filename=filename,
            chunks=chunks,
            embeddings=chunk_embeddings,
        )
        stored_count = len(vector_keys)

        await storage.write_status(
            document_id=document_id,
            filename=filename,
            status="ready",
            chunk_count=stored_count,
            created_at=created_at,
            vector_keys=vector_keys,
        )

        logger.info(
            f"Ingestion complete: document_id={document_id} chunks={stored_count}"
        )

    except Exception as e:
        logger.error(f"Ingestion failed: document_id={document_id} error={e}")
        await storage.write_status(
            document_id=document_id,
            filename=filename,
            status="failed",
            error=str(e),
            created_at=created_at,  # ← passed explicitly
        )


@router.post(
    "/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and ingest a PDF document",
    description=(
        "Uploads a PDF to S3 and starts background ingestion. "
        "Returns immediately with a document_id. "
        "Poll GET /status/{document_id} to check when ingestion is complete."
    ),
)
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    with tracer.start_as_current_span("ingest_document") as span:
        if file.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    f"Unsupported file type: {file.content_type}. "
                    f"Only PDF files are accepted."
                ),
            )

        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Filename is required",
            )

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="File must have a .pdf extension",
            )

        pdf_bytes = await file.read()

        if len(pdf_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty",
            )

        if len(pdf_bytes) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"File size {len(pdf_bytes) / 1024 / 1024:.1f}MB "
                    f"exceeds maximum allowed size of "
                    f"{MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f}MB"
                ),
            )

        document_id = generate_document_id()

        span.set_attribute("document_id", document_id)
        span.set_attribute("filename", file.filename)
        span.set_attribute("file_size_bytes", len(pdf_bytes))

        try:
            await storage.upload_pdf(
                file_bytes=pdf_bytes,
                filename=file.filename,
                document_id=document_id,
            )
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to upload document. Please try again.",
            )

        try:
            # Capture returned status_data to extract created_at
            initial_status = await storage.write_status(
                document_id=document_id,
                filename=file.filename,
                status="processing",
            )
        except Exception as e:
            logger.error(f"Status write failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to initialize document status. Please try again.",
            )

        # Pass created_at to background task so subsequent
        # status writes preserve the original timestamp
        background_tasks.add_task(
            process_document,
            document_id,
            file.filename,
            pdf_bytes,
            initial_status["created_at"],
        )

        logger.info(
            f"Ingest accepted: document_id={document_id} "
            f"filename={file.filename} "
            f"size={len(pdf_bytes)} bytes"
        )

        return {
            "document_id": document_id,
            "filename": file.filename,
            "status": "processing",
            "message": (
                f"Document accepted for processing. "
                f"Poll GET /status/{document_id} to check progress."
            ),
        }
