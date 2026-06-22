import asyncio
import json
import uuid
from datetime import datetime

import boto3
from loguru import logger
from opentelemetry import trace

from src.core.config import settings

# Module level clients — created once, reused across requests
# boto3 clients are thread-safe for reads, fine to share
s3_client = boto3.client(
    "s3",
    region_name=settings.aws_region,
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
    aws_session_token=settings.aws_session_token,
)

# Tracer for this module — spans created here appear as children
# of the parent request span in Jaeger
tracer = trace.get_tracer(__name__)

# S3 key prefixes — all files for this project live under document-api/
PDF_PREFIX = "document-api/pdfs"
STATUS_PREFIX = "document-api/status"


def generate_document_id():
    """
    Generates a unique document ID.
    UUID4 is random — no coordination needed across instances.
    """
    return str(uuid.uuid4())


async def upload_pdf(file_bytes, filename, document_id):
    """
    Uploads raw PDF bytes to S3 under the pdfs/ prefix.
    Returns the S3 key where the file was stored.

    Uses run_in_executor to avoid blocking the async event loop
    with a synchronous boto3 call.
    """
    s3_key = f"{PDF_PREFIX}/{document_id}/{filename}"

    with tracer.start_as_current_span("upload_pdf") as span:
        span.set_attribute("document_id", document_id)
        span.set_attribute("filename", filename)
        span.set_attribute("file_size_bytes", len(file_bytes))
        span.set_attribute("s3_key", s3_key)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: s3_client.put_object(
                Bucket=settings.s3_bucket_name,
                Key=s3_key,
                Body=file_bytes,
                ContentType="application/pdf",
            ),
        )

    logger.info(f"Uploaded PDF to s3://{settings.s3_bucket_name}/{s3_key}")
    return s3_key


async def write_status(
    document_id,
    filename,
    status,
    chunk_count=None,
    error=None,
    created_at=None,
    vector_keys=None,
):
    """
    Writes or updates the status JSON file for a document in S3.

    created_at is optional:
    - First call (status=processing): not passed, generated here
    - Subsequent calls: passed explicitly to preserve original value
      avoiding a redundant S3 read
    """
    s3_key = f"{STATUS_PREFIX}/{document_id}.json"

    # Generate created_at only on first write
    # Subsequent calls pass it explicitly
    if created_at is None:
        created_at = datetime.utcnow().isoformat()

    status_data = {
        "document_id": document_id,
        "filename": filename,
        "status": status,
        "chunk_count": chunk_count,
        "vector_keys": vector_keys,
        "created_at": created_at,
        "completed_at": datetime.utcnow().isoformat()
        if status in ("ready", "failed")
        else None,
        "error": error,
    }

    with tracer.start_as_current_span("write_status") as span:
        span.set_attribute("document_id", document_id)
        span.set_attribute("status", status)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: s3_client.put_object(
                Bucket=settings.s3_bucket_name,
                Key=s3_key,
                Body=json.dumps(status_data),
                ContentType="application/json",
            ),
        )

    logger.info(f"Status updated: document_id={document_id} status={status}")
    return status_data


async def read_status(document_id):
    """
    Reads the status JSON file for a document from S3.
    Returns None if the document does not exist rather than raising.
    This is used by the /status endpoint and by write_status
    to preserve created_at on updates.
    """
    s3_key = f"{STATUS_PREFIX}/{document_id}.json"

    try:
        with tracer.start_as_current_span("read_status") as span:
            span.set_attribute("document_id", document_id)

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: s3_client.get_object(
                    Bucket=settings.s3_bucket_name,
                    Key=s3_key,
                ),
            )

        body = response["Body"].read()
        return json.loads(body)

    except s3_client.exceptions.NoSuchKey:
        # Document does not exist — return None, let caller decide what to do
        return None
    except Exception as e:
        logger.error(f"Failed to read status for {document_id}: {e}")
        return None


async def list_documents():
    """
    Lists all documents by reading status files from the status prefix.
    Returns a list of status dicts ordered by created_at descending.
    """
    try:
        loop = asyncio.get_event_loop()

        # List all objects under the status prefix
        response = await loop.run_in_executor(
            None,
            lambda: s3_client.list_objects_v2(
                Bucket=settings.s3_bucket_name,
                Prefix=f"{STATUS_PREFIX}/",
            ),
        )

        objects = response.get("Contents", [])

        if not objects:
            return []

        # Read each status file
        documents = []
        for obj in objects:
            key = obj["Key"]
            # Skip if not a JSON file
            if not key.endswith(".json"):
                continue

            try:
                file_response = await loop.run_in_executor(
                    None,
                    lambda k=key: s3_client.get_object(
                        Bucket=settings.s3_bucket_name,
                        Key=k,
                    ),
                )
                body = file_response["Body"].read()
                status_data = json.loads(body)

                # Exclude vector_keys from list response
                # They're internal implementation detail
                # not useful to API consumers
                status_data.pop("vector_keys", None)
                documents.append(status_data)

            except Exception as e:
                logger.error(f"Failed to read status file {key}: {e}")
                continue

        # Sort by created_at descending — newest first
        documents.sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )

        return documents

    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise


async def delete_pdf(document_id, filename):
    """
    Deletes the original PDF file from S3.
    """
    s3_key = f"{PDF_PREFIX}/{document_id}/{filename}"

    with tracer.start_as_current_span("delete_pdf") as span:
        span.set_attribute("document_id", document_id)
        span.set_attribute("s3_key", s3_key)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: s3_client.delete_object(
                Bucket=settings.s3_bucket_name,
                Key=s3_key,
            ),
        )

    logger.info(f"Deleted PDF: s3://{settings.s3_bucket_name}/{s3_key}")


async def delete_status(document_id):
    """
    Deletes the status JSON file from S3.
    Called last in the delete pipeline so the document
    remains discoverable if earlier deletion steps fail.
    """
    s3_key = f"{STATUS_PREFIX}/{document_id}.json"

    with tracer.start_as_current_span("delete_status") as span:
        span.set_attribute("document_id", document_id)
        span.set_attribute("s3_key", s3_key)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: s3_client.delete_object(
                Bucket=settings.s3_bucket_name,
                Key=s3_key,
            ),
        )

    logger.info(f"Deleted status: s3://{settings.s3_bucket_name}/{s3_key}")
