import hashlib
import asyncio

import boto3
from loguru import logger
from opentelemetry import trace

from src.core.config import settings

tracer = trace.get_tracer(__name__)

s3vectors_client = boto3.client(
    "s3vectors",
    region_name=settings.aws_region,
)

METADATA_TEXT_LIMIT = 1900


def get_content_hash(text):
    """
    Generates MD5 hash of chunk text.
    Used as part of the vector key — not for security,
    only for deduplication within a document.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_vector_key(document_id, chunk_index, chunk_text):
    """
    Builds a unique vector key scoped to a specific document.

    Format: {document_id}:{chunk_index}:{content_hash}

    Scoping by document_id ensures:
    - Same file uploaded twice creates independent vectors
    - Delete can target exact keys for one document
    - No cross-document key collisions
    """
    content_hash = get_content_hash(chunk_text)
    return f"{document_id}:{chunk_index}:{content_hash}"


async def store_chunks(document_id, filename, chunks, embeddings):
    """
    Stores chunk embeddings with metadata into S3 Vectors.
    Processes in batches of 500 — S3 Vectors API limit per request.

    Returns list of vector keys stored — caller should persist these
    in the status file so delete can target them later without
    needing to query S3 Vectors.
    """
    vectors = []
    vector_keys = []

    for chunk, embedding in zip(chunks, embeddings):
        chunk_text = chunk["text"]
        chunk_index = chunk["chunk_index"]

        # Document-scoped key prevents cross-document collisions
        key = build_vector_key(document_id, chunk_index, chunk_text)
        vector_keys.append(key)

        truncated_text = chunk_text.encode("utf-8")[:METADATA_TEXT_LIMIT].decode(
            "utf-8", errors="ignore"
        )

        vectors.append(
            {
                "key": key,
                "data": {"float32": embedding},
                "metadata": {
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": str(chunk["page_number"]),
                    "chunk_index": str(chunk_index),
                    "text": truncated_text,
                },
            }
        )

    batch_size = 500
    batches = [vectors[i : i + batch_size] for i in range(0, len(vectors), batch_size)]

    with tracer.start_as_current_span("store_chunks") as span:
        span.set_attribute("document_id", document_id)
        span.set_attribute("chunk_count", len(vectors))
        span.set_attribute("batch_count", len(batches))

        loop = asyncio.get_event_loop()

        for i, batch in enumerate(batches):
            logger.debug(f"Storing batch {i + 1}/{len(batches)}")

            await loop.run_in_executor(
                None,
                lambda b=batch: s3vectors_client.put_vectors(
                    vectorBucketName=settings.s3_vector_bucket_name,
                    indexName=settings.s3_vector_index_name,
                    vectors=b,
                ),
            )

    logger.info(f"Stored {len(vectors)} vectors for document_id={document_id}")

    # Return keys so caller can persist them for future delete operations
    return vector_keys


async def query_chunks(query_embedding, document_id=None):
    """
    Queries S3 Vectors for the top-k most similar chunks.
    Optionally filters by document_id to scope search to one document.
    Returns list of metadata dicts ordered by similarity descending.
    """
    vector_filter = None
    if document_id:
        vector_filter = {"document_id": {"$eq": document_id}}

    with tracer.start_as_current_span("query_chunks") as span:
        span.set_attribute("top_k", settings.top_k)
        span.set_attribute("document_id", document_id if document_id else "global")

        loop = asyncio.get_event_loop()

        kwargs = {
            "vectorBucketName": settings.s3_vector_bucket_name,
            "indexName": settings.s3_vector_index_name,
            "queryVector": {"float32": query_embedding},
            "topK": settings.top_k,
            "returnMetadata": True,
            "returnDistance": True,
        }

        if vector_filter:
            kwargs["filter"] = vector_filter

        response = await loop.run_in_executor(
            None, lambda: s3vectors_client.query_vectors(**kwargs)
        )

    results = response.get("vectors", [])

    logger.info(
        f"Retrieved {len(results)} chunks "
        f"document_id={document_id if document_id else 'global'}"
    )

    return results


async def delete_vectors(vector_keys):
    """
    Deletes vectors from S3 Vectors by their exact keys.
    Processes in batches of 500 — S3 Vectors API limit per request.

    vector_keys should come from the status JSON stored at ingestion
    time — this avoids needing to query S3 Vectors to find keys.

    Returns count of successfully deleted vectors.
    """
    if not vector_keys:
        logger.warning("delete_vectors called with empty key list")
        return 0

    batch_size = 500
    batches = [
        vector_keys[i : i + batch_size] for i in range(0, len(vector_keys), batch_size)
    ]

    deleted_count = 0

    with tracer.start_as_current_span("delete_vectors") as span:
        span.set_attribute("key_count", len(vector_keys))
        span.set_attribute("batch_count", len(batches))

        loop = asyncio.get_event_loop()

        for i, batch in enumerate(batches):
            logger.debug(
                f"Deleting batch {i + 1}/{len(batches)} ({len(batch)} vectors)"
            )

            try:
                await loop.run_in_executor(
                    None,
                    lambda b=batch: s3vectors_client.delete_vectors(
                        vectorBucketName=settings.s3_vector_bucket_name,
                        indexName=settings.s3_vector_index_name,
                        keys=b,
                    ),
                )
                deleted_count += len(batch)

            except Exception as e:
                logger.error(f"Failed to delete batch {i + 1}: {e}")
                raise RuntimeError(f"Vector deletion failed on batch {i + 1}: {e}")

    logger.info(f"Deleted {deleted_count} vectors")
    return deleted_count
