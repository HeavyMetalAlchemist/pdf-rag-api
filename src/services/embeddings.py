import asyncio
import json

import boto3
from loguru import logger
from opentelemetry import trace

from src.core.config import settings

# Titan Text v2 maximum input tokens
# Our chunks are 512 tokens so this is a safety guard only
TITAN_MAX_TOKENS = 8192

# Approximate characters per token for truncation guard
# This is a rough estimate — tokenization is model specific
# 4 chars per token is a safe conservative estimate
CHARS_PER_TOKEN = 4
MAX_CHARS = TITAN_MAX_TOKENS * CHARS_PER_TOKEN

tracer = trace.get_tracer(__name__)

# Bedrock runtime client — used for both embedding and generation
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=settings.aws_region,
    aws_access_key_id=settings.aws_access_key_id,
    aws_secret_access_key=settings.aws_secret_access_key,
    aws_session_token=settings.aws_session_token,
)


async def embed_text(text):
    """
    Embeds a single text string using Bedrock Titan Text v2.
    Returns a list of 1024 floats.

    normalize=True is required for cosine similarity to work correctly
    in S3 Vectors — normalized vectors make dot product equivalent
    to cosine similarity.
    """
    # Truncate if text exceeds approximate token limit
    # Better to truncate than to fail the entire ingestion
    if len(text) > MAX_CHARS:
        logger.warning(
            f"Text truncated from {len(text)} to {MAX_CHARS} chars before embedding"
        )
        text = text[:MAX_CHARS]

    request_body = json.dumps(
        {
            "inputText": text,
            "dimensions": 1024,
            "normalize": True,
        }
    )

    with tracer.start_as_current_span("embed_text") as span:
        span.set_attribute("text_length", len(text))
        span.set_attribute("model_id", settings.bedrock_embedding_model_id)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: bedrock_client.invoke_model(
                modelId=settings.bedrock_embedding_model_id,
                body=request_body,
                contentType="application/json",
                accept="application/json",
            ),
        )

    response_body = json.loads(response["body"].read())
    embedding = response_body["embedding"]

    return embedding


async def embed_chunks(chunks):
    """
    chunks is a list of dicts with text, page_number, chunk_index.
    Extracts text from each dict before embedding.
    """
    embeddings = []

    with tracer.start_as_current_span("embed_chunks") as span:
        span.set_attribute("chunk_count", len(chunks))

        for i, chunk in enumerate(chunks):
            # extract text from chunk dict
            logger.debug(f"Embedding chunk {i + 1}/{len(chunks)}")
            embedding = await embed_text(chunk["text"])
            embeddings.append(embedding)

    logger.info(f"Embedded {len(chunks)} chunks successfully")
    return embeddings
