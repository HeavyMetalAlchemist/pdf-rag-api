import json
import asyncio
import re

from loguru import logger
from opentelemetry import trace

from src.core.config import settings
from src.services.embeddings import bedrock_client

tracer = trace.get_tracer(__name__)

SYSTEM_PROMPT = """You are a document assistant. Your job is to answer questions \
using only the context provided below.

Rules:
- Answer using only information from the provided context
- If the answer is not in the context, respond exactly with:
  "I cannot find the answer to this question in the provided documents."
- Do not use your general knowledge under any circumstances
- Be concise and direct
- At the end of your answer, list all sources you used under a
  "Sources:" heading with each source on its own line, using the
  exact full source label provided, for example:
  Sources:
  [Source 1 — aws_whitepaper.pdf, page 3]
  [Source 2 — aws_whitepaper.pdf, page 7]"""


def build_context_block(retrieved_chunks):
    """
    Formats retrieved chunks into labeled context blocks for the prompt.
    Each chunk gets a source label with filename and page number.

    Example output:
        [Source 1 — aws_doc.pdf, page 7]
        Server-side encryption should be enabled...

        [Source 2 — aws_doc.pdf, page 12]
        Bucket policies should follow least-privilege...
    """
    if not retrieved_chunks:
        return "No context available."

    blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        try:
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "unknown")
            page_number = metadata.get("page_number", "?")
            text = metadata.get("text", "").strip()

            if not text:
                logger.warning(f"Chunk {i} has empty text, skipping")
                continue

            label = f"[Source {i} — {filename}, page {page_number}]"
            blocks.append(f"{label}\n{text}")

        except Exception as e:
            logger.error(f"Failed to build context block for chunk {i}: {e}")
            continue

    if not blocks:
        return "No context available."

    return "\n\n".join(blocks)


def build_user_message(question, retrieved_chunks):
    """
    Builds the full user message combining context block and question.
    Context comes before the question so Claude reads the evidence first.
    """
    context_block = build_context_block(retrieved_chunks)

    return f"""Context:
{context_block}

Question: {question}"""


def extract_citations(answer_text):
    """
    Extracts bracketed citations from Claude's answer text using regex.
    Only extracts brackets starting with 'Source' to avoid matching
    other bracketed content in the answer.

    Handles both formats Claude may return:
        Single line: [Source 1 — file.pdf, page 3] [Source 2 — file.pdf, page 7]
        Multi line:
            [Source 1 — file.pdf, page 3]
            [Source 2 — file.pdf, page 7]

    Returns a list of citation strings without the brackets.
    Returns empty list if no citations found or on any error.
    """
    if not answer_text:
        return []

    try:
        all_matches = re.findall(r"\[([^\]]+)\]", answer_text)
        # Only keep matches that are actual source citations
        citations = [
            match.strip() for match in all_matches if match.strip().startswith("Source")
        ]
        return citations

    except Exception as e:
        logger.error(f"Failed to extract citations from answer: {e}")
        return []


def parse_sources(retrieved_chunks, cited_labels):
    """
    Builds structured source list from retrieved chunks.
    Cross-references with extracted citation labels to mark
    which sources Claude actually used in its answer.

    cited_labels is a list of strings like:
        ["Source 1 — aws_doc.pdf, page 3", "Source 2 — aws_doc.pdf, page 7"]

    Returns a list of source dicts with a cited flag indicating
    whether Claude referenced this source in its answer.
    """
    sources = []

    for i, chunk in enumerate(retrieved_chunks, 1):
        try:
            metadata = chunk.get("metadata", {})
            filename = metadata.get("filename", "unknown")
            page_number = metadata.get("page_number", "?")
            text = metadata.get("text", "")
            distance = chunk.get("distance")

            # Build the label we gave Claude for this chunk
            # Check if Claude cited it by matching against extracted labels
            expected_label = f"Source {i} — {filename}, page {page_number}"
            cited = any(
                expected_label.lower() in label.lower() for label in cited_labels
            )

            # Truncate excerpt to 200 chars for clean API response
            excerpt = text[:200] + "..." if len(text) > 200 else text

            sources.append(
                {
                    "source_number": i,
                    "filename": filename,
                    "page_number": page_number,
                    "excerpt": excerpt,
                    "distance": distance,
                    "cited": cited,
                }
            )

        except Exception as e:
            logger.error(f"Failed to parse source {i}: {e}")
            continue

    return sources


async def generate_answer(question, retrieved_chunks):
    """
    Generates a grounded answer from retrieved chunks using Claude.

    Flow:
        1. Early return if no chunks retrieved — avoids hallucination
        2. Format chunks into labeled context block
        3. Build user message with context and question
        4. Call Claude with system prompt and user message
        5. Extract citations from Claude's response
        6. Build structured sources list with cited flags
        7. Return answer text and sources

    Raises ValueError for invalid inputs.
    Raises RuntimeError if Bedrock call fails.
    """
    # Validate inputs
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    question = question.strip()

    # Early return if nothing was retrieved
    # Avoids calling Claude with empty context which risks hallucination
    if not retrieved_chunks:
        logger.warning(f"No chunks retrieved for question: {question[:50]}")
        return {
            "answer": "I cannot find the answer to this question in the provided documents.",
            "sources": [],
        }

    user_message = build_user_message(question, retrieved_chunks)

    request_body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": settings.generation_max_tokens,
            "temperature": settings.generation_temperature,
            "top_p": 0.9,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
        }
    )

    with tracer.start_as_current_span("generate_answer") as span:
        span.set_attribute("question_length", len(question))
        span.set_attribute("chunk_count", len(retrieved_chunks))
        span.set_attribute("model_id", settings.bedrock_generation_model_id)

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: bedrock_client.invoke_model(
                    modelId=settings.bedrock_generation_model_id,
                    body=request_body,
                    contentType="application/json",
                    accept="application/json",
                ),
            )

        except Exception as e:
            logger.error(f"Bedrock generation call failed: {e}")
            raise RuntimeError(f"Failed to generate answer: {e}")

    # Parse response
    try:
        response_body = json.loads(response["body"].read())
        answer_text = response_body["content"][0]["text"]

    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse Bedrock response: {e}")
        raise RuntimeError(f"Failed to parse generation response: {e}")

    # Extract citations Claude included in its answer
    cited_labels = extract_citations(answer_text)
    logger.debug(f"Extracted {len(cited_labels)} citations from answer")

    # Build structured sources with cited flags
    sources = parse_sources(retrieved_chunks, cited_labels)

    logger.info(
        f"Generated answer for question='{question[:50]}' "
        f"using {len(retrieved_chunks)} chunks, "
        f"{len(cited_labels)} cited"
    )

    return {
        "answer": answer_text,
        "sources": sources,
    }
