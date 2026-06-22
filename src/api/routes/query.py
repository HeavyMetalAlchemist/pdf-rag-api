from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator
from loguru import logger
from opentelemetry import trace

from src.api.dependencies import get_ready_document_or_422
from src.services import embeddings, vectors, generator

tracer = trace.get_tracer(__name__)

router = APIRouter()


class QueryRequest(BaseModel):
    """
    Request body for POST /query.

    document_id is optional — if provided, search is scoped to that
    document only. If omitted, search is global across all documents.
    """

    question: str
    document_id: str = None

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


@router.post(
    "/query",
    summary="Query ingested documents",
    description=(
        "Embeds the question, retrieves relevant chunks from S3 Vectors, "
        "and generates a grounded answer using Claude. "
        "Optionally scope search to a specific document by providing document_id."
    ),
)
async def query_documents(
    request: QueryRequest,
    # Only runs dependency if document_id provided
    # If no document_id, skip validation — global search
):
    """
    Full RAG query pipeline:
        1. Embed question via Bedrock Titan v2
        2. Retrieve top-k chunks from S3 Vectors
        3. Generate grounded answer via Claude
        4. Return answer with citations
    """
    with tracer.start_as_current_span("query_documents") as span:
        span.set_attribute("question_length", len(request.question))
        span.set_attribute(
            "document_id", request.document_id if request.document_id else "global"
        )

        # Validate document exists and is ready if document_id provided
        if request.document_id:
            status_data = await get_ready_document_or_422(request.document_id)
            logger.info(
                f"Scoped query: document_id={request.document_id} "
                f"filename={status_data.get('filename')}"
            )
        else:
            logger.info("Global query across all documents")

        # Step 1 — embed the question
        try:
            query_embedding = await embeddings.embed_text(request.question)
        except Exception as e:
            logger.error(f"Failed to embed question: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to process question. Please try again.",
            )

        # Step 2 — retrieve relevant chunks from S3 Vectors
        try:
            retrieved_chunks = await vectors.query_chunks(
                query_embedding=query_embedding,
                document_id=request.document_id,
            )
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to retrieve relevant content. Please try again.",
            )

        span.set_attribute("retrieved_chunks", len(retrieved_chunks))

        # Step 3 — generate grounded answer
        try:
            result = await generator.generate_answer(
                question=request.question,
                retrieved_chunks=retrieved_chunks,
            )
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to generate answer. Please try again.",
            )

        logger.info(
            f"Query complete: question='{request.question[:50]}' "
            f"chunks_retrieved={len(retrieved_chunks)} "
            f"chunks_cited={sum(1 for s in result['sources'] if s['cited'])}"
        )

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"],
            "metadata": {
                "chunks_retrieved": len(retrieved_chunks),
                "chunks_cited": sum(1 for s in result["sources"] if s["cited"]),
                "document_id": request.document_id,
            },
        }
