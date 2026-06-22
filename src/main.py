from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time
from loguru import logger

from src.core.telemetry import setup_telemetry
from src.api.routes import ingest, status, query, documents


@asynccontextmanager
async def lifespan(app):
    logger.info("Starting PDF RAG API")
    yield
    logger.info("Shutting down PDF RAG API")


app = FastAPI(
    title="PDF RAG API",
    description="RAG-based PDF ingestion and grounded document Q&A.",
    version="0.1.0",
    lifespan=lifespan,
)

setup_telemetry(app)


@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = (time.time() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"duration={duration:.2f}ms"
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.exception(f"Unhandled error on {request.method} {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/health")
def health_check():
    logger.info("Health check called ")
    return {"status": "ok"}


# Include routers — this is how FastAPI learns about our routes
app.include_router(ingest.router, tags=["Documents"])
app.include_router(status.router, tags=["Documents"])
app.include_router(query.router, tags=["Documents"])
app.include_router(documents.router, tags=["Documents"])
