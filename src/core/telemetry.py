from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource

from src.core.config import settings


def setup_telemetry(app):
    """
    Initializes OpenTelemetry tracing and wires it to Jaeger via OTLP gRPC.

    Three things happen here:
    1. A TracerProvider is created with service metadata attached
    2. A BatchSpanProcessor exports spans to Jaeger in the background
       without blocking request handling
    3. FastAPIInstrumentor automatically creates spans for every
       incoming HTTP request — no manual instrumentation needed for routes
    """

    # Resource identifies this service in Jaeger
    # Every trace will show "document-intelligence-api" as the service name
    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
        }
    )

    # TracerProvider is the central object that manages tracing
    provider = TracerProvider(resource=resource)

    # OTLPSpanExporter sends spans to Jaeger over gRPC
    exporter = OTLPSpanExporter(
        endpoint=settings.otel_exporter_otlp_endpoint,
        insecure=True,  # no TLS needed for local Jaeger
    )

    # BatchSpanProcessor buffers spans and exports them in batches
    # more efficient than exporting one span at a time
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    # Register as the global tracer provider
    # After this, any call to trace.get_tracer() anywhere in the app
    # uses this provider
    trace.set_tracer_provider(provider)

    # Auto-instrument FastAPI — creates a span for every HTTP request
    # automatically, no manual work needed in route files
    FastAPIInstrumentor.instrument_app(app)
