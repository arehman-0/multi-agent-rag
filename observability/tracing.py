"""OpenTelemetry setup. init_tracing() is idempotent and never raises."""
import logging
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

import config

_initialized = False
log = logging.getLogger(__name__)


def init_tracing(service_name: str = config.SERVICE_NAME, endpoint: str | None = None) -> None:
    global _initialized
    if _initialized:
        return
    try:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint or config.JAEGER_ENDPOINT)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _initialized = True
    except Exception as exc:
        log.warning("Tracing init failed (continuing without traces): %s", exc)
        _initialized = True


def get_tracer():
    return trace.get_tracer(config.SERVICE_NAME)
