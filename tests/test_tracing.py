from observability.tracing import init_tracing, get_tracer

def test_init_tracing_returns_tracer():
    init_tracing("test-service")
    tracer = get_tracer()
    assert tracer is not None

def test_span_context_manager_works():
    init_tracing("test-service")
    tracer = get_tracer()
    with tracer.start_as_current_span("dummy") as span:
        span.set_attribute("k", "v")

def test_init_tracing_survives_unreachable_exporter():
    init_tracing("test-service", endpoint="http://localhost:9/does-not-exist")
