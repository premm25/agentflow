"""Whole-orchestration tracing with OpenTelemetry.

One trace per case. The harness runner opens every span (stage, agent, tool, LLM call, validator,
reporter); agents never touch a tracer. A human wait can last hours, so no span is ever held
open across one: the trace_id and root span_id are minted when the case is created and stored in
the case state, every graph segment attaches to them as a remote parent, the wait itself is
emitted as its own `hil.wait` span with explicit start/end times, and the root `case.orchestration`
span is emitted once, when the case finishes, with the case's real start and end times.
"""

import contextvars
import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator
from opentelemetry.trace import NonRecordingSpan, SpanContext, Status, StatusCode, TraceFlags

from roa.config import settings

log = logging.getLogger("roa.telemetry")

_forced_ids: contextvars.ContextVar[tuple[int, int] | None] = contextvars.ContextVar("roa_forced_ids", default=None)
_provider: TracerProvider | None = None


class _CaseIdGenerator(RandomIdGenerator):
    """Lets us emit the root span under ids that were minted (and handed to children) earlier."""

    def generate_trace_id(self) -> int:
        forced = _forced_ids.get()
        return forced[0] if forced else super().generate_trace_id()

    def generate_span_id(self) -> int:
        forced = _forced_ids.get()
        return forced[1] if forced else super().generate_span_id()


def init_tracing(service_name: str = "roa-orchestrator", exporter: SpanExporter | None = None) -> TracerProvider:
    global _provider
    if _provider is not None:
        return _provider
    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name, "service.namespace": "roa"}),
        id_generator=_CaseIdGenerator(),
    )
    if exporter is None and settings.otel_enabled:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=settings.otlp_endpoint, timeout=3)
    if exporter is not None:
        provider.add_span_processor(BatchSpanProcessor(exporter, schedule_delay_millis=500))
    trace.set_tracer_provider(provider)
    _provider = provider
    return provider


def tracer() -> trace.Tracer:
    return trace.get_tracer("roa.harness")


def flush(timeout_ms: int = 5000) -> None:
    if _provider is not None:
        try:
            _provider.force_flush(timeout_ms)
        except Exception as e:  # noqa: BLE001 - telemetry must never break the run
            log.warning("trace flush failed: %s", e)


def now_ns() -> int:
    return time.time_ns()


def new_case_ids() -> tuple[str, str]:
    gen = RandomIdGenerator()
    return format(gen.generate_trace_id(), "032x"), format(gen.generate_span_id(), "016x")


def _case_context(trace_id: str, root_span_id: str) -> otel_context.Context:
    sc = SpanContext(
        trace_id=int(trace_id, 16),
        span_id=int(root_span_id, 16),
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return trace.set_span_in_context(NonRecordingSpan(sc))


@contextmanager
def attach_case(trace_id: str, root_span_id: str) -> Iterator[None]:
    """Make every span opened inside a child of the case's root span."""
    token = otel_context.attach(_case_context(trace_id, root_span_id))
    try:
        yield
    finally:
        otel_context.detach(token)


@contextmanager
def span(name: str, kind: str = "CHAIN", **attrs: Any) -> Iterator[trace.Span]:
    """kind follows OpenInference span kinds: CHAIN, AGENT, LLM, TOOL, GUARDRAIL, EVALUATOR."""
    with tracer().start_as_current_span(name) as s:
        s.set_attribute("openinference.span.kind", kind)
        for k, v in attrs.items():
            if v is not None:
                s.set_attribute(k, v)
        try:
            yield s
        except Exception as e:
            s.set_status(Status(StatusCode.ERROR, str(e)[:300]))
            s.record_exception(e)
            raise


def emit_span(trace_id: str, root_span_id: str, name: str, start_ns: int, end_ns: int, kind: str = "CHAIN", **attrs: Any) -> None:
    """A span with explicit times, e.g. a human wait that spanned two graph segments."""
    s = tracer().start_span(name, context=_case_context(trace_id, root_span_id), start_time=start_ns)
    s.set_attribute("openinference.span.kind", kind)
    for k, v in attrs.items():
        if v is not None:
            s.set_attribute(k, v)
    s.end(end_time=end_ns)


def emit_root(trace_id: str, root_span_id: str, start_ns: int, end_ns: int, ok: bool, **attrs: Any) -> None:
    token = _forced_ids.set((int(trace_id, 16), int(root_span_id, 16)))
    try:
        s = tracer().start_span("case.orchestration", context=otel_context.Context(), start_time=start_ns)
    finally:
        _forced_ids.reset(token)
    s.set_attribute("openinference.span.kind", "CHAIN")
    for k, v in attrs.items():
        if v is not None:
            s.set_attribute(k, v)
    s.set_status(Status(StatusCode.OK if ok else StatusCode.ERROR))
    s.end(end_time=end_ns)
    flush()
