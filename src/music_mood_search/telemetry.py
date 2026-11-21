"""OpenTelemetry bootstrap for API, batch source and workers.

Works without the Collector: exporters enqueue asynchronously and drop on failure.
Set OTEL_SDK_DISABLED=true to avoid all exporter traffic in local development.
"""

from __future__ import annotations

import os
import socket
from functools import lru_cache


def _resource():
    from opentelemetry.sdk.resources import Resource

    attrs = {
        "service.name": os.getenv("OTEL_SERVICE_NAME", "music-mood-search"),
        "service.version": os.getenv("SERVICE_VERSION", "0.2.0"),
        "deployment.environment.name": os.getenv("ENVIRONMENT", "development"),
        "service.instance.id": socket.gethostname(),
    }
    # Resource.create also reads OTEL_RESOURCE_ATTRIBUTES.
    return Resource.create(attrs)


@lru_cache(maxsize=1)
def configure_telemetry():
    """Configure providers once per process; return (tracer_provider, meter)."""
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.metrics.view import ExplicitBucketHistogramAggregation, View
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.trace.sampling import (
        ALWAYS_ON,
        ParentBased,
        TraceIdRatioBased,
    )

    sampler_name = os.getenv("OTEL_TRACES_SAMPLER", "parentbased_always_on").lower()
    sampler_arg = float(os.getenv("OTEL_TRACES_SAMPLER_ARG", "1.0"))
    sampler = (
        ParentBased(TraceIdRatioBased(max(0.0, min(sampler_arg, 1.0))))
        if sampler_name in {"parentbased_traceidratio", "traceidratio"}
        else ParentBased(ALWAYS_ON)
    )
    resource = _resource()
    tracer_provider = TracerProvider(resource=resource, sampler=sampler)
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").rstrip("/")
    enabled = os.getenv("OTEL_SDK_DISABLED", "false").lower() not in {
        "1",
        "true",
        "yes",
    }

    readers = []
    if enabled and endpoint:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
            OTLPMetricExporter,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces", timeout=2),
                schedule_delay_millis=1000,
                max_queue_size=1024,
            )
        )
        readers.append(
            PeriodicExportingMetricReader(
                OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics", timeout=2),
                export_interval_millis=5000,
            )
        )

    trace.set_tracer_provider(tracer_provider)
    delay = float(os.getenv("INFERENCE_DELAY_S", "1.0"))
    inference_buckets = sorted(
        set(
            [
                0.05,
                0.1,
                0.25,
                0.5,
                delay * 0.5,
                delay,
                delay * 1.5,
                delay * 2,
                delay * 4,
                10.0,
            ]
        )
    )
    http_buckets = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    meter_provider = MeterProvider(
        resource=resource,
        metric_readers=readers,
        views=[
            View(
                instrument_name="inference_duration_seconds",
                aggregation=ExplicitBucketHistogramAggregation(
                    boundaries=inference_buckets
                ),
            ),
            View(
                instrument_name="http_duration_seconds",
                aggregation=ExplicitBucketHistogramAggregation(boundaries=http_buckets),
            ),
        ],
    )
    metrics.set_meter_provider(meter_provider)
    meter = metrics.get_meter("music_mood_search", "0.2.0")
    return tracer_provider, meter


@lru_cache(maxsize=1)
def instruments():
    """No high-cardinality attributes: never attach task_id or trace_id."""
    _, meter = configure_telemetry()
    return {
        "http_requests": meter.create_counter("http_requests", unit="1"),
        "http_duration": meter.create_histogram("http_duration_seconds", unit="s"),
        "published": meter.create_counter("tasks_published", unit="1"),
        "rejected": meter.create_counter("broker_publish_rejected", unit="1"),
        "source_wait": meter.create_counter("source_wait_seconds", unit="s"),
        "processed": meter.create_counter("tasks_processed", unit="1"),
        "inflight": meter.create_up_down_counter("worker_inflight", unit="1"),
        "redelivered": meter.create_counter("worker_redeliveries", unit="1"),
        "inference": meter.create_histogram("inference_duration_seconds", unit="s"),
    }


def record_process_memory():
    """Export per-process Linux RSS rather than host-level total RAM."""
    from opentelemetry.metrics import Observation

    _, meter = configure_telemetry()

    def observe(_options):
        try:
            with open("/proc/self/statm", encoding="ascii") as f:
                rss_pages = int(f.read().split()[1])
            yield Observation(rss_pages * os.sysconf("SC_PAGE_SIZE"))
        except (OSError, ValueError, IndexError):
            return

    meter.create_observable_gauge(
        "process_resident_memory_bytes",
        callbacks=[observe],
        unit="By",
        description="Resident set size of this service process",
    )


def flush_telemetry():
    """Useful for short-lived batch source, call at the end of a run."""
    from opentelemetry import metrics, trace

    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush(timeout_millis=2000)
    meter_provider = metrics.get_meter_provider()
    if hasattr(meter_provider, "force_flush"):
        meter_provider.force_flush(timeout_millis=2000)


async def traced_inference(
    predict, embeddings, *, delay_s: float, alias: str, version: str
):
    """Record inference under the current trace, preserving context across threads."""
    import asyncio
    import contextvars
    from time import perf_counter

    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    with trace.get_tracer(__name__).start_as_current_span(
        "inference",
        attributes={"model.alias": str(alias), "model.version": str(version)},
    ) as span:
        start = perf_counter()
        try:
            await asyncio.sleep(delay_s)
            ctx = contextvars.copy_context()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, ctx.run, predict, embeddings)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise
        finally:
            instruments()["inference"].record(perf_counter() - start)


async def traced_db_execute(pool, query, *args):
    """Create a child span around storing the task result."""
    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    with trace.get_tracer(__name__).start_as_current_span("db.write_result") as span:
        try:
            return await pool.execute(query, *args)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR))
            raise
