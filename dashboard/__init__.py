"""Standalone observability dashboard for the whole orchestration.

A separate process from the orchestrator: it exposes an OTLP/HTTP `/v1/traces` receiver, stores spans in
its own sqlite file, and serves aggregate metrics plus a trace waterfall UI. The orchestrator only
exports OTLP to it and has no other coupling, so any OTLP backend (Jaeger, Tempo, Phoenix) could replace it.
"""
