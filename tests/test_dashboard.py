"""The dashboard is a separate OTLP receiver: encode real orchestrator spans, post them, read the APIs."""

import importlib
import os

from fastapi.testclient import TestClient
from opentelemetry.exporter.otlp.proto.common._internal.trace_encoder import encode_spans

from roa import telemetry
from tests.conftest import MEM


def _client(tmp_path):
    os.environ["DASHBOARD_DB"] = str(tmp_path / "spans.db")
    import dashboard.server as server

    importlib.reload(server)
    return TestClient(server.app)


def test_otlp_roundtrip_and_metrics(tmp_path):
    MEM.clear()
    tid, root = telemetry.new_case_ids()
    with telemetry.attach_case(tid, root):
        with telemetry.span("agent.pricing_agent", "AGENT", **{"roa.case_id": "C-1", "roa.agent_id": "pricing_agent"}):
            with telemetry.span("tool.read_pricing_snapshot", "TOOL", **{"roa.case_id": "C-1"}):
                pass
        with telemetry.span("llm.planner", "LLM", **{"roa.case_id": "C-1", "gen_ai.request.model": "m",
                                                     "gen_ai.usage.input_tokens": 7, "gen_ai.usage.output_tokens": 3}):
            pass
        with telemetry.span("validate.pricing_agent", "EVALUATOR", **{"roa.case_id": "C-1", "roa.verdict": "FAIL",
                                                                      "roa.checks_failed": '["confidence_floor"]'}):
            pass
    t0 = telemetry.now_ns()
    telemetry.emit_span(tid, root, "hil.wait", t0, t0 + 5_000_000_000, "CHAIN",
                        **{"roa.case_id": "C-1", "roa.hil.stage": "validation", "roa.hil.decision": "continue_with_note"})
    telemetry.emit_root(tid, root, t0 - 1_000_000, t0 + 6_000_000_000, ok=True,
                        **{"roa.case_id": "C-1", "roa.final_stage": "COMPLETED", "roa.agents": "pricing_agent"})

    c = _client(tmp_path)
    payload = encode_spans(MEM.get_finished_spans()).SerializeToString()
    assert c.post("/v1/traces", content=payload, headers={"content-type": "application/x-protobuf"}).status_code == 200

    lst = c.get("/api/traces").json()
    assert len(lst) == 1 and lst[0]["case_id"] == "C-1" and lst[0]["stage"] == "COMPLETED" and lst[0]["complete"]
    assert lst[0]["hil_count"] == 1 and lst[0]["hil_wait_ms"] >= 5000 and lst[0]["tokens"] == 10

    d = c.get(f"/api/traces/{tid}").json()
    names = [s["name"] for s in d["spans"]]
    assert names[0] == "case.orchestration" and {"agent.pricing_agent", "tool.read_pricing_snapshot", "hil.wait"} <= set(names)
    tool = next(s for s in d["spans"] if s["name"] == "tool.read_pricing_snapshot")
    assert tool["depth"] == 2  # root -> agent -> tool

    m = c.get("/api/metrics").json()
    assert m["verdicts"] == {"FAIL": 1} and m["failed_checks"] == {"confidence_floor": 1}
    assert m["hil_by_stage"] == {"validation": 1} and m["llm_calls"] == 1
    assert c.get("/").status_code == 200
