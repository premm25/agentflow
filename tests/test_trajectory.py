"""Trajectory = the path a case actually took. Build a realistic case (planner, two agents, a validation failure,
a human retry, a second attempt, approval), push it through the OTLP receiver, and read the trajectory back."""

import importlib
import json
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


def _agent(agent_id, attempt, required, called, status="DONE"):
    with telemetry.span(f"agent.{agent_id}", "AGENT", **{"roa.case_id": "C-T", "roa.agent_id": agent_id, "roa.attempt": attempt}) as sp:
        for t in called:
            with telemetry.span(f"tool.{t}", "TOOL", **{"roa.case_id": "C-T", "roa.tool": t, "roa.call_id": f"CALL-{t}"}):
                pass
        sp.set_attribute("roa.status", status)
        sp.set_attribute("roa.evidence_count", len(called))
        sp.set_attribute("roa.required_tools", json.dumps(required))
        sp.set_attribute("roa.tools_called", json.dumps(called))


def _case():
    MEM.clear()
    tid, root = telemetry.new_case_ids()
    t0 = telemetry.now_ns()
    with telemetry.attach_case(tid, root):
        with telemetry.span("intake", "GUARDRAIL", **{"roa.case_id": "C-T", "roa.guardrail.passed": True}):
            pass
        with telemetry.span("planner_agent", "AGENT", **{"roa.case_id": "C-T", "roa.understanding.hint": "pricing",
                                                         "roa.understanding.horizon": "LONG_TERM", "roa.plan.status": "ok",
                                                         "roa.plan.case_type": "pricing", "roa.plan.source": "llm",
                                                         "roa.plan.agents": json.dumps(["pricing_agent", "forecast_agent"])}):
            pass
        full = ["read_pricing_snapshot", "read_occupancy", "read_competitor_rates"]
        _agent("pricing_agent", 1, full, ["read_rate_configuration"])          # inconclusive: wrong tool, misses 3 required
        _agent("forecast_agent", 1, ["read_occupancy_forecast"], ["read_occupancy_forecast"])
        with telemetry.span("validation_layer", "CHAIN", **{"roa.case_id": "C-T"}):
            with telemetry.span("validate.pricing_agent", "EVALUATOR", **{"roa.attempt": 1, "roa.verdict": "FAIL",
                                                                          "roa.checks_failed": json.dumps(["required_tools_called"])}):
                pass
            with telemetry.span("validate.forecast_agent", "EVALUATOR", **{"roa.attempt": 1, "roa.verdict": "PASS",
                                                                           "roa.judge": "PASS"}):
                pass
    now = telemetry.now_ns()
    telemetry.emit_span(tid, root, "hil.wait", now, now + 2_000_000_000, "CHAIN",
                        **{"roa.case_id": "C-T", "roa.hil.stage": "validation", "roa.hil.decision": "retry:pricing_agent", "roa.hil.type": "failure_review"})
    with telemetry.attach_case(tid, root):
        _agent("pricing_agent", 2, ["read_pricing_snapshot", "read_occupancy", "read_competitor_rates"],
               ["read_pricing_snapshot", "read_occupancy", "read_competitor_rates", "read_lrv"])  # now complete, plus one extra call
        with telemetry.span("validation_layer", "CHAIN", **{"roa.case_id": "C-T"}):
            with telemetry.span("validate.pricing_agent", "EVALUATOR", **{"roa.attempt": 2, "roa.verdict": "PASS"}):
                pass
        with telemetry.span("reporting_layer", "CHAIN", **{"roa.case_id": "C-T"}):
            pass
    end = telemetry.now_ns()
    telemetry.emit_span(tid, root, "hil.wait", end, end + 1_000_000_000, "CHAIN",
                        **{"roa.case_id": "C-T", "roa.hil.stage": "report", "roa.hil.decision": "APPROVED", "roa.hil.type": "final_approval"})
    telemetry.emit_root(tid, root, t0, end + 2_000_000_000, ok=True, **{"roa.case_id": "C-T", "roa.final_stage": "COMPLETED"})
    return tid


def test_trajectory_reconstructs_the_path_with_retry_and_tool_correctness(tmp_path):
    tid = _case()
    c = _client(tmp_path)
    c.post("/v1/traces", content=encode_spans(MEM.get_finished_spans()).SerializeToString(),
           headers={"content-type": "application/x-protobuf"})
    r = c.get(f"/api/traces/{tid}/trajectory")
    assert r.status_code == 200
    d = r.json()
    assert d["case_id"] == "C-T"

    kinds = [p["type"] for p in d["phases"]]
    assert kinds == ["intake", "planner", "agent", "agent", "validation", "hil", "agent", "validation", "report", "hil"]

    planner = d["phases"][1]
    assert planner["plan_agents"] == ["pricing_agent", "forecast_agent"] and planner["hint"] == "pricing" and planner["horizon"] == "LONG_TERM"

    first, forecast, retry = d["phases"][2], d["phases"][3], d["phases"][6]
    assert (first["agent_id"], first["attempt"], first["tools_correct"]) == ("pricing_agent", 1, False)
    assert first["missing_tools"] == ["read_pricing_snapshot", "read_occupancy", "read_competitor_rates"]
    assert first["extra_tools"] == ["read_rate_configuration"] and [t["tool"] for t in first["tools"]] == ["read_rate_configuration"]
    assert forecast["tools_correct"] and forecast["missing_tools"] == []
    assert (retry["attempt"], retry["tools_correct"], retry["extra_tools"]) == (2, True, ["read_lrv"])
    assert [t["tool"] for t in retry["tools"]] == ["read_pricing_snapshot", "read_occupancy", "read_competitor_rates", "read_lrv"]  # order kept

    v1 = d["phases"][4]["verdicts"]
    assert [(v["agent_id"], v["verdict"]) for v in v1] == [("pricing_agent", "FAIL"), ("forecast_agent", "PASS")]
    assert v1[0]["failed_checks"] == ["required_tools_called"] and v1[1]["judge"] == "PASS"
    hil = [p for p in d["phases"] if p["type"] == "hil"]
    assert [(h["stage"], h["decision"]) for h in hil] == [("validation", "retry:pricing_agent"), ("report", "APPROVED")]

    m = d["metrics"]
    assert (m["agent_runs"], m["distinct_agents"], m["retries"]) == (3, 2, 1)
    assert m["tool_calls"] == 1 + 1 + 4 and m["missing_required_tool_calls"] == 3 and m["extra_tool_calls"] == 2
    assert abs(m["tool_correctness"] - 2 / 3) < 1e-9
    assert m["first_try_validation_pass_rate"] == 0.5  # pricing failed first try, forecast passed
    assert m["human_interventions"] == 2 and m["human_wait_ms"] >= 3000 and m["final_stage"] == "COMPLETED"

    texts = [p["text"] for p in d["path"]]
    assert texts[2] == "pricing_agent #1" and "pricing_agent #2" in texts and any(t.startswith("human (validation): retry") for t in texts)
    assert next(p for p in d["path"] if p["text"] == "pricing_agent #2")["retry"] is True


def test_trajectory_of_unknown_trace_is_404_and_config_is_served(tmp_path):
    c = _client(tmp_path)
    assert c.get("/api/traces/nope/trajectory").status_code == 404
    assert c.get("/api/config").json()["orchestrator_url"].startswith("http")
