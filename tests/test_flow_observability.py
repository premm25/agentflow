"""Observability of the selected flow: real spans from a real graph run are posted to the dashboard's OTLP receiver, and the
trace list, the trajectory and the metrics must show the same flow the runtime selected. Observability records what happened;
it does not decide anything."""

import importlib
import os

import pytest
from fastapi.testclient import TestClient
from opentelemetry.exporter.otlp.proto.common._internal.trace_encoder import encode_spans

from roa import state, telemetry
from tests.conftest import MEM
from tests.flow_helpers import RoutedLLM
from tests.test_flows import answer, env, new_case, run, start


@pytest.fixture()
def fake(bundle, monkeypatch):
    return RoutedLLM(bundle).install(monkeypatch)


def _dashboard(tmp_path):
    os.environ["DASHBOARD_DB"] = str(tmp_path / "spans.db")
    import dashboard.server as server

    importlib.reload(server)
    return TestClient(server.app)


def _run_cases(tmp_path, bundle, texts, approve=True):
    MEM.clear()

    async def scenario():
        async with env(tmp_path, bundle) as (graph, store):
            ids = []
            for t in texts:
                cid = new_case(bundle, t)
                await start(graph, cid)
                if approve:
                    await answer(graph, cid, "APPROVED")
                ids.append(cid)
            telemetry.flush()
            return ids

    return run(scenario())


def test_the_dashboard_shows_the_flow_each_case_followed(tmp_path, bundle, fake):
    texts = ["Why is my rate only 1000 on 15 August?", "I received an overbooking notification for a booking I thought was confirmed.",
             "The occupancy forecast for next week looks off compared to last year."]
    ids = _run_cases(tmp_path, bundle, texts)
    c = _dashboard(tmp_path)
    assert c.post("/v1/traces", content=encode_spans(MEM.get_finished_spans()).SerializeToString(),
                  headers={"content-type": "application/x-protobuf"}).status_code == 200

    by_case = {t["case_id"]: t for t in c.get("/api/traces").json()}
    assert {cid: by_case[cid]["flow"] for cid in ids} == dict(zip(ids, ["pricing", "overbooking", "forecast"]))  # different cases, different flows
    assert {cid: by_case[cid]["intent"] for cid in ids} == dict(zip(ids, ["pricing", "overbooking", "forecast"]))
    assert all(by_case[cid]["complete"] and by_case[cid]["stage"] == "COMPLETED" for cid in ids)

    cs = state.get(ids[0])
    traj = c.get(f"/api/traces/{cs.trace_id}/trajectory").json()
    kinds = [p["type"] for p in traj["phases"]]
    assert kinds[:4] == ["intake", "planner", "flow", "agent"]  # intake, planner agent, flow selection, then the worker
    flow = next(p for p in traj["phases"] if p["type"] == "flow")
    sel = cs.flow_selection
    assert (flow["flow_id"], flow["status"], flow["source"], flow["intent"], flow["flow_name"], flow["version"]) == \
        (sel.flow_id, sel.status, sel.source, sel.intent, sel.name, sel.version)
    assert flow["reason"] == sel.reason and flow["agents"] == sel.participating_agents  # the trace agrees with the canonical case state
    planner = next(p for p in traj["phases"] if p["type"] == "planner")
    assert planner["flow_id"] == "pricing" and planner["case_type"] == "pricing" and planner["hint"] == "pricing"
    assert traj["metrics"]["flow_id"] == "pricing" and traj["metrics"]["final_stage"] == "COMPLETED"
    assert next(p for p in traj["path"] if p["kind"] == "flow") == {"kind": "flow", "text": "flow: pricing", "ok": True}
    assert [p["text"] for p in traj["path"]][:3] == ["intake guardrails", "planner → pricing_agent", "flow: pricing"]


def test_a_case_in_progress_already_shows_its_flow_before_the_root_span_exists(tmp_path, bundle, fake):
    (cid,) = _run_cases(tmp_path, bundle, ["The last room value looks stuck, please check."], approve=False)  # paused at final approval
    c = _dashboard(tmp_path)
    c.post("/v1/traces", content=encode_spans(MEM.get_finished_spans()).SerializeToString(), headers={"content-type": "application/x-protobuf"})
    (t,) = [t for t in c.get("/api/traces").json() if t["case_id"] == cid]
    assert t["stage"] == "IN_PROGRESS" and not t["complete"]  # no root span yet: the case has not ended
    assert (t["flow"], t["intent"]) == ("lrv", "lrv")  # the flow comes from the flow-resolution span


def test_an_unresolved_case_is_shown_as_unresolved_not_as_a_flow(tmp_path, bundle, fake):
    (cid,) = _run_cases(tmp_path, bundle, ["Something strange happened yesterday, can you please take a look at it soon?"], approve=False)
    c = _dashboard(tmp_path)
    c.post("/v1/traces", content=encode_spans(MEM.get_finished_spans()).SerializeToString(), headers={"content-type": "application/x-protobuf"})
    cs = state.get(cid)
    traj = c.get(f"/api/traces/{cs.trace_id}/trajectory").json()
    flow = next(p for p in traj["phases"] if p["type"] == "flow")
    assert (flow["flow_id"], flow["status"], flow["source"]) == (None, "UNRESOLVED", "unresolved")
    assert next(p for p in traj["path"] if p["kind"] == "flow") == {"kind": "flow", "text": "flow: unresolved by registry", "ok": False}
    assert [t for t in c.get("/api/traces").json() if t["case_id"] == cid][0]["flow"] is None


def test_existing_traces_without_a_flow_span_still_render(tmp_path):
    """Backward compatibility: traces recorded before flow resolution existed have no flow span or attributes."""
    MEM.clear()
    tid, root = telemetry.new_case_ids()
    with telemetry.attach_case(tid, root):
        with telemetry.span("planner_agent", "AGENT", **{"roa.case_id": "C-OLD", "roa.understanding.hint": "pricing", "roa.plan.status": "ok",
                                                         "roa.plan.case_type": "pricing", "roa.plan.source": "llm", "roa.plan.agents": '["pricing_agent"]'}):
            pass
    t0 = telemetry.now_ns()
    telemetry.emit_root(tid, root, t0, t0 + 1_000_000, ok=True, **{"roa.case_id": "C-OLD", "roa.final_stage": "COMPLETED"})
    c = _dashboard(tmp_path)
    c.post("/v1/traces", content=encode_spans(MEM.get_finished_spans()).SerializeToString(), headers={"content-type": "application/x-protobuf"})
    (t,) = c.get("/api/traces").json()
    assert t["flow"] is None and t["intent"] is None  # no flow evidence recorded: shown as absent, nothing is invented
    traj = c.get(f"/api/traces/{tid}/trajectory").json()
    assert [p["type"] for p in traj["phases"]] == ["planner"] and traj["phases"][0]["flow_id"] is None and traj["metrics"]["flow_id"] is None


def test_a_flow_a_human_chose_shows_in_the_dashboard_while_the_case_is_still_running(tmp_path, bundle, fake):
    MEM.clear()

    async def scenario():
        async with env(tmp_path, bundle) as (graph, store):
            cid = new_case(bundle, "Something strange happened yesterday, can you please take a look at it soon?")
            await start(graph, cid)  # the registry cannot decide: a human is asked
            await answer(graph, cid, "continue_with_agents", agent_ids=["forecast_agent"])
            telemetry.flush()
            return cid

    cid = run(scenario())
    assert state.get(cid).flow_selection.source == "human"
    c = _dashboard(tmp_path)
    c.post("/v1/traces", content=encode_spans(MEM.get_finished_spans()).SerializeToString(), headers={"content-type": "application/x-protobuf"})
    (t,) = [t for t in c.get("/api/traces").json() if t["case_id"] == cid]
    assert (t["flow"], t["stage"]) == ("forecast", "IN_PROGRESS")  # from the agent span: the flow span only said "unresolved"
    traj = c.get(f"/api/traces/{state.get(cid).trace_id}/trajectory").json()
    assert traj["metrics"]["flow_id"] == "forecast"


def test_the_dashboard_trajectory_calls_the_customer_facing_step_response(tmp_path, bundle, fake):
    ids = _run_cases(tmp_path, bundle, ["Why is my rate only 1000 on 15 August?"])
    c = _dashboard(tmp_path)
    c.post("/v1/traces", content=encode_spans(MEM.get_finished_spans()).SerializeToString(), headers={"content-type": "application/x-protobuf"})
    traj = c.get(f"/api/traces/{state.get(ids[0]).trace_id}/trajectory").json()
    resp = next(p for p in traj["phases"] if p["type"] == "report")  # the phase type (an identifier) is unchanged
    assert resp["label"] == "Response"
    hil = [p for p in traj["phases"] if p["type"] == "hil"]
    assert hil[-1]["stage"] == "report" and hil[-1]["label"] == "Human: response"  # data keeps 'report'; the label shows 'response'
    texts = [p["text"] for p in traj["path"]]
    assert "response" in texts and "human (response): APPROVED" in texts and "reporting layer" not in texts
