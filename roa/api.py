"""Orchestrator service. ONE process hosts the planner, every worker agent, the validation layer and
the reporting layer. Endpoints: case intake, live state, the human-in-the-loop decision, reports, and
read-only views of the harness and of the runtime files agents write.
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel

from roa import llm, state, telemetry
from roa.config import PROJECT_ROOT, settings
from roa.graph import build_graph, validate_human_decision
from roa.harness import GuardedStore, load_harness
from roa.logging_setup import configure_logging
from roa.models import Case, CaseState, CaseStatus, HILDecision
from roa.timeline import build_timeline
from roa.tools import check_registry as check_tool_registry

log = logging.getLogger("roa.api")

_graph = None
_bundle = None
_store: GuardedStore | None = None
_saver_cm = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _bundle, _store, _saver_cm
    configure_logging(settings.log_level)
    telemetry.init_tracing()
    state.init_db()
    _bundle = load_harness(settings.harness_dir)
    check_tool_registry(_bundle)
    recovered = state.recover_after_restart()
    if recovered:
        log.warning("recovered after restart: %s", recovered)
    _store = GuardedStore(settings.runtime_dir, settings.harness_dir)
    _saver_cm = AsyncSqliteSaver.from_conn_string(str(settings.data_dir / "checkpoints.sqlite"))
    saver = await _saver_cm.__aenter__()
    _graph = build_graph(saver, _bundle, _store)
    log.info("orchestrator started: harness v%s hash %s, %d agents in-process",
             _bundle.version, _bundle.hash[:12], len(_bundle.agents))
    yield
    await _saver_cm.__aexit__(None, None, None)
    await llm.aclose()
    telemetry.flush()


app = FastAPI(title="AgentFlow Orchestrator", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class CreateCaseRequest(BaseModel):
    description: str
    property_name: Optional[str] = None
    source: str = "demo_ui"


class HILRequestBody(BaseModel):
    hil_id: str
    decision: str
    comment: Optional[str] = None
    agent_id: Optional[str] = None
    agent_ids: list[str] = []


def _config(case_id: str) -> dict:
    return {"configurable": {"thread_id": case_id}, "recursion_limit": 200}


async def _run_graph(case_id: str, resume: dict | None = None) -> None:
    try:
        if resume is not None:
            await _graph.ainvoke(Command(resume=resume), config=_config(case_id))
        else:
            await _graph.ainvoke({"case_id": case_id}, config=_config(case_id))
    except Exception as e:  # noqa: BLE001 - a run failure must surface on the case, not vanish
        log.exception("graph run failed for %s", case_id)
        try:
            state.set_stage(case_id, CaseStatus.FAILED, f"internal error: {type(e).__name__}: {e}")
            cs = state.get(case_id)
            telemetry.emit_root(cs.trace_id, cs.root_span_id, cs.started_ns, telemetry.now_ns(), ok=False,
                                **{"roa.case_id": case_id, "roa.final_stage": "FAILED", "roa.final_reason": str(e)[:300]})
        except Exception:  # noqa: BLE001
            pass


@app.get("/health")
async def health():
    return {"status": "ok", "harness_version": _bundle.version, "harness_hash": _bundle.hash,
            "agents_in_process": list(_bundle.agents)}


@app.post("/cases")
async def create_case(req: CreateCaseRequest):
    case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
    trace_id, root_span_id = telemetry.new_case_ids()
    case = Case(case_id=case_id, source=req.source, description=req.description, property_name=req.property_name)
    cs = CaseState(case=case, trace_id=trace_id, root_span_id=root_span_id, started_ns=telemetry.now_ns(),
                   harness_hash=_bundle.hash, harness_version=_bundle.version)
    state.create_case(cs)
    _store.write_json("orchestrator", f"cases/{case_id}/case.json", case.model_dump(mode="json"))
    _store.log("orchestrator", f"cases/{case_id}/orchestrator/log.jsonl", "case_received", trace_id=trace_id,
               harness_hash=_bundle.hash)
    asyncio.create_task(_run_graph(case_id))
    return {"case_id": case_id, "trace_id": trace_id}


@app.get("/cases")
async def list_cases():
    return [cs.model_dump(mode="json") for cs in state.list_cases()]


@app.get("/cases/{case_id}")
async def get_case(case_id: str):
    cs = state.get(case_id)
    if cs is None:
        raise HTTPException(404, "case not found")
    return cs.model_dump(mode="json")


@app.get("/cases/{case_id}/timeline")
async def get_timeline(case_id: str):
    """The end-to-end execution timeline (trigger, intent, flow, plan, workers, validation, human, report, outcome), built from
    the same canonical case state the graph writes. Polled by the UI while a case runs."""
    cs = state.get(case_id)
    if cs is None:
        raise HTTPException(404, "case not found")
    return build_timeline(cs, _bundle)


@app.get("/flows")
async def list_flows():
    """The Flow Registry (harness/flows/flows.json): every registered flow with its intents, agents and order."""
    reg = _bundle.flow_registry()
    return [s.to_dict(reg.task_order(s)) for s in reg.all()]


@app.get("/flows/{flow_id}")
async def get_flow(flow_id: str):
    reg = _bundle.flow_registry()
    spec = reg.get(flow_id)
    if spec is None:
        raise HTTPException(404, "flow not found")
    return spec.to_dict(reg.task_order(spec))


@app.post("/cases/{case_id}/hil")
async def decide(case_id: str, body: HILRequestBody):
    cs = state.get(case_id)
    if cs is None:
        raise HTTPException(404, "case not found")
    pending = cs.pending_hil
    if pending is None or pending.hil_id != body.hil_id:
        raise HTTPException(409, "no such pending human request for this case")
    if pending.resuming:
        raise HTTPException(409, "this request is already being resolved")
    err = validate_human_decision(_bundle, pending, body.decision, body.agent_ids)
    if err:
        raise HTTPException(422, err)
    if state.begin_resume(case_id, body.hil_id) is None:
        raise HTTPException(409, "this request is already being resolved")
    asyncio.create_task(_run_graph(case_id, resume={"decision": body.decision, "comment": body.comment,
                                                    "agent_id": body.agent_id, "agent_ids": body.agent_ids}))
    return {"status": "accepted"}


@app.get("/cases/{case_id}/report")
async def get_report(case_id: str):
    if state.get(case_id) is None:
        raise HTTPException(404, "case not found")
    md = _store.read(f"cases/{case_id}/report/report.md")
    if not md:
        raise HTTPException(404, "report not generated yet")
    return {"markdown": md, "json": _store.read(f"cases/{case_id}/report/report.json")}


@app.get("/cases/{case_id}/files")
async def list_files(case_id: str):
    """Everything the run wrote under runtime/ for this case (logs, results, validation, report)."""
    if state.get(case_id) is None:
        raise HTTPException(404, "case not found")
    root = _store.path_of(f"cases/{case_id}")
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


@app.get("/cases/{case_id}/files/{path:path}")
async def read_file(case_id: str, path: str):
    root = _store.path_of(f"cases/{case_id}").resolve()
    target = (root / path).resolve()
    if root not in target.parents or not target.is_file():
        raise HTTPException(404, "file not found")
    return {"path": path, "content": target.read_text(encoding="utf-8")}


@app.get("/agents")
async def list_agents():
    return [{"agent_id": a.agent_id, "display_name": a.display_name, "purpose": a.purpose, "version": a.version,
             "tools_allowed": a.tools_allowed, "expected_evidence": a.expected_evidence,
             "memory": _store.read(f"agents/{a.agent_id}/MEMORY.md")[-1500:]} for a in _bundle.agents.values()]


@app.get("/harness")
async def harness_view():
    try:
        _bundle.verify()
        intact = True
    except Exception:  # noqa: BLE001
        intact = False
    return {"version": _bundle.version, "hash": _bundle.hash, "intact": intact, "manifest": _bundle.manifest,
            "flows": _bundle.flows, "canonical_order": _bundle.canonical_order, "step_policy": _bundle.step_policy}


@app.get("/config")
async def public_config():
    return {"dashboard_url": settings.dashboard_url}


@app.get("/")
async def root():
    return RedirectResponse(url="/ui/")


app.mount("/ui", StaticFiles(directory=str(PROJECT_ROOT / "web"), html=True), name="ui")
