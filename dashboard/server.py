import gzip
import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2

from dashboard.trajectory import build_trajectory

HERE = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("DASHBOARD_DB", HERE / "data" / "spans.db"))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()

app = FastAPI(title="AgentFlow Observability Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@contextmanager
def conn():
    c = sqlite3.connect(str(DB_PATH), timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    with conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS spans (
            trace_id TEXT, span_id TEXT, parent_id TEXT, name TEXT, kind TEXT,
            start_ns INTEGER, end_ns INTEGER, status INTEGER, status_msg TEXT,
            service TEXT, case_id TEXT, attrs TEXT,
            PRIMARY KEY (trace_id, span_id))""")
        c.execute("CREATE INDEX IF NOT EXISTS ix_trace ON spans(trace_id)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_case ON spans(case_id)")


init_db()


def _val(v):
    which = v.WhichOneof("value")
    if which is None:
        return None
    if which == "array_value":
        return [_val(x) for x in v.array_value.values]
    if which == "kvlist_value":
        return {kv.key: _val(kv.value) for kv in v.kvlist_value.values}
    if which == "bytes_value":
        return v.bytes_value.hex()
    return getattr(v, which)


@app.post("/v1/traces")
async def receive_traces(request: Request):
    body = await request.body()
    if request.headers.get("content-encoding") == "gzip":
        body = gzip.decompress(body)
    req = trace_service_pb2.ExportTraceServiceRequest()
    try:
        req.ParseFromString(body)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"invalid OTLP payload: {e}")
    rows = []
    for rs in req.resource_spans:
        res = {kv.key: _val(kv.value) for kv in rs.resource.attributes}
        service = res.get("service.name", "unknown")
        for ss in rs.scope_spans:
            for sp in ss.spans:
                attrs = {kv.key: _val(kv.value) for kv in sp.attributes}
                events = [{"name": e.name, "time_ns": e.time_unix_nano,
                           "attrs": {kv.key: _val(kv.value) for kv in e.attributes}} for e in sp.events]
                if events:
                    attrs["_events"] = events
                rows.append((sp.trace_id.hex(), sp.span_id.hex(), sp.parent_span_id.hex() or None, sp.name,
                             attrs.get("openinference.span.kind"), sp.start_time_unix_nano, sp.end_time_unix_nano,
                             sp.status.code, sp.status.message, service, attrs.get("roa.case_id"), json.dumps(attrs)))
    with _lock, conn() as c:
        c.executemany("INSERT OR REPLACE INTO spans VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    return Response(trace_service_pb2.ExportTraceServiceResponse().SerializeToString(),
                    media_type="application/x-protobuf")


def _load(where: str = "", args: tuple = ()):
    with conn() as c:
        return [dict(r) for r in c.execute(f"SELECT * FROM spans {where}", args).fetchall()]


def _ms(s):
    return (s["end_ns"] - s["start_ns"]) / 1e6


def _pct(vals, p):
    if not vals:
        return 0
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(round(p * (len(vals) - 1))))]


def _summarize(spans):
    by_trace: dict[str, list] = {}
    for s in spans:
        by_trace.setdefault(s["trace_id"], []).append(s)
    out = []
    for tid, ss in by_trace.items():
        attrs = {s["span_id"]: json.loads(s["attrs"]) for s in ss}
        root = next((s for s in ss if s["name"] == "case.orchestration"), None)
        start = min(s["start_ns"] for s in ss)
        end = max(s["end_ns"] for s in ss)
        waits = sum(_ms(s) for s in ss if s["name"] == "hil.wait")
        llm = [s for s in ss if s["name"].startswith("llm.")]
        ra = attrs[root["span_id"]] if root else {}
        stage = ra.get("roa.final_stage") or "IN_PROGRESS"  # root span is only emitted when the case ends
        ordered = sorted(ss, key=lambda s: s["start_ns"])

        def first(key):  # earliest span that recorded it: works while the case runs, and for a flow a human chose after the registry could not
            return next((attrs[s["span_id"]].get(key) for s in ordered if attrs[s["span_id"]].get(key)), None)
        out.append({
            "trace_id": tid,
            "case_id": next((s["case_id"] for s in ss if s["case_id"]), None),
            "stage": stage,
            "start_ns": start,
            "duration_ms": (end - start) / 1e6,
            "active_ms": max(0.0, (end - start) / 1e6 - waits),
            "hil_wait_ms": waits,
            "hil_count": len([s for s in ss if s["name"] == "hil.wait"]),
            "spans": len(ss),
            "errors": len([s for s in ss if s["status"] == 2]),
            "llm_calls": len(llm),
            "tokens": sum(int(json.loads(s["attrs"]).get("gen_ai.usage.input_tokens", 0)) +
                          int(json.loads(s["attrs"]).get("gen_ai.usage.output_tokens", 0)) for s in llm),
            "agents": ra.get("roa.agents") or ",".join(sorted({s["name"][6:] for s in ss if s["name"].startswith("agent.")})),
            "complete": root is not None,
            "flow": ra.get("roa.flow.id") or first("roa.flow.id"),
            "intent": ra.get("roa.intent") or first("roa.intent"),
        })
    return sorted(out, key=lambda t: -t["start_ns"])


@app.get("/api/traces")
def traces(limit: int = 100):
    return _summarize(_load())[:limit]


@app.get("/api/traces/{trace_id}")
def trace_detail(trace_id: str):
    spans = _load("WHERE trace_id = ?", (trace_id,))
    if not spans:
        raise HTTPException(404, "trace not found")
    summary = _summarize(spans)[0]  # before attrs are decoded in place below
    ids = {s["span_id"] for s in spans}
    children: dict = {}
    for s in spans:
        s["attrs"] = json.loads(s["attrs"])
        parent = s["parent_id"] if s["parent_id"] in ids else None
        children.setdefault(parent, []).append(s)
    ordered = []

    def walk(parent, depth):
        for s in sorted(children.get(parent, []), key=lambda x: x["start_ns"]):
            s["depth"] = depth
            ordered.append(s)
            walk(s["span_id"], depth + 1)

    walk(None, 0)
    return {"trace_id": trace_id, "summary": summary, "spans": ordered}


@app.get("/api/traces/{trace_id}/trajectory")
def trajectory(trace_id: str):
    spans = _load("WHERE trace_id = ?", (trace_id,))
    if not spans:
        raise HTTPException(404, "trace not found")
    for s in spans:
        s["attrs"] = json.loads(s["attrs"])
    case_id = next((s["case_id"] for s in spans if s["case_id"]), None)
    return {"trace_id": trace_id, "case_id": case_id, **build_trajectory(spans)}


@app.get("/api/config")
def config():
    return {"orchestrator_url": os.environ.get("ORCHESTRATOR_URL", "http://localhost:8100")}


@app.get("/api/metrics")
def metrics():
    spans = _load()
    traces_ = _summarize(spans)
    done = [t for t in traces_ if t["complete"]]
    stages: dict[str, int] = {}
    for t in traces_:
        stages[t["stage"]] = stages.get(t["stage"], 0) + 1

    comp: dict[str, dict] = {}
    for s in spans:
        n = s["name"]
        key = n if n.split(".")[0] in ("planner_agent", "agent", "validate", "validation_layer", "reporting_layer",
                                       "llm", "tool", "guardrail", "intake", "hil") else None
        if key is None or n == "case.orchestration":
            continue
        d = comp.setdefault(key, {"name": key, "kind": s["kind"], "durations": [], "errors": 0})
        d["durations"].append(_ms(s))
        d["errors"] += 1 if s["status"] == 2 else 0
    components = [{"name": d["name"], "kind": d["kind"], "count": len(d["durations"]),
                   "avg_ms": sum(d["durations"]) / len(d["durations"]), "p95_ms": _pct(d["durations"], 0.95),
                   "max_ms": max(d["durations"]), "errors": d["errors"]}
                  for d in comp.values()]

    verdicts: dict[str, int] = {}
    guard_fail = 0
    failed_checks: dict[str, int] = {}
    hil_stage: dict[str, int] = {}
    llm_by_model: dict[str, dict] = {}
    for s in spans:
        a = json.loads(s["attrs"])
        if s["name"].startswith("validate."):
            verdicts[a.get("roa.verdict", "?")] = verdicts.get(a.get("roa.verdict", "?"), 0) + 1
            for chk in json.loads(a.get("roa.checks_failed", "[]")):
                failed_checks[chk] = failed_checks.get(chk, 0) + 1
        if s["name"].startswith("guardrail.") and a.get("roa.guardrail.passed") is False:
            guard_fail += 1
        if s["name"] == "hil.wait":
            k = a.get("roa.hil.stage", "?")
            hil_stage[k] = hil_stage.get(k, 0) + 1
        if s["name"].startswith("llm."):
            m = a.get("gen_ai.request.model", "?")
            d = llm_by_model.setdefault(m, {"model": m, "calls": 0, "input_tokens": 0, "output_tokens": 0, "ms": 0.0})
            d["calls"] += 1
            d["input_tokens"] += int(a.get("gen_ai.usage.input_tokens", 0))
            d["output_tokens"] += int(a.get("gen_ai.usage.output_tokens", 0))
            d["ms"] += _ms(s)

    return {
        "traces": len(traces_), "completed_traces": len(done), "stages": stages,
        "avg_active_ms": sum(t["active_ms"] for t in done) / len(done) if done else 0,
        "avg_hil_wait_ms": sum(t["hil_wait_ms"] for t in done) / len(done) if done else 0,
        "hil_total": sum(t["hil_count"] for t in traces_), "hil_by_stage": hil_stage,
        "errors": sum(t["errors"] for t in traces_), "tokens": sum(t["tokens"] for t in traces_),
        "llm_calls": sum(t["llm_calls"] for t in traces_),
        "verdicts": verdicts, "failed_checks": failed_checks, "guardrail_failures": guard_fail,
        "components": sorted(components, key=lambda c: c["name"]),
        "llm_by_model": list(llm_by_model.values()),
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")
