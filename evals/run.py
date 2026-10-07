"""Run the offline evals:  python -m evals.run [--cases a,b] [--repeats N] [--only golden|judge] [--otel] [--out FILE]

Golden cases run through the real graph and the real LLM gateway (in-process, isolated temp data/runtime dirs,
no servers needed). A scripted human policy answers every pause so runs are unattended. Exit code 1 if a release gate
in evals/thresholds.json is missed.
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
_ARGV = sys.argv[1:]

# Isolate state BEFORE roa is imported (settings are read at import time).
_TMP = Path(tempfile.mkdtemp(prefix="roa_evals_"))
os.environ["ROA_DATA_DIR"] = str(_TMP / "data")
os.environ["ROA_RUNTIME_DIR"] = str(_TMP / "runtime")
if "--otel" not in _ARGV:
    os.environ["ROA_OTEL_ENABLED"] = "false"

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

from evals.scoring import aggregate, gate, judge_metrics, score_case  # noqa: E402
from roa import state, telemetry  # noqa: E402
from roa.config import settings  # noqa: E402
from roa.graph import build_graph  # noqa: E402
from roa.harness import GuardedStore, load_harness  # noqa: E402
from roa.models import Case, CaseState  # noqa: E402
from roa.validation.validator import judge_evidence  # noqa: E402

DEFAULT_POLICY = {"input": "abort", "plan": "abort", "agent": "abort", "validation": "continue_with_note", "report": "APPROVED"}


def _description(spec: dict) -> str:
    if "description_repeat" in spec:
        r = spec["description_repeat"]
        return r["text"] * r["times"]
    return spec["description"]


async def _run_case(graph, bundle, spec: dict) -> CaseState:
    tid, sid = telemetry.new_case_ids()
    cid = f"EVAL-{spec['id']}-{tid[:6]}"
    state.create_case(CaseState(case=Case(case_id=cid, source="eval", description=_description(spec),
                                          property_name=spec.get("property_name")),
                                trace_id=tid, root_span_id=sid, started_ns=telemetry.now_ns(),
                                harness_hash=bundle.hash, harness_version=bundle.version))
    policy = {**DEFAULT_POLICY, **spec.get("hil_policy", {})}
    cfg = {"configurable": {"thread_id": cid}, "recursion_limit": 200}
    await graph.ainvoke({"case_id": cid}, config=cfg)
    for _ in range(12):  # bounded: a scripted human never loops forever
        p = state.get(cid).pending_hil
        if p is None:
            break
        decision = policy[p.stage]
        await graph.ainvoke(Command(resume={"decision": decision, "comment": "eval policy", "agent_ids": []}), config=cfg)
    return state.get(cid)


async def run_golden(graph, bundle, cases: list[dict], repeats: int) -> tuple[list[dict], dict]:
    results: list[dict] = []
    stability: dict[str, list] = {}
    for spec in cases:
        for i in range(repeats):
            t0 = time.monotonic()
            try:
                cs = await _run_case(graph, bundle, spec)
                r = score_case(spec, cs, bundle)
            except Exception as e:  # noqa: BLE001 - an eval crash is a failed case, recorded, not fatal to the run
                r = {"case_id": spec["id"], "critical": bool(spec.get("critical")), "plan_ok": False, "hil_ok": False, "verdict_ok": False, "final_ok": False,
                     "grounded": False, "draft_safe": False, "passed": False, "error": f"{type(e).__name__}: {e}",
                     "observed": {}, "expected": spec["expect"], "cost": {"llm_calls": 0, "tokens": 0, "llm_ms": 0}}
            r["run"], r["wall_s"] = i + 1, round(time.monotonic() - t0, 1)
            results.append(r)
            stability.setdefault(spec["id"], []).append(json.dumps(r["observed"].get("plan")))
            flag = "PASS" if r["passed"] else "FAIL"
            print(f"  [{flag}] {spec['id']} run {i + 1}  plan={r['observed'].get('plan')} hil={r['observed'].get('hil_path')} "
                  f"verdicts={r['observed'].get('verdicts')} final={r['observed'].get('final_stage')} ({r['wall_s']}s)"
                  + (f"  ERROR {r['error']}" if r.get("error") else ""), flush=True)
    stable = {k: len(set(v)) == 1 for k, v in stability.items()}
    return results, {"plan_stability": (sum(stable.values()) / len(stable)) if repeats > 1 and stable else None,
                     "unstable_cases": [k for k, v in stable.items() if not v]}


async def run_judge_calibration(bundle, rows: list[dict]) -> tuple[list[dict], dict]:
    tid, sid = telemetry.new_case_ids()
    cid = "EVAL-judge-calibration"
    state.create_case(CaseState(case=Case(case_id=cid, source="eval", description="judge calibration"), trace_id=tid,
                                root_span_id=sid, started_ns=telemetry.now_ns()))
    out = []
    for row in rows:
        try:
            j = await judge_evidence(bundle, cid, row["case"], bundle.agents[row["agent_id"]], row["evidence"])
            verdict, why = j.verdict, j.rationale
        except Exception as e:  # noqa: BLE001
            verdict, why = "ERROR", str(e)[:200]
        out.append({"id": row["id"], "label": row["label"], "verdict": verdict, "rationale": why})
        mark = "ok " if verdict == row["label"] else "BAD"
        print(f"  [{mark}] {row['id']}: label={row['label']} judge={verdict}", flush=True)
    return out, judge_metrics(out)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", help="comma-separated golden case ids")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--only", choices=["golden", "judge"])
    ap.add_argument("--otel", action="store_true", help="export traces to the dashboard")
    ap.add_argument("--out", help="write the JSON report here (default evals/results/<timestamp>.json)")
    args = ap.parse_args(_ARGV)

    telemetry.init_tracing()
    state.init_db()
    bundle = load_harness(settings.harness_dir, lock=False)
    store = GuardedStore(settings.runtime_dir, settings.harness_dir)
    thresholds = json.loads((HERE / "thresholds.json").read_text(encoding="utf-8"))
    report: dict = {"at": datetime.now(timezone.utc).isoformat(), "harness_version": bundle.version,
                    "harness_hash": bundle.hash, "models": bundle.manifest["models"]}
    golden = judge = None

    async with AsyncSqliteSaver.from_conn_string(str(_TMP / "checkpoints.sqlite")) as saver:
        graph = build_graph(saver, bundle, store)
        if args.only != "judge":
            spec = json.loads((HERE / "golden_cases.json").read_text(encoding="utf-8"))["cases"]
            if args.cases:
                want = set(args.cases.split(","))
                spec = [c for c in spec if c["id"] in want]
            print(f"Golden cases: {len(spec)} x {args.repeats} run(s), models {bundle.manifest['models']}")
            results, stab = await run_golden(graph, bundle, spec, args.repeats)
            golden = aggregate(results)
            report["golden"] = {"metrics": golden, "stability": stab, "results": results}
        if args.only != "golden":
            rows = json.loads((HERE / "judge_cases.json").read_text(encoding="utf-8"))["cases"]
            print(f"Judge calibration: {len(rows)} labeled examples")
            judged, judge = await run_judge_calibration(bundle, rows)
            report["judge"] = {"metrics": judge, "results": judged}

    failures = gate(golden, judge, thresholds)
    report["gate_failures"] = failures
    out = Path(args.out) if args.out else HERE / "results" / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n=== Summary ===")
    if golden:
        print("golden:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in golden.items()})
        if report["golden"]["stability"]["plan_stability"] is not None:
            print("plan stability across repeats:", report["golden"]["stability"])
    if judge:
        print("judge :", {k: round(v, 2) for k, v in judge.items()})
    print("GATES :", "all met" if not failures else "MISSED -> " + "; ".join(failures))
    print("report:", out)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
