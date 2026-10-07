"""Live case-state store (sqlite). Written as a side effect the instant something changes, so the
API and UI reflect reality independent of LangGraph checkpoint timing. LangGraph state itself only
carries routing fields; every business fact lives here.
"""

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Callable

from roa.config import settings
from roa.models import (
    Case,
    CaseState,
    CaseStatus,
    GuardrailEvent,
    HILDecision,
    FlowSelection,
    HILRequest,
    LLMCall,
    TaskRecord,
    Verdict,
)

_write_lock = threading.RLock()


@contextmanager
def _conn():
    conn = sqlite3.connect(str(settings.db_path), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS case_states (
                case_id TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )


def create_case(state: CaseState) -> CaseState:
    with _write_lock, _conn() as conn:
        conn.execute(
            "INSERT INTO case_states (case_id, state_json, created_at) VALUES (?, ?, ?)",
            (state.case.case_id, state.model_dump_json(), state.case.created_at.isoformat()),
        )
    return state


def get(case_id: str) -> CaseState | None:
    with _conn() as conn:
        row = conn.execute("SELECT state_json FROM case_states WHERE case_id = ?", (case_id,)).fetchone()
    return CaseState.model_validate_json(row[0]) if row else None


def list_cases() -> list[CaseState]:
    with _conn() as conn:
        rows = conn.execute("SELECT state_json FROM case_states ORDER BY created_at DESC").fetchall()
    return [CaseState.model_validate_json(r[0]) for r in rows]


def mutate(case_id: str, fn: Callable[[CaseState], None]) -> CaseState:
    with _write_lock:
        state = get(case_id)
        if state is None:
            raise KeyError(f"Unknown case_id: {case_id}")
        fn(state)
        with _conn() as conn:
            conn.execute(
                "UPDATE case_states SET state_json = ? WHERE case_id = ?",
                (state.model_dump_json(), case_id),
            )
        return state


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_stage(case_id: str, stage: CaseStatus, reason: str | None = None) -> None:
    def fn(s: CaseState):
        s.stage = stage
        s.history.append({"stage": stage.value, "at": _now(), **({"reason": reason} if reason else {})})
        if reason and stage in (CaseStatus.FAILED, CaseStatus.ABORTED, CaseStatus.COMPLETED):
            s.final_reason = reason

    mutate(case_id, fn)


def upsert_task(case_id: str, task: TaskRecord) -> None:
    def fn(s: CaseState):
        for i, t in enumerate(s.tasks):
            if t.task_id == task.task_id:
                s.tasks[i] = task
                return
        s.tasks.append(task)

    mutate(case_id, fn)


def add_guardrail_events(case_id: str, events: list[GuardrailEvent]) -> None:
    if events:
        mutate(case_id, lambda s: s.guardrail_events.extend(events))


def add_verdict(case_id: str, verdict: Verdict) -> None:
    mutate(case_id, lambda s: s.verdicts.append(verdict))


def add_llm_usage(case_id: str, prompt: int, completion: int, latency_ms: int, role: str | None = None,
                  model: str | None = None) -> None:
    def fn(s: CaseState):
        s.llm_usage.calls += 1
        s.llm_usage.prompt_tokens += prompt
        s.llm_usage.completion_tokens += completion
        s.llm_usage.total_latency_ms += latency_ms
        if role:  # the per-call record the timeline shows; totals above are unchanged
            s.llm_usage.detail.append(LLMCall(role=role, model=model or "", prompt_tokens=prompt, completion_tokens=completion,
                                              latency_ms=latency_ms))

    mutate(case_id, fn)


def set_flow_selection(case_id: str, selection: FlowSelection) -> None:
    mutate(case_id, lambda s: setattr(s, "flow_selection", selection))


def record_event(case_id: str, event: str, **fields) -> None:
    """A lifecycle milestone the stage transitions do not capture (intent resolved, flow resolved, plan ready...). Lives in the
    same `history` list as the stage transitions, so the case state stays the one canonical record of what happened."""
    mutate(case_id, lambda s: s.history.append({"event": event, "at": _now(), **fields}))


def latest_task(cs: CaseState, agent_id: str) -> TaskRecord | None:
    tasks = [t for t in cs.tasks if t.agent_id == agent_id]
    return tasks[-1] if tasks else None


def latest_verdict(cs: CaseState, agent_id: str) -> Verdict | None:
    vs = [v for v in cs.verdicts if v.agent_id == agent_id]
    return vs[-1] if vs else None


# ---- human-in-the-loop -----------------------------------------------------------------
def ensure_pending_hil(case_id: str, build: Callable[[str], HILRequest]) -> HILRequest:
    """Idempotent: LangGraph re-runs a node from the top on resume, so the same request must be
    returned until it has been resolved. hil_id is derived from the number of resolved requests."""
    box: dict[str, HILRequest] = {}

    def fn(s: CaseState):
        if s.pending_hil is not None:
            box["req"] = s.pending_hil
            return
        hil_id = f"{s.case.case_id}-HIL-{len(s.hil_decisions) + 1}"
        req = build(hil_id)
        s.pending_hil = req
        s.hil_requests.append(req)
        s.stage = CaseStatus.WAITING_FOR_HUMAN
        s.history.append({"stage": CaseStatus.WAITING_FOR_HUMAN.value, "at": _now(), "hil_id": hil_id})
        box["req"] = req

    mutate(case_id, fn)
    return box["req"]


def begin_resume(case_id: str, hil_id: str) -> HILRequest | None:
    """Claim the pending request for a resume. Returns None if nothing pending / already claimed."""
    box: dict[str, HILRequest | None] = {"req": None}

    def fn(s: CaseState):
        p = s.pending_hil
        if p is not None and p.hil_id == hil_id and not p.resuming:
            p.resuming = True
            for r in s.hil_requests:
                if r.hil_id == hil_id:
                    r.resuming = True
            box["req"] = p

    mutate(case_id, fn)
    return box["req"]


def resolve_hil(case_id: str, decision: HILDecision) -> None:
    def fn(s: CaseState):
        s.hil_decisions.append(decision)
        s.pending_hil = None

    mutate(case_id, fn)


def recover_after_restart() -> dict[str, list[str]]:
    """The process died: a resume that had been claimed but not finished would otherwise 409 forever, and a
    case that was mid-run (not waiting on a human) cannot continue. Be honest about both."""
    out: dict[str, list[str]] = {"resume_released": [], "interrupted": []}
    for cs in list_cases():
        cid = cs.case.case_id
        if cs.stage in (CaseStatus.COMPLETED, CaseStatus.FAILED, CaseStatus.ABORTED):
            continue
        if cs.pending_hil is not None and cs.pending_hil.resuming:
            def release(s: CaseState):
                s.pending_hil.resuming = False
                for r in s.hil_requests:
                    if r.hil_id == s.pending_hil.hil_id:
                        r.resuming = False

            mutate(cid, release)
            out["resume_released"].append(cid)
        elif cs.pending_hil is None:
            set_stage(cid, CaseStatus.FAILED, "interrupted by a server restart while running; resubmit the case")
            out["interrupted"].append(cid)
    return {k: v for k, v in out.items() if v}
