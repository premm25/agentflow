"""Shared contracts. Agents, validators and the reporter only ever exchange these shapes."""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class TaskStatus(str, Enum):
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    SKIPPED = "SKIPPED"  # human chose to continue without this agent


class CaseStatus(str, Enum):
    RECEIVED = "RECEIVED"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    VALIDATING = "VALIDATING"
    REPORTING = "REPORTING"
    WAITING_FOR_HUMAN = "WAITING_FOR_HUMAN"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


class TimeHorizon(str, Enum):
    NEAR_TERM = "NEAR_TERM"
    LONG_TERM = "LONG_TERM"
    UNKNOWN = "UNKNOWN"


_PLACEHOLDER_VALUES = {"unknown", "not mentioned", "n/a", "na", "none", "not specified", ""}


class CaseUnderstanding(BaseModel):
    case_type_hint: Optional[str] = None
    time_horizon: TimeHorizon = TimeHorizon.UNKNOWN
    property_name: Optional[str] = None
    entities: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""

    @field_validator("property_name", mode="before")
    @classmethod
    def _drop_placeholder(cls, v):
        if isinstance(v, str) and v.strip().lower() in _PLACEHOLDER_VALUES:
            return None
        return v


class FlowSelection(BaseModel):
    """The explicit runtime decision: which registered flow this case follows, and why. Written by the planner step (source
    intent_match / keyword_match / multiple_topics from the Flow Registry, or planner / human when the registry could not
    decide), persisted in the case state, and the single source the API, telemetry and UI read the flow from."""

    status: Literal["SELECTED", "UNRESOLVED"] = "SELECTED"
    flow_id: Optional[str] = None
    name: str = ""
    version: str = ""
    description: str = ""
    intent: Optional[str] = None
    source: Literal["intent_match", "keyword_match", "multiple_topics", "planner", "human", "unresolved"] = "unresolved"
    reason: str = ""
    matched: dict[str, Any] = Field(default_factory=dict)
    participating_agents: list[str] = Field(default_factory=list)
    selected_at: datetime = Field(default_factory=utcnow)


class Plan(BaseModel):
    case_type: str
    agent_ids: list[str]
    reasoning: str = ""
    source: Literal["llm", "keyword_fallback", "human", "flow_default"] = "llm"


class ToolCallRecord(BaseModel):
    """Written by the harness (never by the agent) for every tool call."""

    call_id: str = Field(default_factory=lambda: new_id("CALL"))
    agent_id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] = Field(default_factory=dict)
    result_hash: str = ""
    duration_ms: int = 0
    error: Optional[str] = None
    at: datetime = Field(default_factory=utcnow)


class EvidenceSource(BaseModel):
    type: str = "TOOL"
    tool: str
    call_id: str
    result_hash: str


class Evidence(BaseModel):
    evidence_id: str = Field(default_factory=lambda: new_id("EVID"))
    claim: str
    value: dict[str, Any] = Field(default_factory=dict)
    source: EvidenceSource
    confidence: float = 0.8


class AgentTaskInput(BaseModel):
    case_id: str
    task_id: str
    case_text: str
    understanding: CaseUnderstanding
    attempt: int = 1


class AgentTaskResult(BaseModel):
    task_id: str
    agent_id: str
    status: TaskStatus
    evidence: list[Evidence] = Field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""
    reason: Optional[str] = None


class TaskRecord(BaseModel):
    task_id: str
    case_id: str
    agent_id: str
    attempt: int = 1
    status: TaskStatus = TaskStatus.RUNNING
    started_at: datetime = Field(default_factory=utcnow)
    ended_at: Optional[datetime] = None
    result: Optional[AgentTaskResult] = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)


class CheckResult(BaseModel):
    name: str
    passed: bool
    severity: Literal["fail", "warn"] = "fail"
    detail: str = ""


class JudgeResult(BaseModel):
    verdict: Literal["PASS", "FAIL", "UNSURE"]
    rationale: str = ""
    model: str = ""

    @field_validator("verdict", mode="before")
    @classmethod
    def _upper(cls, v):
        return v.strip().upper() if isinstance(v, str) else v


class Verdict(BaseModel):
    agent_id: str
    task_id: str
    attempt: int
    status: Literal["PASS", "FAIL", "WARN"]
    checks: list[CheckResult] = Field(default_factory=list)
    judge: Optional[JudgeResult] = None
    judge_unavailable: bool = False
    at: datetime = Field(default_factory=utcnow)


class GuardrailEvent(BaseModel):
    point: Literal["input", "plan", "agent", "output", "budget", "harness"]
    rule: str
    passed: bool
    detail: str = ""
    at: datetime = Field(default_factory=utcnow)


class HILRequest(BaseModel):
    hil_id: str
    type: Literal["failure_review", "final_approval"]
    stage: Literal["input", "plan", "agent", "validation", "report"]
    reason: str
    agent_id: Optional[str] = None
    failing_agents: list[str] = Field(default_factory=list)
    options: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    requested_at: datetime = Field(default_factory=utcnow)
    requested_at_ns: int = 0
    resuming: bool = False


class HILDecision(BaseModel):
    hil_id: str
    type: str
    stage: str
    decision: str
    comment: Optional[str] = None
    agent_id: Optional[str] = None
    agent_ids: list[str] = Field(default_factory=list)  # plan-stage human choice
    decided_at: datetime = Field(default_factory=utcnow)
    waited_ms: int = 0


class LLMCall(BaseModel):
    """One LLM call: which role asked (understanding / planner / validator_judge / reporter), never the prompt or key."""

    role: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    at: datetime = Field(default_factory=utcnow)


class LLMUsage(BaseModel):
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_ms: int = 0
    detail: list[LLMCall] = Field(default_factory=list)


class Case(BaseModel):
    case_id: str
    source: str = "demo_ui"
    description: str
    property_name: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)


class CaseState(BaseModel):
    case: Case
    stage: CaseStatus = CaseStatus.RECEIVED
    trace_id: str = ""
    root_span_id: str = ""
    started_ns: int = 0
    harness_hash: str = ""
    harness_version: str = ""
    understanding: Optional[CaseUnderstanding] = None
    flow_selection: Optional[FlowSelection] = None
    plan: Optional[Plan] = None
    tasks: list[TaskRecord] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)
    waived_agents: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    guardrail_events: list[GuardrailEvent] = Field(default_factory=list)
    hil_requests: list[HILRequest] = Field(default_factory=list)
    hil_decisions: list[HILDecision] = Field(default_factory=list)
    pending_hil: Optional[HILRequest] = None
    response_draft: Optional[str] = None
    revision_count: int = 0
    report_paths: dict[str, str] = Field(default_factory=dict)
    llm_usage: LLMUsage = Field(default_factory=LLMUsage)
    final_reason: Optional[str] = None
    history: list[dict[str, Any]] = Field(default_factory=list)
