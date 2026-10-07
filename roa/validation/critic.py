"""Harness Config Critic: a deterministic consistency check of the validation configuration.

It reads only an already-loaded HarnessBundle (harness/validation/validation.json plus each agent's AGENT.md contract)
and reports problems that load_harness() does not catch. It is not an agent-result validator and never runs at case
time: no I/O, no LLM, no state. Findings are aggregated so one call reports every problem.

Deliberately NOT re-checked here because load_harness() already raises HarnessError for them: tools_allowed vs the tool
registry, required_tools vs tools_allowed, flow / canonical-order agent references, planner enum vs flows.
"""

from dataclasses import dataclass
from typing import Any, Literal

from roa.harness.loader import HarnessBundle

# The checks roa.validation.validator._deterministic emits, in order. Duplicated on purpose so the critic does not depend
# on the runtime validator; tests/test_critic.py fails if the two drift apart.
CHECK_NAMES = ("status_done", "required_tools_called", "min_evidence", "evidence_provenance", "numbers_grounded",
               "confidence_floor", "allowed_tools_only")
SEVERITIES = ("fail", "warn")
# Severity entries that are not a deterministic check: the judge's effect comes from judge.fail_blocks, so this is unread.
UNUSED_SEVERITY_KEYS = ("judge",)
JUDGE_KEYS = ("enabled", "fail_blocks")
POLICY_KEYS = ("validation_version", "deterministic_checks", "severity", "judge")
EVIDENCE_KEYS = ("required_tools", "min_evidence", "confidence_floor", "evidence_confidence_floor")


@dataclass(frozen=True)
class Finding:
    severity: Literal["error", "warn"]
    code: str
    location: str
    message: str


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_str_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) for x in v)


def critique_harness(bundle: HarnessBundle) -> list[Finding]:
    findings: list[Finding] = []

    def add(severity: str, code: str, location: str, message: str):
        findings.append(Finding(severity, code, location, message))  # type: ignore[arg-type]

    _critique_policy(bundle.validation, add)
    judge = bundle.validation.get("judge") if isinstance(bundle.validation, dict) else None
    judge_enabled = judge.get("enabled") if isinstance(judge, dict) else None
    for spec in bundle.agents.values():
        _critique_agent(spec, judge_enabled, add)
    return findings


def _critique_policy(v: Any, add) -> None:
    loc = "validation.json"
    if not isinstance(v, dict):
        add("error", "policy_not_object", loc, "validation policy must be a JSON object")
        return
    for key in v:
        if key not in POLICY_KEYS:
            add("error", "unknown_policy_key", f"{loc}:{key}", f"'{key}' is never read by the validator (typo?)")

    # --- deterministic_checks: must list exactly the checks the validator runs
    checks = v.get("deterministic_checks")
    if not _is_str_list(checks):
        add("error", "checks_not_list", f"{loc}:deterministic_checks", "must be a list of check names")
        checks = []
    for name in sorted({c for c in checks if checks.count(c) > 1}):
        add("error", "duplicate_check", f"{loc}:deterministic_checks", f"check '{name}' is listed more than once")
    for name in CHECK_NAMES:
        if name not in checks:
            add("error", "missing_check", f"{loc}:deterministic_checks", f"validator runs '{name}' but it is not listed")
    for name in dict.fromkeys(checks):
        if name not in CHECK_NAMES:
            add("error", "unknown_check", f"{loc}:deterministic_checks", f"'{name}' is not a check the validator runs")

    # --- severity: one valid entry per check (a missing one silently defaults to "fail" at runtime)
    sev = v.get("severity")
    if not isinstance(sev, dict):
        add("error", "severity_not_object", f"{loc}:severity", "must be an object mapping check name to 'fail' or 'warn'")
    else:
        for name in CHECK_NAMES:
            if name not in sev:
                add("error", "missing_severity", f"{loc}:severity", f"no severity for check '{name}'")
        for name, value in sev.items():
            here = f"{loc}:severity.{name}"
            if name in UNUSED_SEVERITY_KEYS:
                add("warn", "dead_severity", here, f"'{name}' is not a deterministic check and is never read")
            elif name not in CHECK_NAMES:
                add("error", "unknown_severity", here, f"'{name}' is not a check the validator runs")
            if value not in SEVERITIES:
                add("error", "invalid_severity", here, f"must be 'fail' or 'warn', got {value!r}")

    # --- judge: the validator indexes judge["enabled"] and judge["fail_blocks"] directly
    judge = v.get("judge")
    if not isinstance(judge, dict):
        add("error", "judge_not_object", f"{loc}:judge", "must be an object with 'enabled' and 'fail_blocks'")
        return
    for key in JUDGE_KEYS:
        if key not in judge:
            add("error", "missing_judge_key", f"{loc}:judge.{key}", f"required key '{key}' is missing")
        elif not isinstance(judge[key], bool):
            add("error", "judge_not_bool", f"{loc}:judge.{key}", f"must be true or false, got {judge[key]!r}")
    for key in judge:
        if key not in JUDGE_KEYS:
            add("error", "unknown_judge_key", f"{loc}:judge.{key}", f"'{key}' is not read by the validator")


def _critique_agent(spec, judge_enabled: Any, add) -> None:
    loc = spec.spec_path
    # Types the runtime relies on: enabled / semantic_check are used as flags, handles is iterated with .lower() and
    # ', '.join, max_tool_calls and timeout_s go into min() against the guardrail budgets (runner.py, context.py).
    if not isinstance(spec.enabled, bool):
        add("error", "enabled_type", f"{loc}:enabled", f"must be true or false, got {spec.enabled!r}")
    if not isinstance(spec.semantic_check, bool):
        add("error", "semantic_check_type", f"{loc}:semantic_check", f"must be true or false, got {spec.semantic_check!r}")
    if not _is_str_list(spec.handles):
        add("error", "handles_type", f"{loc}:handles", "must be a list of strings")
    if not isinstance(spec.max_tool_calls, int) or isinstance(spec.max_tool_calls, bool) or spec.max_tool_calls < 1:
        add("error", "invalid_max_tool_calls", f"{loc}:max_tool_calls", f"must be an integer >= 1, got {spec.max_tool_calls!r}")
    if not _is_number(spec.timeout_s) or not spec.timeout_s > 0:
        add("error", "invalid_timeout", f"{loc}:timeout_s", f"must be a number > 0, got {spec.timeout_s!r}")
    if judge_enabled is False and spec.semantic_check is True:
        add("warn", "semantic_check_without_judge", f"{loc}:semantic_check",
            "semantic_check is true but the global judge is disabled, so it is never applied")

    unknowns = spec.allowed_unknowns
    if not _is_str_list(unknowns):
        add("error", "allowed_unknowns_type", f"{loc}:allowed_unknowns", "must be a list of strings")
    elif unknowns:
        add("warn", "allowed_unknowns_unused", f"{loc}:allowed_unknowns",
            f"{unknowns} is never read by validation, so it has no effect")

    exp = spec.expected_evidence
    here = f"{loc}:expected_evidence"
    if not isinstance(exp, dict):
        add("error", "expected_evidence_not_object", here, "must be an object")
        return
    for key in exp:
        if key not in EVIDENCE_KEYS:
            add("error", "unknown_evidence_key", f"{here}.{key}", f"'{key}' is ignored by validation (typo?)")
    for key in EVIDENCE_KEYS:
        if key not in exp:
            add("warn", "missing_evidence_key", f"{here}.{key}", f"'{key}' is unset; validation falls back to its default")

    required = exp.get("required_tools", [])
    if not _is_str_list(required):
        add("error", "required_tools_type", f"{here}.required_tools", "must be a list of tool names")
        required = []
    elif len(set(required)) != len(required):
        add("warn", "duplicate_required_tools", f"{here}.required_tools", "lists a tool more than once")

    for key in ("confidence_floor", "evidence_confidence_floor"):
        if key in exp and not (_is_number(exp[key]) and 0 <= exp[key] <= 1):
            add("error", f"invalid_{key}", f"{here}.{key}", f"must be a number in [0, 1], got {exp[key]!r}")

    # Every evidence item must trace to a recorded tool call, and each agent produces one claim per call, so the tool
    # budget bounds how much evidence (and how many required tools) can ever be obtained.
    budget = spec.max_tool_calls if isinstance(spec.max_tool_calls, int) and not isinstance(spec.max_tool_calls, bool) else None
    if budget is not None and len(required) > budget:
        add("error", "required_tools_exceed_budget", f"{here}.required_tools",
            f"{len(required)} required tools cannot all be called within max_tool_calls={budget}")
    if "min_evidence" in exp:
        n = exp["min_evidence"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            add("error", "invalid_min_evidence", f"{here}.min_evidence", f"must be an integer >= 1, got {n!r}")
        elif budget is not None and n > budget:
            add("error", "impossible_min_evidence", f"{here}.min_evidence",
                f"needs {n} evidence items but max_tool_calls={budget} allows at most {budget}")
        elif spec.tools_allowed and n > len(spec.tools_allowed):
            add("warn", "min_evidence_exceeds_tools", f"{here}.min_evidence",
                f"needs {n} evidence items but only {len(spec.tools_allowed)} tools are allowed")
