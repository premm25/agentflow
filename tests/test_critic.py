import copy
import json
import shutil
from dataclasses import replace

from roa.config import settings
from roa.harness import load_harness
from roa.harness.loader import parse_front_matter
from roa.validation.critic import CHECK_NAMES, critique_harness
from roa.validation.validator import _deterministic
from tests.test_validation import _cs, _good, _task


def _with_policy(bundle, edit):
    policy = copy.deepcopy(bundle.validation)
    edit(policy)
    return replace(bundle, validation=policy)


def _with_evidence(bundle, agent_id="pricing_agent", **changes):
    spec = bundle.agents[agent_id]
    exp = {**spec.expected_evidence, **changes}
    return replace(bundle, agents={**bundle.agents, agent_id: replace(spec, expected_evidence=exp)})


def _codes(bundle, severity=None):
    return {f.code for f in critique_harness(bundle) if severity in (None, f.severity)}


def test_real_harness_has_no_error_findings(bundle):
    assert _codes(bundle, "error") == set()
    assert {f.code for f in critique_harness(bundle)} == {"dead_severity"}  # severity.judge is never read


def test_check_names_match_what_the_validator_emits(bundle):
    calls, ev = _good()
    emitted = [c.name for c in _deterministic(bundle, _cs(), _task(calls, ev))]
    assert emitted == list(CHECK_NAMES)


def test_critic_is_deterministic_and_does_not_mutate_the_bundle(bundle):
    before = copy.deepcopy((bundle.validation, {a: s.expected_evidence for a, s in bundle.agents.items()}))
    assert critique_harness(bundle) == critique_harness(bundle)
    assert before == (bundle.validation, {a: s.expected_evidence for a, s in bundle.agents.items()})


def test_missing_deterministic_check_is_detected(bundle):
    b = _with_policy(bundle, lambda p: p["deterministic_checks"].remove("numbers_grounded"))
    assert "missing_check" in _codes(b, "error")


def test_unknown_and_duplicate_deterministic_checks_are_detected(bundle):
    b = _with_policy(bundle, lambda p: p["deterministic_checks"].extend(["ghost_check", "status_done"]))
    assert {"unknown_check", "duplicate_check"} <= _codes(b, "error")


def test_missing_severity_is_detected(bundle):
    b = _with_policy(bundle, lambda p: p["severity"].pop("min_evidence"))
    assert "missing_severity" in _codes(b, "error")


def test_invalid_severity_value_is_detected(bundle):
    b = _with_policy(bundle, lambda p: p["severity"].update(status_done="warning"))
    assert "invalid_severity" in _codes(b, "error")


def test_unknown_and_dead_severity_entries_are_detected(bundle):
    b = _with_policy(bundle, lambda p: p["severity"].update(ghost_check="fail"))
    assert "unknown_severity" in _codes(b, "error")
    assert "dead_severity" in _codes(bundle, "warn")


def test_invalid_confidence_floors_are_detected(bundle):
    for key in ("confidence_floor", "evidence_confidence_floor"):
        for bad in (1.5, -0.1, "0.7", True):
            assert f"invalid_{key}" in _codes(_with_evidence(bundle, **{key: bad}), "error"), (key, bad)
    assert _codes(_with_evidence(bundle, confidence_floor=0, evidence_confidence_floor=1), "error") == set()


def test_invalid_min_evidence_is_detected(bundle):
    for bad in (0, -1, 2.5, "3", True):
        assert "invalid_min_evidence" in _codes(_with_evidence(bundle, min_evidence=bad), "error"), bad


def test_impossible_min_evidence_and_required_tools_are_detected(bundle):
    budget = bundle.agents["pricing_agent"].max_tool_calls
    assert "impossible_min_evidence" in _codes(_with_evidence(bundle, min_evidence=budget + 1), "error")
    small = replace(bundle.agents["pricing_agent"], max_tool_calls=2)  # pricing requires 3 tools
    b = replace(bundle, agents={**bundle.agents, "pricing_agent": small})
    assert "required_tools_exceed_budget" in _codes(b, "error")
    assert "min_evidence_exceeds_tools" in _codes(_with_evidence(bundle, min_evidence=5), "warn")


def test_invalid_expected_evidence_structure_is_detected(bundle):
    assert "unknown_evidence_key" in _codes(_with_evidence(bundle, confidence_flor=0.7), "error")
    assert "required_tools_type" in _codes(_with_evidence(bundle, required_tools="read_lrv"), "error")
    assert "required_tools_type" in _codes(_with_evidence(bundle, required_tools=[1]), "error")
    spec = replace(bundle.agents["lrv_agent"], expected_evidence=["not", "an", "object"])
    assert "expected_evidence_not_object" in _codes(replace(bundle, agents={**bundle.agents, "lrv_agent": spec}), "error")
    spec = replace(bundle.agents["lrv_agent"], expected_evidence={})
    assert "missing_evidence_key" in _codes(replace(bundle, agents={**bundle.agents, "lrv_agent": spec}), "warn")


def test_judge_configuration_problems_are_detected(bundle):
    b = _with_policy(bundle, lambda p: p["judge"].update(enabled="yes"))
    assert "judge_not_bool" in _codes(b, "error")
    b = _with_policy(bundle, lambda p: p["judge"].pop("fail_blocks"))
    assert "missing_judge_key" in _codes(b, "error")
    b = _with_policy(bundle, lambda p: p["judge"].update(threshold=0.5))
    assert "unknown_judge_key" in _codes(b, "error")
    b = _with_policy(bundle, lambda p: p.update(judge=[]))
    assert "judge_not_object" in _codes(b, "error")


def test_semantic_check_without_judge_is_flagged(bundle):
    b = _with_policy(bundle, lambda p: p["judge"].update(enabled=False))
    assert "semantic_check_without_judge" in _codes(b, "warn")
    assert "semantic_check_without_judge" not in _codes(bundle)


def test_allowed_unknowns_is_flagged_when_used_or_malformed(bundle):
    spec = replace(bundle.agents["lrv_agent"], allowed_unknowns=["days_since_update"])
    assert "allowed_unknowns_unused" in _codes(replace(bundle, agents={**bundle.agents, "lrv_agent": spec}), "warn")
    spec = replace(bundle.agents["lrv_agent"], allowed_unknowns="x")
    assert "allowed_unknowns_type" in _codes(replace(bundle, agents={**bundle.agents, "lrv_agent": spec}), "error")


def test_problems_are_aggregated_and_located(bundle):
    b = _with_policy(bundle, lambda p: (p["severity"].update(status_done="nope"), p["judge"].update(enabled=1)))
    b = _with_evidence(b, min_evidence=0, confidence_floor=2)
    found = critique_harness(b)
    assert {"invalid_severity", "judge_not_bool", "invalid_min_evidence", "invalid_confidence_floor"} <= {f.code for f in found}
    assert any(f.location == "agents/pricing_agent/AGENT.md:expected_evidence.min_evidence" for f in found)


def test_on_disk_agent_md_edit_is_reported_after_load(tmp_path):
    copy_root = tmp_path / "harness"
    shutil.copytree(settings.harness_dir, copy_root, copy_function=shutil.copyfile)  # copyfile: do not inherit the read-only lock
    md = copy_root / "agents" / "pricing_agent" / "AGENT.md"
    meta, body = parse_front_matter(md.read_text(encoding="utf-8"))
    meta["expected_evidence"]["evidence_confidence_floor"] = 1.5
    meta["expected_evidence"]["min_evidnce"] = 3
    md.write_text(f"---\n{json.dumps(meta, indent=2)}\n---\n{body}\n", encoding="utf-8")
    codes = _codes(load_harness(copy_root, lock=False), "error")
    assert {"invalid_evidence_confidence_floor", "unknown_evidence_key"} <= codes


# ---------------------------------------------------------------- agent-level types the runtime relies on
def _with_spec(bundle, agent_id="lrv_agent", **changes):
    return replace(bundle, agents={**bundle.agents, agent_id: replace(bundle.agents[agent_id], **changes)})


def test_invalid_semantic_check_type_is_detected(bundle):
    for bad in ("yes", 1, None, "false"):
        assert "semantic_check_type" in _codes(_with_spec(bundle, semantic_check=bad), "error"), bad
    assert "semantic_check_type" not in _codes(bundle)


def test_invalid_max_tool_calls_is_detected(bundle):
    """Regression: 'ten' silently disabled the feasibility check. The runtime does min(max_tool_calls, budget)."""
    for bad in ("ten", 0, -3, 2.5, True, None, [6]):
        assert "invalid_max_tool_calls" in _codes(_with_spec(bundle, max_tool_calls=bad), "error"), bad


def test_invalid_timeout_is_detected(bundle):
    for bad in (-5, 0, "60", None, True, float("nan"), [60]):
        assert "invalid_timeout" in _codes(_with_spec(bundle, timeout_s=bad), "error"), bad
    assert "invalid_timeout" not in _codes(_with_spec(bundle, timeout_s=0.5))  # any positive number is a valid timeout


def test_invalid_enabled_and_handles_types_are_detected(bundle):
    assert "enabled_type" in _codes(_with_spec(bundle, enabled="yes"), "error")
    for bad in ("pricing", ["ok", 5], None, [["nested"]]):
        assert "handles_type" in _codes(_with_spec(bundle, handles=bad), "error"), bad


def test_unknown_top_level_validation_keys_are_detected(bundle):
    """Regression: a misspelt or unread key in validation.json used to be silently ignored."""
    b = replace(bundle, validation={**bundle.validation, "qa": {"low_confidence_margin": 0.1}, "severty": {}})
    found = [f for f in critique_harness(b) if f.code == "unknown_policy_key"]
    assert {f.location for f in found} == {"validation.json:qa", "validation.json:severty"}
    assert all(f.severity == "error" for f in found)


def test_every_actual_top_level_key_is_known(bundle):
    from roa.validation.critic import POLICY_KEYS

    assert set(bundle.validation) == set(POLICY_KEYS)  # the real file uses exactly the keys the validator can read


def test_all_the_audit_probe_misconfigurations_are_found_together(bundle):
    b = replace(_with_spec(bundle, semantic_check="yes", max_tool_calls="ten", timeout_s=-5), validation={**bundle.validation, "typo_key": 1})
    assert {"semantic_check_type", "invalid_max_tool_calls", "invalid_timeout", "unknown_policy_key"} <= _codes(b, "error")


def test_the_config_critic_never_raises_on_hostile_shapes(bundle):
    for bad in (None, [], "x", 5, {"judge": []}, {"severity": [], "judge": 5, "deterministic_checks": {"a": 1}},
                {"deterministic_checks": [["a"]], "severity": {"status_done": ["x"]}, "judge": {"enabled": [], "fail_blocks": {}}}):
        assert critique_harness(replace(bundle, validation=bad)) == critique_harness(replace(bundle, validation=bad))
    hostile = _with_spec(bundle, expected_evidence={"required_tools": [["x"]], "min_evidence": [], "confidence_floor": {}, "junk": 1},
                         allowed_unknowns=[1], tools_allowed=["read_lrv"])
    assert critique_harness(hostile) == critique_harness(hostile)


# ---------------------------------------------------------------- scope guards
def test_the_critic_is_pure_no_io_llm_or_runtime_imports():
    """critic.py may import only the standard library and the harness loader's types: no validator, state, llm, store."""
    import ast
    from pathlib import Path

    from roa.validation import critic

    tree = ast.parse(Path(critic.__file__).read_text(encoding="utf-8"))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert modules == {"dataclasses", "typing", "roa.harness.loader"}, modules
    called = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert not called & {"eval", "exec", "compile", "open", "__import__", "print"}, called


def test_the_original_validation_api_is_intact():
    """Adding the critic must not change what roa.validation exports or how the runtime validator is reached."""
    from roa.validation import __all__ as exported, pending_failures, run_validation_layer, validate_agent  # noqa: F401

    assert sorted(exported) == ["pending_failures", "run_validation_layer", "validate_agent"]
