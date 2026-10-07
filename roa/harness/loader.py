"""Loads the immutable harness (registries, flows, guardrails, schemas, prompts, specs).

The harness directory is hashed at load time. `verify()` re-hashes it and raises if anything
changed, so tampering is detected before a case runs and again before its report is issued.
Files are also flipped read-only at the OS level as a best-effort second layer.
"""

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from roa.harness.flows import FlowRegistry, FlowResolution


class HarnessError(Exception):
    """The harness definition itself is invalid or inconsistent."""


class HarnessTampered(HarnessError):
    """The harness on disk no longer matches the hash taken at startup."""


@dataclass
class AgentSpec:
    agent_id: str
    display_name: str
    version: str
    enabled: bool
    entrypoint: str
    purpose: str
    handles: list[str]
    tools_allowed: list[str]
    max_tool_calls: int
    timeout_s: int
    expected_evidence: dict[str, Any]
    allowed_unknowns: list[str]
    semantic_check: bool
    body_md: str
    spec_path: str

    def card_text(self) -> str:
        return f"- {self.agent_id}: {self.purpose} (handles: {', '.join(self.handles)})"


def parse_front_matter(text: str, path: str = "") -> tuple[dict[str, Any], str]:
    """AGENT.md = a JSON block between two '---' lines, then a markdown body."""
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        raise HarnessError(f"{path}: missing '---' front matter")
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        raise HarnessError(f"{path}: unterminated front matter") from None
    try:
        meta = json.loads("\n".join(lines[1:end]))
    except json.JSONDecodeError as e:
        raise HarnessError(f"{path}: front matter is not valid JSON: {e}") from e
    return meta, "\n".join(lines[end + 1 :]).strip()


def hash_tree(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode())
            h.update(b"\0")
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HarnessError(f"cannot load {path}: {e}") from e


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:
        raise HarnessError(f"cannot load {path}: {e}") from e


@dataclass
class HarnessBundle:
    root: Path
    hash: str
    manifest: dict[str, Any]
    agents: dict[str, AgentSpec]
    tools: dict[str, Any]
    flows: dict[str, Any]
    canonical_order: list[str]
    step_policy: dict[str, Any]
    guardrails: dict[str, Any]
    planner_schema: dict[str, Any]
    planner_prompt: str
    understanding_schema: dict[str, Any]
    understanding_prompt: str
    fallback: dict[str, Any]
    validation: dict[str, Any]
    judge_prompt: str
    reporting: dict[str, Any]
    report_prompt: str
    _extra: dict[str, Any] = field(default_factory=dict)

    @property
    def version(self) -> str:
        return self.manifest.get("harness_version", "?")

    def model_for(self, role: str) -> str:
        return self.manifest["models"][role]

    def llm_opts(self, role: str) -> dict[str, Any]:
        """Per-role call options from the manifest: reasoning effort and an output-token cap."""
        cfg = self.manifest["llm"]
        return {"reasoning_effort": cfg.get("reasoning_effort", {}).get(role),
                "max_tokens": cfg.get("max_tokens", {}).get(role, 2048)}

    def enabled_agents(self) -> list[AgentSpec]:
        return [a for a in self.agents.values() if a.enabled]

    def agent_menu(self) -> str:
        return "\n".join(a.card_text() for a in self.enabled_agents())

    def flow_menu(self) -> str:
        return "\n".join(
            f"- {name}: {f['description']} (allowed agents: {', '.join(f['allowed_agents'])})"
            for name, f in self.flows.items()
        )

    def keyword_hits(self, case_text: str) -> dict[str, list[str]]:
        """Deterministic evidence: for each enabled agent, which of its declared `handles` keywords appear in the case text."""
        text = case_text.lower()
        hits = {a.agent_id: [k for k in a.handles if k.lower() in text] for a in self.enabled_agents()}
        return {a: ks for a, ks in hits.items() if ks}

    def keyword_match(self, case_text: str) -> list[str]:
        """Deterministic fallback: which agents' declared `handles` appear in the case text."""
        return list(self.keyword_hits(case_text))

    def flow_registry(self) -> FlowRegistry:
        """The typed view over harness/flows/flows.json (the one flow registry)."""
        return FlowRegistry(self.flows, self.canonical_order)

    def resolve_flow(self, intent: str | None, case_text: str) -> FlowResolution:
        """Intent + keyword evidence -> the registered flow to plan in, with the reason. Pure, no LLM."""
        return self.flow_registry().resolve(intent, self.keyword_hits(case_text))

    def order(self, agent_ids: list[str]) -> list[str]:
        rank = {a: i for i, a in enumerate(self.canonical_order)}
        return sorted(agent_ids, key=lambda a: rank.get(a, len(rank)))

    def verify(self) -> None:
        current = hash_tree(self.root)
        if current != self.hash:
            raise HarnessTampered(
                f"harness changed on disk (expected {self.hash[:12]}, found {current[:12]})"
            )


def _lock_readonly(root: Path) -> None:
    for p in root.rglob("*"):
        if p.is_file():
            try:
                os.chmod(p, stat.S_IREAD)
            except OSError:
                pass


def load_harness(root: Path, lock: bool = True) -> HarnessBundle:
    root = Path(root).resolve()
    manifest = _json(root / "manifest.json")
    comp = manifest["components"]

    reg = _json(root / comp["agent_registry"])
    agents: dict[str, AgentSpec] = {}
    for entry in reg["agents"]:
        spec_path = root / entry["spec"]
        meta, body = parse_front_matter(_text(spec_path), str(spec_path))
        if meta.get("agent_id") != entry["agent_id"]:
            raise HarnessError(f"{spec_path}: agent_id does not match registry entry {entry['agent_id']}")
        agents[entry["agent_id"]] = AgentSpec(
            agent_id=entry["agent_id"],
            display_name=entry.get("display_name", entry["agent_id"]),
            version=entry.get("version", "0"),
            enabled=entry.get("enabled", True),
            entrypoint=entry["entrypoint"],
            purpose=meta["purpose"],
            handles=meta.get("handles", []),
            tools_allowed=meta.get("tools_allowed", []),
            max_tool_calls=meta.get("max_tool_calls", 10),
            timeout_s=meta.get("timeout_s", 60),
            expected_evidence=meta.get("expected_evidence", {}),
            allowed_unknowns=meta.get("allowed_unknowns", []),
            semantic_check=meta.get("semantic_check", True),
            body_md=body,
            spec_path=entry["spec"],
        )

    tools = _json(root / comp["tool_registry"])["tools"]
    for a in agents.values():
        unknown = [t for t in a.tools_allowed if t not in tools]
        if unknown:
            raise HarnessError(f"{a.agent_id}: tools_allowed lists tools missing from the tool registry: {unknown}")
        stray = [t for t in a.expected_evidence.get("required_tools", []) if t not in a.tools_allowed]
        if stray:
            raise HarnessError(f"{a.agent_id}: required_tools not in tools_allowed: {stray}")

    flows_doc = _json(root / comp["flow_registry"])
    claimed: dict[str, str] = {}
    for name, flow in flows_doc["flows"].items():
        for a in flow["allowed_agents"] + flow.get("default_agents", []):
            if a not in agents:
                raise HarnessError(f"flow '{name}' references unknown agent '{a}'")
        if "enabled" in flow and not isinstance(flow["enabled"], bool):
            raise HarnessError(f"flow '{name}': enabled must be true or false, got {flow['enabled']!r}")
        intents = flow.get("intents", [])
        if not isinstance(intents, list) or not all(isinstance(i, str) and i.strip() for i in intents):
            raise HarnessError(f"flow '{name}': intents must be a list of non-empty strings")
        for i in intents:  # an intent selects exactly one flow, so selection is deterministic
            if i.strip().lower() in claimed:
                raise HarnessError(f"intent '{i}' is registered to both flow '{claimed[i.strip().lower()]}' and flow '{name}'")
            claimed[i.strip().lower()] = name
    for a in flows_doc["canonical_agent_order"]:
        if a not in agents:
            raise HarnessError(f"canonical_agent_order references unknown agent '{a}'")

    planner_schema = _json(root / comp["planner_schema"])
    declared = set(planner_schema["properties"]["case_type"]["enum"])
    if declared != set(flows_doc["flows"]):
        raise HarnessError(
            f"planner schema case_type enum {sorted(declared)} does not match flows {sorted(flows_doc['flows'])}"
        )

    bundle = HarnessBundle(
        root=root,
        hash=hash_tree(root),
        manifest=manifest,
        agents=agents,
        tools=tools,
        flows=flows_doc["flows"],
        canonical_order=flows_doc["canonical_agent_order"],
        step_policy=flows_doc["step_policy"],
        guardrails=_json(root / comp["guardrails"]),
        planner_schema=planner_schema,
        planner_prompt=_text(root / comp["planner_prompt"]),
        understanding_schema=_json(root / comp["understanding_schema"]),
        understanding_prompt=_text(root / comp["understanding_prompt"]),
        fallback=_json(root / comp["fallback"]),
        validation=_json(root / comp["validation_policy"]),
        judge_prompt=_text(root / comp["judge_prompt"]),
        reporting=_json(root / comp["reporting_policy"]),
        report_prompt=_text(root / comp["report_prompt"]),
    )
    if lock:
        _lock_readonly(root)
        # chmod does not change contents, so the hash taken above stays valid.
    return bundle
