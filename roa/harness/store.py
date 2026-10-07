"""The one and only write path for agents, validators and stages.

Rules, enforced in code (not by convention):
  * Nothing outside runtime/ is writable, so harness/ is immutable to every principal.
  * A principal may only write inside the subtree it owns.
  * Logs (*.jsonl, *.log) are append-only. Other files are write-once, except the
    explicitly rewritable ones (an agent's MEMORY.md and the reporter's report files).
  * Every refused attempt is appended to runtime/audit.log.jsonl.

Principals: "orchestrator", "planner", "validator", "reporter", "agent:<agent_id>".
"""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


class ImmutableViolation(PermissionError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class GuardedStore:
    def __init__(self, runtime_root: Path, harness_root: Path):
        self.runtime_root = Path(runtime_root).resolve()
        self.harness_root = Path(harness_root).resolve()
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._audit_path = self.runtime_root / "audit.log.jsonl"

    # ---- policy -----------------------------------------------------------------
    @staticmethod
    def _owner(parts: tuple[str, ...]) -> str | None:
        """Which principal owns this runtime-relative path (None = nobody may write)."""
        if len(parts) >= 3 and parts[0] == "cases":
            area = parts[2]
            if area == "orchestrator":
                return "orchestrator"
            if area == "planner":
                return "planner"
            if area == "validation":
                return "validator"
            if area == "report":
                return "reporter"
            if area == "agents" and len(parts) >= 4:
                return f"agent:{parts[3]}"
            if len(parts) == 3 and parts[2] == "case.json":
                return "orchestrator"
        if len(parts) >= 3 and parts[0] == "agents":
            return f"agent:{parts[1]}"
        return None

    @staticmethod
    def _is_log(name: str) -> bool:
        return name.endswith(".jsonl") or name.endswith(".log")

    @staticmethod
    def _rewritable(parts: tuple[str, ...]) -> bool:
        return parts[-1] == "MEMORY.md" or (len(parts) >= 3 and parts[0] == "cases" and parts[2] == "report")

    def _resolve(self, principal: str, rel: str, op: str) -> tuple[Path, tuple[str, ...]]:
        posix = PurePosixPath(rel.replace("\\", "/"))
        target = (self.runtime_root / posix).resolve()
        try:
            parts = target.relative_to(self.runtime_root).parts
        except ValueError:
            zone = "harness" if self._within(target, self.harness_root) else "outside runtime"
            self._deny(principal, rel, op, f"path resolves into {zone}; only runtime/ is writable")
        owner = self._owner(parts)
        if owner is None:
            self._deny(principal, rel, op, "no principal owns this path")
        if owner != principal:
            self._deny(principal, rel, op, f"path is owned by '{owner}'")
        return target, parts

    @staticmethod
    def _within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _deny(self, principal: str, rel: str, op: str, why: str):
        entry = {"at": _now(), "principal": principal, "path": rel, "op": op, "reason": why}
        with self._lock, open(self._audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        raise ImmutableViolation(f"{principal} may not {op} '{rel}': {why}")

    # ---- writes -----------------------------------------------------------------
    def append(self, principal: str, rel: str, data: Any) -> Path:
        target, _ = self._resolve(principal, rel, "append")
        line = data if isinstance(data, str) else json.dumps(data, default=str)
        if not line.endswith("\n"):
            line += "\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, open(target, "a", encoding="utf-8") as f:
            f.write(line)
        return target

    def log(self, principal: str, rel: str, event: str, **fields: Any) -> None:
        self.append(principal, rel, {"at": _now(), "event": event, **fields})

    def write(self, principal: str, rel: str, text: str) -> Path:
        target, parts = self._resolve(principal, rel, "write")
        if self._is_log(parts[-1]):
            self._deny(principal, rel, "write", "logs are append-only")
        if target.exists() and not self._rewritable(parts):
            self._deny(principal, rel, "write", "file already exists and is write-once")
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            target.write_text(text, encoding="utf-8")
        return target

    def write_json(self, principal: str, rel: str, obj: Any) -> Path:
        return self.write(principal, rel, json.dumps(obj, indent=2, default=str))

    # ---- reads (unrestricted inside runtime/) -----------------------------------
    def read(self, rel: str, default: str = "") -> str:
        target = (self.runtime_root / PurePosixPath(rel)).resolve()
        if not self._within(target, self.runtime_root) or not target.exists():
            return default
        return target.read_text(encoding="utf-8")

    def read_jsonl(self, rel: str) -> list[dict]:
        out = []
        for line in self.read(rel).splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def path_of(self, rel: str) -> Path:
        return self.runtime_root / PurePosixPath(rel)
