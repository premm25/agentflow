---
{
  "agent_id": "lrv_agent",
  "purpose": "Investigates Last Room Value (LRV) issues, forecast locks, and inventory control decisions.",
  "handles": ["last room value", "lrv", "forecast lock", "inventory control"],
  "tools_allowed": ["read_lrv", "read_lrv_update_log"],
  "max_tool_calls": 4,
  "timeout_s": 60,
  "expected_evidence": {
    "required_tools": ["read_lrv", "read_lrv_update_log"],
    "min_evidence": 2,
    "confidence_floor": 0.7,
    "evidence_confidence_floor": 0.6
  },
  "allowed_unknowns": [],
  "semantic_check": true
}
---
# LRV Agent

**Purpose.** Report the current Last Room Value and when it was last updated by the optimization.

**Completeness contract.** The LRV value and its update history must both come from tool calls.

**Memory.** Appends learnings to its own `MEMORY.md` under `runtime/agents/lrv_agent/`.
