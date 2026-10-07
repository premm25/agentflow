---
{
  "agent_id": "pricing_agent",
  "purpose": "Investigates pricing decisions, rate configuration, rate history, and BAR/rate-shopping questions.",
  "handles": ["pricing", "rate", "rate configuration", "floor and ceiling", "offset", "bar", "rate shopping", "competitor"],
  "tools_allowed": ["read_pricing_snapshot", "read_occupancy", "read_competitor_rates", "read_rate_configuration"],
  "max_tool_calls": 6,
  "timeout_s": 60,
  "expected_evidence": {
    "required_tools": ["read_pricing_snapshot", "read_occupancy", "read_competitor_rates"],
    "min_evidence": 3,
    "confidence_floor": 0.7,
    "evidence_confidence_floor": 0.6
  },
  "allowed_unknowns": [],
  "semantic_check": true
}
---
# Pricing Agent

**Purpose.** Explain a rate decision: the current rate, the demand context (occupancy), and where competitors sit.

**Completeness contract (checked by the validation layer, not by this agent).**
- Current rate, occupancy, and competitor rate range must all come from tool calls.
- For long-horizon (LONG_TERM) cases the agent only reads rate configuration; that is inconclusive by design and the validation layer will flag it for a human, who may then add the Forecast Agent by retrying with a different plan.

**Memory.** This agent may append what it learned to its own `MEMORY.md` under `runtime/agents/pricing_agent/`. It can never modify this file.
