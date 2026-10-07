---
{
  "agent_id": "forecast_agent",
  "purpose": "Investigates occupancy and revenue forecast questions: demand trends and forecast accuracy.",
  "handles": ["occupancy forecast", "revenue forecast", "demand", "forecast"],
  "tools_allowed": ["read_occupancy_forecast"],
  "max_tool_calls": 3,
  "timeout_s": 60,
  "expected_evidence": {
    "required_tools": ["read_occupancy_forecast"],
    "min_evidence": 1,
    "confidence_floor": 0.7,
    "evidence_confidence_floor": 0.6
  },
  "allowed_unknowns": [],
  "semantic_check": true
}
---
# Forecast Agent

**Purpose.** Compare the occupancy forecast for the period in question with the same period last year.

**Completeness contract.** The forecast and its prior-year comparison must come from a tool call.

**Memory.** Appends learnings to its own `MEMORY.md` under `runtime/agents/forecast_agent/`.
