"""OpenAI-compatible chat client (e.g. a LiteLLM gateway) with schema-constrained JSON output,
one repair re-prompt, per-case usage accounting, budgets, and an LLM span on every call.
"""

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
import jsonschema

from roa import state, telemetry
from roa.config import settings


class LLMError(Exception):
    """Transport/gateway failure: the LLM is unavailable."""


class LLMOutputError(LLMError):
    """The model answered but could not produce schema-valid JSON after the repair attempt."""


class LLMBudgetExceeded(LLMError):
    pass


@dataclass
class LLMMeta:
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


_clients: dict[int, httpx.AsyncClient] = {}


def _client() -> httpx.AsyncClient:
    """One keep-alive connection pool per event loop (no new TCP/TLS handshake per LLM call)."""
    import asyncio

    key = id(asyncio.get_running_loop())
    c = _clients.get(key)
    if c is None or c.is_closed:
        c = _clients[key] = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10, keepalive_expiry=60),
            timeout=httpx.Timeout(120.0, connect=5.0))
    return c


async def aclose() -> None:
    for c in list(_clients.values()):
        if not c.is_closed:
            try:
                await c.aclose()
            except RuntimeError:  # bound to a loop that is already gone
                pass
    _clients.clear()


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"}


async def _chat(messages: list[dict], model: str, temperature: float, timeout: float,
                response_format: dict | None = None, max_tokens: int = 4096,
                reasoning_effort: str | None = None) -> tuple[str, dict]:
    body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if response_format:
        body["response_format"] = response_format
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    url = f"{settings.llm_base_url}/chat/completions"
    try:
        client = _client()
        resp = await client.post(url, json=body, headers=_headers(), timeout=timeout)
        if resp.status_code == 400 and reasoning_effort and "reasoning_effort" in resp.text:
            body.pop("reasoning_effort")  # model does not take the parameter
            resp = await client.post(url, json=body, headers=_headers(), timeout=timeout)
        if resp.status_code == 400 and response_format and "response_format" in resp.text:
            body["response_format"] = {"type": "json_object"}
            resp = await client.post(url, json=body, headers=_headers(), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        raise LLMError(f"LLM gateway call failed: {type(e).__name__}: {e}") from e
    try:
        content = data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError) as e:
        raise LLMError(f"unexpected LLM response shape: {str(data)[:200]}") from e
    return content, data.get("usage") or {}


def _check_budget(case_id: str, max_calls: int) -> None:
    cs = state.get(case_id)
    if cs is not None and cs.llm_usage.calls >= max_calls:
        raise LLMBudgetExceeded(f"LLM call budget of {max_calls} per case exhausted")


def _account(case_id: str, model: str, usage: dict, latency_ms: int, span, role: str | None = None) -> LLMMeta:
    p, c = int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
    span.set_attribute("gen_ai.usage.input_tokens", p)
    span.set_attribute("gen_ai.usage.output_tokens", c)
    span.set_attribute("llm.latency_ms", latency_ms)
    state.add_llm_usage(case_id, p, c, latency_ms, role=role, model=model)
    return LLMMeta(model=model, prompt_tokens=p, completion_tokens=c, latency_ms=latency_ms)


async def call_structured(
    case_id: str,
    role: str,
    model: str,
    system: str,
    user: str,
    schema: dict,
    max_calls: int = 30,
    temperature: float = 0.1,
    timeout: float = 120.0,
    reasoning_effort: str | None = None,
    max_tokens: int = 4096,
) -> tuple[dict, LLMMeta]:
    """Returns (validated JSON dict, meta). Raises LLMOutputError / LLMError / LLMBudgetExceeded."""
    _check_budget(case_id, max_calls)
    name = schema.get("title", "Output")
    rf = {"type": "json_schema", "json_schema": {"name": name, "strict": False, "schema": schema}}
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    with telemetry.span(f"llm.{role}", "LLM", **{"roa.case_id": case_id, "gen_ai.system": "openai-compatible",
                                                 "gen_ai.request.model": model, "roa.llm.role": role}) as sp:
        t0 = time.monotonic()
        raw, usage = await _chat(messages, model, temperature, timeout, rf, max_tokens, reasoning_effort)
        parsed, err = _parse(raw, schema)
        total_usage = dict(usage)
        if parsed is None:
            sp.add_event("repair_attempt", {"error": err[:200]})
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That output did not match the required schema ({err}). "
                                            "Reply again with ONLY corrected JSON matching the schema."},
            ]
            raw, usage2 = await _chat(messages, model, temperature, timeout, rf, max_tokens, reasoning_effort)
            for k in ("prompt_tokens", "completion_tokens"):
                total_usage[k] = int(total_usage.get(k, 0)) + int(usage2.get(k, 0))
            parsed, err = _parse(raw, schema)
        meta = _account(case_id, model, total_usage, int((time.monotonic() - t0) * 1000), sp, role)
        if parsed is None:
            raise LLMOutputError(f"{name}: no schema-valid JSON after repair: {err}; raw={raw[:300]!r}")
        return parsed, meta


async def call_text(
    case_id: str, role: str, model: str, system: str, user: str,
    max_calls: int = 30, temperature: float = 0.3, timeout: float = 120.0,
    reasoning_effort: str | None = None, max_tokens: int = 4096,
) -> tuple[str, LLMMeta]:
    _check_budget(case_id, max_calls)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    with telemetry.span(f"llm.{role}", "LLM", **{"roa.case_id": case_id, "gen_ai.system": "openai-compatible",
                                                 "gen_ai.request.model": model, "roa.llm.role": role}) as sp:
        t0 = time.monotonic()
        raw, usage = await _chat(messages, model, temperature, timeout, None, max_tokens, reasoning_effort)
        meta = _account(case_id, model, usage, int((time.monotonic() - t0) * 1000), sp, role)
        return raw.strip(), meta


def _parse(raw: str, schema: dict) -> tuple[dict | None, str]:
    text = _FENCE.sub("", raw.strip())
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    try:
        jsonschema.validate(obj, schema)
    except jsonschema.ValidationError as e:
        return None, e.message
    return obj, ""
