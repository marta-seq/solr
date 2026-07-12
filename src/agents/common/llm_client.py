"""
llm_client.py
Thin wrapper around OpenRouter's free-tier models, with automatic fallback
across a chain of models (a free-tier model timing out or rate-limiting is
common - don't let one bad model kill the whole run).

Requires env var OPENROUTER_API_KEY. Get one free at https://openrouter.ai/keys

Usage:
    from common.llm_client import call_llm_json
    result, model_used = call_llm_json(system_prompt, user_prompt)
"""

import json
import os
import re
import time

import requests

from . import config


class LLMError(Exception):
    pass


def _extract_json(raw: str):
    """Models sometimes wrap JSON in prose or ```json fences - strip that off."""
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    # If there's leading/trailing prose, grab the outermost {...} or [...]
    if not (raw.startswith("{") or raw.startswith("[")):
        m = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if m:
            raw = m.group(1)
    return json.loads(raw)


def _call_one_model(model: str, system_prompt: str, user_prompt: str) -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY environment variable not set")

    resp = requests.post(
        config.OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        },
        timeout=config.LLM_TIMEOUT_S,
    )
    if resp.status_code != 200:
        raise LLMError(f"{model} returned HTTP {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise LLMError(f"{model} returned no choices: {data}")
    return choices[0]["message"]["content"]


def call_llm_json(system_prompt: str, user_prompt: str):
    """
    Tries each model in config.LLM_MODEL_FALLBACK_CHAIN in order, with a couple
    of retries per model, until one returns parseable JSON.

    Returns (parsed_json, model_name_used).
    Raises LLMError if every model in the chain fails.
    """
    last_error = None
    for model in config.LLM_MODEL_FALLBACK_CHAIN:
        for attempt in range(config.LLM_MAX_RETRIES_PER_MODEL):
            call_start = time.time()
            try:
                raw = _call_one_model(model, system_prompt, user_prompt)
                elapsed = time.time() - call_start
                parsed = _extract_json(raw)
                print(f"[llm_client] {model} responded in {elapsed:.2f}s "
                      f"({len(raw)} chars back)", flush=True)
                return parsed, model
            except (LLMError, json.JSONDecodeError, requests.RequestException) as e:
                elapsed = time.time() - call_start
                last_error = e
                print(f"[llm_client] {model} failed after {elapsed:.2f}s "
                      f"(attempt {attempt + 1}/{config.LLM_MAX_RETRIES_PER_MODEL}): {e}", flush=True)
                time.sleep(1.5 * (attempt + 1))
                continue
        print(f"[llm_client] {model} exhausted all retries, "
              f"falling back to next model in chain...", flush=True)
    raise LLMError(f"All models in fallback chain failed. Last error: {last_error}")
