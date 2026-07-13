"""
llm_client.py
Tries OpenRouter's free-tier model chain first (config.LLM_MODEL_FALLBACK_CHAIN),
and only once ALL of those are exhausted, falls through to genuinely separate
providers (config.FALLBACK_PROVIDERS) - each with its own independent quota,
not just more OpenRouter model names. A provider whose API key isn't set in
.env is silently skipped, so this is safe to leave as-is before you've signed
up for any of the fallback providers.

All of these speak the same OpenAI-compatible chat-completions shape (model,
messages, temperature -> choices[0].message.content), so one generic call
function handles every provider - only the URL, key, and model name differ.

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


def _call_model(url: str, api_key: str, model: str, system_prompt: str, user_prompt: str) -> tuple:
    """Generic OpenAI-compatible chat-completions call - works for OpenRouter,
    Groq, Cerebras, Gemini's OpenAI-compatible endpoint, and DeepSeek direct
    alike, since they all share this request/response shape.

    Returns (content, actual_model_used). For auto-routing aliases like
    "openrouter/free", the request model and the ACTUAL model that served it
    are different - OpenRouter reports the real one back in the response
    body's "model" field, which matters for knowing which model to credit/
    distrust for a given result, not just the alias we asked for."""
    resp = requests.post(
        url,
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
    actual_model = data.get("model") or model
    return choices[0]["message"]["content"], actual_model


def _build_provider_chain():
    """OpenRouter first (its own model chain), then each configured fallback
    provider in order - skipping any whose API key isn't set."""
    chain = [{
        "name": "openrouter",
        "url": config.OPENROUTER_URL,
        "api_key_env": "OPENROUTER_API_KEY",
        "models": config.LLM_MODEL_FALLBACK_CHAIN,
    }]
    chain.extend(config.FALLBACK_PROVIDERS)
    return chain


def call_llm_json(system_prompt: str, user_prompt: str):
    """
    Tries OpenRouter's model chain first, then each configured fallback
    provider in turn, with a couple of retries per model, until one returns
    parseable JSON.

    Returns (parsed_json, model_name_used).
    Raises LLMError if every provider/model combination fails.
    """
    last_error = None

    for provider in _build_provider_chain():
        api_key = os.environ.get(provider["api_key_env"])
        if not api_key:
            print(f"[llm_client] {provider['name']}: no {provider['api_key_env']} "
                  f"set, skipping", flush=True)
            continue

        for model in provider["models"]:
            for attempt in range(config.LLM_MAX_RETRIES_PER_MODEL):
                call_start = time.time()
                try:
                    raw, actual_model = _call_model(provider["url"], api_key, model, system_prompt, user_prompt)
                    elapsed = time.time() - call_start
                    parsed = _extract_json(raw)
                    routed_note = f" (routed to {actual_model})" if actual_model != model else ""
                    print(f"[llm_client] {provider['name']}/{model} responded in "
                          f"{elapsed:.2f}s ({len(raw)} chars back){routed_note}", flush=True)
                    return parsed, actual_model
                except (LLMError, json.JSONDecodeError, requests.RequestException) as e:
                    elapsed = time.time() - call_start
                    last_error = e
                    print(f"[llm_client] {provider['name']}/{model} failed after "
                          f"{elapsed:.2f}s (attempt {attempt + 1}/"
                          f"{config.LLM_MAX_RETRIES_PER_MODEL}): {e}", flush=True)
                    time.sleep(1.5 * (attempt + 1))
                    continue
            print(f"[llm_client] {provider['name']}/{model} exhausted all retries, "
                  f"trying next model/provider...", flush=True)

    raise LLMError(f"All providers/models failed. Last error: {last_error}")
