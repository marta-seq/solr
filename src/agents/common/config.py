"""
config.py
Central, tunable configuration for the Phase 2 agent pipeline.
Nothing here should require touching agent code to change behaviour.
"""

from pathlib import Path
import os
import re

# ── Paths (mirrors 01_parse_excel.py conventions) ────────────────────────────
ROOT           = Path(__file__).resolve().parents[3]
CURATED_DIR    = ROOT / "data" / "data_curated"
PROCESSED_DIR  = ROOT / "data" / "processed"
CACHE_DIR      = ROOT / "data" / "paper_cache"        # cached full text / abstracts
STAGING_DIR    = ROOT / "data" / "agent_review"        # candidate rows for human review

# Load ROOT/.env explicitly (not relying on cwd-based auto-discovery) so
# SOLR_EMAIL/OPENROUTER_API_KEY get picked up regardless of which directory
# you run the script from, or whether you're in a plain shell or an IDE run
# config. Degrades gracefully (prints a warning, doesn't crash) if
# python-dotenv isn't installed - `pip install python-dotenv --break-system-packages`.
_env_file = ROOT / ".env"
try:
    from dotenv import load_dotenv
    if _env_file.exists():
        load_dotenv(_env_file)
    else:
        print(f"[config] No .env file found at {_env_file} - "
              f"relying on real shell environment variables only.")
except ImportError:
    print("[config] python-dotenv not installed - .env file will NOT be loaded, "
          "only real shell environment variables will be seen. "
          "Run: pip install python-dotenv --break-system-packages")

for d in (CACHE_DIR, STAGING_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Recursion control (compared-methods track only) ──────────────────────────
# depth 0 = papers already tagged as computational method in the master DB
# depth 1 = new method papers discovered because a depth-0 paper compared against them
MAX_HOPS = 1                # how many hops beyond the seed set to recurse into
MAX_PAPERS_PER_RUN = 40     # hard safety cap on total papers processed in one run,
                            # independent of depth - protects the LLM/API budget

# If True (recommended given free-tier rate limits), agents skip the LLM
# call ENTIRELY when the target section (methods / data_availability)
# wasn't cleanly isolated - i.e. get_agent_text's source label must be
# exactly "section:<name>", not "full_text_fallback" (heading not found,
# sent unfocused text from the start of the paper - may miss late-paper
# comparisons entirely) or "abstract_only" (weakest signal). Set False to
# also try the LLM on those lower-confidence inputs - costs more queries
# per genuinely-useful result.
REQUIRE_ISOLATED_SECTION = True

# ── Category tags that mark a paper as "computational method" ───────────────
# used to decide whether a paper (seed or newly-discovered) enters the
# compared-methods queue at all
METHOD_CATEGORY_KEYWORDS = ["computational analysis - method", "technical methods"]

# ── LLM (free-tier via OpenRouter) ───────────────────────────────────────────
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Tried in order; first one that returns a valid response wins.
# Verified against openrouter.ai/models (Price: Free) on 2026-07-13 - the
# free roster rotates OFTEN, re-verify before a big run if this has aged.
# NOTE: DeepSeek currently has ZERO free models on OpenRouter (confirmed
# 2026-07-13) despite many guides/tutorials still referencing deepseek:free
# slugs - don't add those back without checking openrouter.ai/models first.
# IMPORTANT: OpenRouter's free-tier rate limit (50-1000 req/day) is ACCOUNT-
# WIDE across every :free model, not per-model - adding more entries here
# gives resilience against any single model rotating out or misbehaving,
# it does NOT raise your total daily quota. For more total quota, you need
# a genuinely separate provider (Groq/Gemini/Cerebras/Mistral/DeepSeek-direct),
# each with its own account and key.
LLM_MODEL_FALLBACK_CHAIN = [
    "openrouter/free",                          # auto-router: picks a live free model for you
    # "meta-llama/llama-3.3-70b-instruct:free" REMOVED 2026-08-07 - confirmed
    # dead (HTTP 404 "unavailable for free, use meta-llama/llama-3.3-70b-instruct
    # instead") across multiple live runs on gaia, every single time, with zero
    # exceptions. Keeping a permanently-dead entry in this list wasn't adding
    # resilience, just guaranteeing 2 wasted attempts (plus the 1.5s/3s
    # retry sleeps) on every single paper before falling through. Re-add only
    # if you check openrouter.ai/models and see the :free slug actually listed
    # again.
    "nvidia/nemotron-3-super-120b-a12b:free",   # verified free 2026-07-12, 1M context
    "qwen/qwen3-coder:free",                    # verified free 2026-07-13 (replaced gpt-oss-120b:free,
                                                 # which went paid-only - confirmed via your own error log)
    "google/gemma-4-31b-it:free",               # verified free 2026-07-12 - can hit upstream 429
                                                 # rate-limits during busy periods (seen live 2026-08-07);
                                                 # that's expected free-tier behaviour, not a bug - the
                                                 # chain just moves on to the next model/provider.
]
# ── Fallback providers (tried only after ALL OpenRouter models above are
# exhausted) - genuinely separate services with their own independent quotas,
# not just more OpenRouter model names. Each needs its own API key env var;
# any provider whose key isn't set in .env is simply skipped, so this is
# safe to leave as-is even before you've signed up for any of them.
# All of these speak the same OpenAI-compatible chat-completions shape.
#
# Order matters: tried top to bottom, first success wins. ollama_local comes
# first - genuinely zero cost and zero rate limit once a model is pulled, no
# quota to exhaust at all (added 2026-09-17: gaia already has an Ollama
# server running as a systemd service, 2x idle RTX 2080 Ti, confirmed via
# `systemctl status ollama`). Groq comes next - its free tier is explicitly
# documented as such (see its own comment below). Cerebras REMOVED 2026-09-20
# - both of its free-catalog models (gpt-oss-120b, zai-glm-4.7) were
# confirmed 100% broken (402 payment-required, 404 archived respectively) and
# never recovered, so it was permanent dead weight in the chain. Gemini is
# deliberately LAST among the keyed providers - reordered 2026-09-17 after a
# real run silently spent a chunk of calls on Gemini (whose free-vs-billed
# status for this specific key was unconfirmed at the time, since confirmed
# free) when OpenRouter's daily cap hit, even though Groq (confirmed free)
# was sitting right there unused, later in the list.
#
# ollama_local only actually resolves when running ON gaia (or through an
# SSH tunnel to it) - localhost:11434 isn't reachable from anywhere else.
# From a machine without that tunnel, every attempt just fails fast
# ("connection refused", no hang) and falls through to the next provider -
# same harmless-overhead pattern as OpenRouter's 429s when its quota's
# exhausted, not a real slowdown.
FALLBACK_PROVIDERS = [
    {
        "name": "ollama_local",
        "url": "http://localhost:11434/v1/chat/completions",
        # Ollama's OpenAI-compatible endpoint ignores the actual key content -
        # this just needs to be a non-empty string so call_llm_json's truthy
        # check doesn't skip the provider. Set OLLAMA_API_KEY=ollama (or any
        # placeholder) in gaia's .env - it is NOT a real secret.
        "api_key_env": "OLLAMA_API_KEY",
        # Pull with `ollama pull qwen2.5:14b` first - fits comfortably on a
        # single RTX 2080 Ti (11GB VRAM, ~9GB model at Ollama's default
        # Q4_K_M quantization). qwen2.5:32b is the stronger-quality option if
        # 14b's classification accuracy turns out too weak in practice -
        # splits across both idle 2080 Tis (~19GB), still fits.
        "models": ["qwen2.5:14b"],
    },
    {
        "name": "groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_env": "GROQ_API_KEY",
        "models": ["llama-3.3-70b-versatile"],
    },
    {
        "name": "gemini",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "api_key_env": "GEMINI_API_KEY",
        "models": ["gemini-3.5-flash"],  # gemini-2.5-flash deprecated for new users as of ~July 2026
                                          # (confirmed by your own 404 error) - 3.5-flash is the
                                          # current GA replacement per Google's own deprecation page
                                          # NOTE: unlike Groq/Cerebras above, whether THIS specific key
                                          # is on a free or billed Google tier isn't confirmed - kept
                                          # last in the chain until that's checked.
    },
    {
        "name": "zhipu",
        # Z.ai (formerly Zhipu AI) - verified 2026-07-13: GLM-4.7-Flash and
        # GLM-4.5-Flash are genuinely free (not trial-limited) to all
        # registered users, OpenAI-compatible endpoint. Sign up at z.ai or
        # bigmodel.cn (same company - z.ai is the newer international
        # branding) - UNVERIFIED which domain/base URL works best from
        # outside China, check whichever your account dashboard shows you.
        "url": "https://api.z.ai/api/paas/v4/chat/completions",
        "api_key_env": "ZHIPU_API_KEY",
        "models": ["glm-4.7-flash", "glm-4.5-flash"],
    },
    {
        "name": "deepseek_direct",
        "url": "https://api.deepseek.com/chat/completions",
        "api_key_env": "DEEPSEEK_API_KEY",
        "models": ["deepseek-v4-flash"],
    },
]

LLM_TIMEOUT_S = 120
LLM_MAX_RETRIES_PER_MODEL = 2

# ── Paper fetching ────────────────────────────────────────────────────────────
# Reads the env var you already have set up in .venv for Unpaywall/PubMed.
# If your var is named differently, just change the key below.
CONTACT_EMAIL = os.environ.get("SOLR_EMAIL", "your@email.com")
FETCH_TIMEOUT_S = 20

# ── Section text sent to the LLM ─────────────────────────────────────────────
# Raised 2026-09-20 (was 12000, ~3000 tokens) - every provider actually in
# FALLBACK_PROVIDERS/OpenRouter's chain (Groq's Llama-3.3-70B, Gemini,
# Zhipu's GLM, DeepSeek, even locally-hosted qwen2.5) supports well beyond
# this in real context window, so the old cap was truncating real Methods
# sections on any paper with a long baseline-comparison writeup long before
# hitting any actual model limit - a self-imposed bottleneck, not a real one.
# Free-tier cost here is latency/rate-limit, not per-token billing, so there's
# real room to stop truncating this aggressively.
MAX_SECTION_CHARS = 60000  # roughly ~15000 tokens

# ── Review status enum (written back to the master DB after human merge) ────
# Redesigned 2026-09-18, per Marta's ask to stop conflating "how did this row
# enter the DB" with "has it been reviewed / how many automated stages have
# touched it". Two independent axes now:
#   - REVIEW_STATUS (this section): "manual" (a human has reviewed/confirmed
#     this row - regardless of how it originally got there), "needs_review"
#     (an agent extracted/filled fields with LOW confidence - kept as its
#     own distinct signal, not folded into the auto-N levels below, because
#     it specifically means "retry/look at this again", not just "was
#     touched once"), or "auto-N" (auto_level_status(N) below) meaning
#     "automatically processed by N distinct stages so far" - e.g. N=1 for
#     the literature-search scanner's own classification, N=2 if it's later
#     also enriched by the citation-chasing desk, etc. Always construct/
#     parse the "auto-N" string via the two helpers below, never hand-write
#     it, so the format can't drift.
#   - ADDITION_METHOD (further below): how the row FIRST entered the DB -
#     set once at creation, never changed again. Independent of the above.
REVIEW_STATUS_MANUAL = "manual"
REVIEW_STATUS_NEEDS_REVIEW = "needs_review"
# Staging-time marker ONLY - never write this literally into the master DB.
# An agent writes this when it doesn't know (and shouldn't need to know)
# what level this row is currently at - merge_candidates.py resolves it into
# the real "auto-N" at merge time, based on the row's current state. A
# create_entry action for a brand-new row (always level 1 - e.g. the
# literature-search scanner, which only ever creates new rows) can skip this
# marker entirely and just call auto_level_status(1) directly, since there's
# no ambiguity to resolve.
REVIEW_STATUS_AUTO = "auto"

_AUTO_LEVEL_RE = re.compile(r"^auto-(\d+)$")


def auto_level_status(level: int) -> str:
    return f"auto-{level}"


def parse_auto_level(status) -> int:
    """Returns the level N if `status` matches "auto-N" (case-insensitive),
    else None (covers "manual", "needs_review", blank, or anything else)."""
    m = _AUTO_LEVEL_RE.match(str(status).strip().lower())
    return int(m.group(1)) if m else None


# ── Addition method (how a row first entered the DB) ─────────────────────────
# New 2026-09-18. Set once at row creation, never touched again - independent
# of REVIEW_STATUS above. The literature-search scanner previously (mis)used
# REVIEW_STATUS="scraped" for this; that value is retired in favor of
# ADDITION_METHOD_SCRAPED + REVIEW_STATUS=auto_level_status(1).
ADDITION_METHOD_MANUAL = "manual"
ADDITION_METHOD_SCRAPED = "scraped"
ADDITION_METHOD_CITATION_CHASE = "citation_chase"
