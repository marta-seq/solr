"""
config.py
Central, tunable configuration for the Phase 2 agent pipeline.
Nothing here should require touching agent code to change behaviour.
"""

from pathlib import Path
import os

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
    "meta-llama/llama-3.3-70b-instruct:free",   # long-running, stable, good general fallback
    "nvidia/nemotron-3-super-120b-a12b:free",   # verified free 2026-07-12, 1M context
    "qwen/qwen3-coder:free",                    # verified free 2026-07-13 (replaced gpt-oss-120b:free,
                                                 # which went paid-only - confirmed via your own error log)
    "google/gemma-4-31b-it:free",               # verified free 2026-07-12
]
# ── Fallback providers (tried only after ALL OpenRouter models above are
# exhausted) - genuinely separate services with their own independent quotas,
# not just more OpenRouter model names. Each needs its own API key env var;
# any provider whose key isn't set in .env is simply skipped, so this is
# safe to leave as-is even before you've signed up for any of them.
# All of these speak the same OpenAI-compatible chat-completions shape.
FALLBACK_PROVIDERS = [
    {
        "name": "gemini",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "api_key_env": "GEMINI_API_KEY",
        "models": ["gemini-3.5-flash"],  # gemini-2.5-flash deprecated for new users as of ~July 2026
                                          # (confirmed by your own 404 error) - 3.5-flash is the
                                          # current GA replacement per Google's own deprecation page
    },
    {
        "name": "groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_env": "GROQ_API_KEY",
        "models": ["llama-3.3-70b-versatile"],
    },
    {
        "name": "cerebras",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "api_key_env": "CEREBRAS_API_KEY",
        # Cerebras's free self-serve catalog narrowed hard by mid-2026 to just
        # these two - llama-3.3-70b (what this used to say) moved behind the
        # paid Dedicated Endpoints tier. gpt-oss-120b is the "production"-
        # labeled one, zai-glm-4.7 is preview/evaluation - tried in that order.
        "models": ["gpt-oss-120b", "zai-glm-4.7"],
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
# Free-tier models vary a lot in usable context before quality degrades, and
# every extra character costs against the 20 req/min pacing too - cap what
# any single agent call gets, regardless of how long the real section is.
MAX_SECTION_CHARS = 12000  # roughly ~3000 tokens

# ── Review status enum (written back to the master DB after human merge) ────
REVIEW_STATUS_MANUAL = "manual"
REVIEW_STATUS_AUTO = "auto"
REVIEW_STATUS_NEEDS_REVIEW = "needs_review"
