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
MAX_PAPERS_PER_RUN = 40      # hard safety cap on total papers processed in one run,
                             # independent of depth - protects the LLM/API budget

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
LLM_MODEL_FALLBACK_CHAIN = [
    "openrouter/free",                      # auto-router: picks a live free model for you
    "meta-llama/llama-3.3-70b-instruct:free",  # long-running, stable, good general fallback
    "openai/gpt-oss-120b:free",             # second independent fallback if both above are down
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
