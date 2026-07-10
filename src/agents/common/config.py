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
# (OpenRouter's free-tier slugs change occasionally - verify at
# https://openrouter.ai/models?max_price=0 before a big run)
LLM_MODEL_FALLBACK_CHAIN = [
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat:free",
    "meta-llama/llama-3.3-70b-instruct:free",
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
