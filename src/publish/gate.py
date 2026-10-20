"""
gate.py - deterministic sanity gate run BEFORE the weekly publish (cron / solr-bot).

Compares a freshly built candidate `docs/data/` against the baseline that is currently
published (the bot clone's `docs/data/`) and decides whether the candidate may be pushed.
No LLM involved on purpose: every check is reproducible and cannot be talked past.

Usage:
    python src/publish/gate.py                       # candidate = ./docs/data, baseline = ~/solr-publish/docs/data
    python src/publish/gate.py --candidate DIR --baseline DIR [--report out.json]

Exit code 0 = pass (warnings allowed), 1 = FAIL (do not push, send the failure notice).
A missing baseline (first ever run) only runs the structural checks.

Hard checks (FAIL):
  1. all four JSON files exist and parse; stats.json has the freshness fields
  2. no duplicate ids in methods.json / datasets.json; no empty ids
  3. row counts did not drop more than MAX_DROP_FRAC vs the baseline (methods, datasets)
  4. no more than MAX_NEW_ROWS new rows per run (papers+datasets each) - runaway-run / volume guard
  5. no human-reviewed row ("manual") of the baseline disappeared or lost its manual status
  6. dangling links did not increase by more than MAX_NEW_DANGLING vs the baseline
  7. file sizes did not shrink more than MAX_SHRINK_FRAC
Soft checks (WARN, still passes): new rows with empty title, share of new rows lacking a
pub_date, embeddings.json covering far fewer papers than methods.json.
"""
import argparse
import json
import sys
from pathlib import Path

FILES = ("methods.json", "datasets.json", "stats.json", "embeddings.json")
STATS_REQUIRED = ("total_papers", "total_datasets", "last_updated", "last_human_review",
                  "unreviewed_papers", "unreviewed_datasets")

# Tunables. MAX_NEW_ROWS must be raised deliberately when new modalities (spatial transcriptomics etc.) go live.
MAX_DROP_FRAC     = 0.02   # a weekly run may lose at most 2% of rows (dedup/deletions are done by hand, not by the cron)
MAX_NEW_ROWS      = 400    # per file per run
MAX_NEW_DANGLING  = 25     # new dataset->paper links pointing at a non-existent paper id
MAX_SHRINK_FRAC   = 0.10   # a file may shrink by at most 10% in bytes
RELAXED_DROP_FRAC = 0.25   # used instead of both of the above when publishing with --allow-drop
REVIEWED = ("manual", "auto_confirmed")


def _load(d: Path, name: str, fails: list):
    f = d / name
    if not f.exists():
        fails.append(f"{name}: missing in {d}")
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001 - any parse failure is a hard fail
        fails.append(f"{name}: invalid JSON ({e})")
        return None


def _ids(rows, name, fails):
    ids = [r.get("id", "") for r in rows]
    if any(not i for i in ids):
        fails.append(f"{name}: {sum(1 for i in ids if not i)} rows with an empty id")
    dup = sorted({i for i in ids if i and ids.count(i) > 1}) if len(ids) < 5000 else []
    if dup:
        fails.append(f"{name}: duplicate ids {dup[:10]}{' ...' if len(dup) > 10 else ''}")
    return set(ids)


def _dangling(datasets, method_ids):
    return {d["id"] for d in datasets if d.get("paper_entry_id") and d["paper_entry_id"] not in method_ids}


def run(candidate: Path, baseline: Path | None, relaxed: bool = False):
    """relaxed=True (manual `--allow-drop`, after a big deliberate cleanup): allow up to 25% fewer rows / 25% smaller files."""
    max_drop = RELAXED_DROP_FRAC if relaxed else MAX_DROP_FRAC
    max_shrink = RELAXED_DROP_FRAC if relaxed else MAX_SHRINK_FRAC
    fails, warns, info = [], [], []
    methods = _load(candidate, "methods.json", fails)
    datasets = _load(candidate, "datasets.json", fails)
    stats = _load(candidate, "stats.json", fails)
    emb = _load(candidate, "embeddings.json", fails)
    if fails:
        return fails, warns, info
    if not isinstance(methods, list) or not isinstance(datasets, list) or not methods or not datasets:
        fails.append("methods.json / datasets.json are not non-empty lists")
        return fails, warns, info
    for k in STATS_REQUIRED:
        if k not in stats:
            fails.append(f"stats.json: missing field '{k}'")

    m_ids = _ids(methods, "methods.json", fails)
    d_ids = _ids(datasets, "datasets.json", fails)
    info.append(f"candidate: {len(methods)} papers, {len(datasets)} datasets")

    if isinstance(emb, dict) and emb and len(emb) < 0.8 * len([m for m in methods if not m.get("is_placeholder")]):
        warns.append(f"embeddings.json covers {len(emb)} papers vs {len(methods)} in methods.json")

    if baseline is None or not (baseline / "methods.json").exists():
        info.append("no baseline found - structural checks only")
        return fails, warns, info

    bf = []
    b_methods = _load(baseline, "methods.json", bf)
    b_datasets = _load(baseline, "datasets.json", bf)
    if bf:
        warns.append("baseline unreadable, skipping comparison checks: " + "; ".join(bf))
        return fails, warns, info
    b_m_ids = {r["id"] for r in b_methods}
    b_d_ids = {r["id"] for r in b_datasets}
    info.append(f"baseline:  {len(b_methods)} papers, {len(b_datasets)} datasets")

    for name, cand, base in (("methods", methods, b_methods), ("datasets", datasets, b_datasets)):
        if len(cand) < len(base) * (1 - max_drop):
            fails.append(f"{name}: row count dropped {len(base)} -> {len(cand)} (> {max_drop:.0%})")
        if len(cand) - len(base) > MAX_NEW_ROWS:
            fails.append(f"{name}: {len(cand) - len(base)} new rows (> cap {MAX_NEW_ROWS})")

    # human-reviewed rows must survive with their status
    cand_status = {r["id"]: r.get("review_status", "") for r in methods + datasets}
    lost = [r["id"] for r in b_methods + b_datasets
            if r.get("review_status") in REVIEWED and cand_status.get(r["id"]) not in REVIEWED]
    if lost:
        fails.append(f"{len(lost)} human-reviewed rows disappeared or lost their reviewed status, e.g. {lost[:8]}")

    new_dang = _dangling(datasets, m_ids) - _dangling(b_datasets, b_m_ids)
    if len(new_dang) > MAX_NEW_DANGLING:
        fails.append(f"{len(new_dang)} new dataset rows link to a non-existent paper id, e.g. {sorted(new_dang)[:8]}")

    for name in FILES:
        c, b = candidate / name, baseline / name
        if c.exists() and b.exists() and c.stat().st_size < b.stat().st_size * (1 - max_shrink):
            fails.append(f"{name}: shrank {b.stat().st_size} -> {c.stat().st_size} bytes (> {max_shrink:.0%})")

    new_m = [r for r in methods if r["id"] not in b_m_ids]
    new_d = [r for r in datasets if r["id"] not in b_d_ids]
    info.append(f"new rows: {len(new_m)} papers, {len(new_d)} datasets; "
                f"removed: {len(b_m_ids - m_ids)} papers, {len(b_d_ids - d_ids)} datasets")
    if new_m:
        no_title = [r["id"] for r in new_m if not r.get("title") and not r.get("is_placeholder")]
        if no_title:
            warns.append(f"{len(no_title)} new papers without a title, e.g. {no_title[:5]}")
        no_date = sum(1 for r in new_m if not r.get("pub_date") and not r.get("is_placeholder"))
        if no_date > 0.5 * len(new_m):
            warns.append(f"{no_date}/{len(new_m)} new papers have no pub_date (metadata fetch failing?)")
    return fails, warns, info


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    ap.add_argument("--candidate", type=Path, default=root / "docs" / "data")
    ap.add_argument("--baseline", type=Path, default=Path.home() / "solr-publish" / "docs" / "data")
    ap.add_argument("--report", type=Path)
    a = ap.parse_args()
    fails, warns, info = run(a.candidate, a.baseline if a.baseline.exists() else None)
    for l in info:
        print("INFO ", l)
    for l in warns:
        print("WARN ", l)
    for l in fails:
        print("FAIL ", l)
    print("GATE:", "FAIL" if fails else "PASS")
    if a.report:
        a.report.write_text(json.dumps({"pass": not fails, "fails": fails, "warns": warns, "info": info}, indent=2))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
