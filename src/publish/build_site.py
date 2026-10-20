"""
build_site.py - build the website data from "everything we currently know" and publish it (weekly cron / manual).

This is the ONLY producer of the published docs/data. It never touches the master, the autoreview files, data/processed or
docs/data on this machine: everything happens in data/site_build/.

  1. base file  = the newest *autoreview* file if it is newer than the master (Marta is mid-review: her already-reviewed
                  rows go live, unreviewed ones stay flagged), else the master
  2. merge      = apply the ACCUMULATED data/agent_review/staging.xlsx onto a copy of the base (staging is NOT archived;
                  it keeps accumulating until Marta runs the real merge), result -> data/site_build/datasets_curated_snapshot.xlsx
  3. build      = 01 -> 02 -> 03 -> 04 on that copy, outputs under data/site_build/{processed,docs_data} (env overrides)
  4. publish    = gate + src/publish/publish.py against the bot clone (dry run unless --push)

Usage (conda python):
    python -m src.publish.build_site                    # build + gate + DRY-RUN publish
    python -m src.publish.build_site --push             # build + gate + REAL publish   ("publish now", e.g. right after promoting)
    python -m src.publish.build_site --push --allow-drop   # after a big deliberate cleanup (gate tolerates up to -25% rows)
    options: --skip-embeddings (skip step 04), --skip-publish (stop after the build), --no-lock-check (caller already holds the orchestrate lock)
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import publish as pub  # noqa: E402

CURATED_DIR = ROOT / "data" / "data_curated"
BUILD_DIR = ROOT / "data" / "site_build"
LOCK_FILE = ROOT / "data" / "agent_review" / "runs" / "orchestrate.lock"
PRE = ROOT / "src" / "preprocessing"
SNAPSHOT_XLSX = BUILD_DIR / "datasets_curated_snapshot.xlsx"


def die(msg):
    print("BUILD FAILED:", msg)
    sys.exit(1)


def lock_is_held() -> bool:
    if not LOCK_FILE.exists():
        return False
    try:
        os.kill(int(LOCK_FILE.read_text().split()[0]), 0)
        return True
    except (ValueError, IndexError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def choose_base() -> tuple:
    masters = [f for f in CURATED_DIR.glob("datasets_curated_*.xlsx")
               if "autoreview" not in f.name.lower() and not f.name.startswith("~$")]
    if len(masters) != 1:
        die(f"expected exactly one master in {CURATED_DIR}, found {[m.name for m in masters]}")
    master = masters[0]
    newer = sorted((f for f in CURATED_DIR.glob("*autoreview*.xlsx")
                    if not f.name.startswith("~$") and f.stat().st_mtime > master.stat().st_mtime), key=lambda f: f.name)
    return (newer[-1], "newest autoreview file (newer than the master)") if newer else (master, "the promoted master")


def embedding_inputs(methods_json: Path) -> dict:
    """id -> text exactly as 04_compute_embeddings.py embeds it (same filter)."""
    import json
    rows = json.loads(methods_json.read_text(encoding="utf-8"))
    return {e["id"]: f"{e['title']}. {e.get('abstract') or ''}".strip()
            for e in rows if not e.get("is_placeholder") and e.get("title")}


def embeddings_unchanged(new_methods: Path, clone: Path) -> bool:
    """True if the published embeddings.json still matches the new build's embedding inputs. UMAP is global, so one
    changed title/abstract would move every point and make the bot commit a ~6500-line diff for nothing."""
    import json
    old_methods, old_emb = clone / "docs" / "data" / "methods.json", clone / "docs" / "data" / "embeddings.json"
    if not (old_methods.exists() and old_emb.exists()):
        return False
    try:
        old_in, new_in = embedding_inputs(old_methods), embedding_inputs(new_methods)
        return old_in == new_in and set(json.loads(old_emb.read_text(encoding="utf-8"))) == set(new_in)
    except (OSError, ValueError, KeyError):
        return False


def run(cmd, env_extra=None, label=""):
    env = os.environ.copy()
    env.update(env_extra or {})
    print(f"\n=== {label}\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, env=env)
    if r.returncode != 0:
        die(f"{label} exited with {r.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--allow-drop", action="store_true")
    ap.add_argument("--skip-embeddings", action="store_true")
    ap.add_argument("--skip-publish", action="store_true")
    ap.add_argument("--no-lock-check", action="store_true")
    ap.add_argument("--clone", type=Path, default=Path.home() / "solr-publish")
    ap.add_argument("--expect-remote", default="marta-seq/solr")
    a = ap.parse_args()

    if not a.no_lock_check and lock_is_held():
        die(f"the orchestrator is running (lock {LOCK_FILE}); staging.xlsx is being written - try again later")

    base, why = choose_base()
    print(f"Base file: {base.name} ({why})")

    # fresh build dir every time: 02 would otherwise "resume" from a stale partial output
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    proc_dir, out_dir = BUILD_DIR / "processed", BUILD_DIR / "docs_data"
    proc_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)

    py = sys.executable
    run([py, "-m", "src.agents.living_ingestion.merge_candidates", "--base", base, "--output", SNAPSHOT_XLSX, "--keep-staging"],
        label="merge accumulated staging onto a copy of the base")
    if not SNAPSHOT_XLSX.exists():          # nothing staged -> the snapshot is just the base
        shutil.copyfile(base, SNAPSHOT_XLSX)
        print("No staged candidates: snapshot = base file.")

    env = {"SOLR_SOURCE_XLSX": str(SNAPSHOT_XLSX), "SOLR_PROCESSED_DIR": str(proc_dir),
           "SOLR_OUT_DIR": str(out_dir), "SOLR_SKIP_BACKUP": "1"}
    run([py, PRE / "01_parse_excel.py"], env, "01 parse")
    run([py, PRE / "02_fetch_metadata.py"], env, "02 fetch metadata (only rows missing it)")
    run([py, PRE / "03_export_json.py"], env, "03 export json")
    if not a.skip_embeddings and embeddings_unchanged(out_dir / "methods.json", a.clone):
        shutil.copyfile(a.clone / "docs" / "data" / "embeddings.json", out_dir / "embeddings.json")
        print("04 embeddings: title/abstract of every paper unchanged -> kept the published embeddings.json")
    elif not a.skip_embeddings:
        prev = {"SOLR_PREV_EMBEDDINGS": str(a.clone / "docs" / "data" / "embeddings.json"),
                "SOLR_PREV_METHODS": str(a.clone / "docs" / "data" / "methods.json")}   # stable layout: see 04's docstring
        run([py, PRE / "04_compute_embeddings.py"], {**env, **prev}, "04 embeddings (inputs changed; published layout reused)")
    else:
        # keep the published embeddings so the gate/publish compare like with like
        prev = a.clone / "docs" / "data" / "embeddings.json"
        if prev.exists():
            shutil.copyfile(prev, out_dir / "embeddings.json")
            print("embeddings.json: kept the currently published one (--skip-embeddings)")

    print(f"\nBuild finished: {out_dir}")
    if a.skip_publish:
        print("--skip-publish: stopping here.")
        return
    print("\n=== gate + publish" + (" (REAL PUSH)" if a.push else " (dry run)"))
    try:
        sys.exit(pub.publish(out_dir, a.clone, a.push, a.expect_remote, a.allow_drop))
    except pub.Refuse as e:
        print("REFUSED:", e)
        if not isinstance(e, pub.Precheck) and (a.clone / ".git").exists():
            pub.restore(a.clone)
        sys.exit(1)


if __name__ == "__main__":
    main()
