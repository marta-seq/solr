"""
weekly.py - the weekly cron entry point: scrape + agents -> build the site data -> gate -> publish as solr-bot -> email digest.

    python -m src.publish.weekly                 # everything, but the publish step is a DRY RUN
    python -m src.publish.weekly --push          # the real thing (this is what the cron line uses)
    options: --no-orchestrate (skip scrape+agents, e.g. to re-publish), --orchestrate-args "--dry-run --no-scrape" (pass-through),
             --max-papers N, --skip-embeddings, --no-email

What runs, in order (every step logged to data/weekly/<date>_<time>.log):
  0. weekly lock (never two runs at once; also refuses while an orchestrate run is active)
  1. backup of master + newest autoreview + staging.xlsx + scan_state.json -> data/weekly/backups/<stamp>/ (last 6 kept)
  2. `python -m src.agents.orchestrate`  (preflight, scrape, agents -> staging.xlsx; never merges, never touches master/git)
       exit 2 (a stage FAILED)  -> nothing is built/published, failure notice sent
       exit 0 / 1 (ok/degraded) -> continue
  3. `python -m src.publish.build_site [--push]`  (base+staging -> 01..04 -> gate -> publish; the gate may refuse)
  4. digest (ok / degraded) or failure notice (anything else), incl. a reminder when the last manual review is old: always written to
     data/weekly/last_digest.txt + STATUS.txt, and emailed if ~/.solr_secrets is complete (email NOT set up as of 2026-10-19 - see TODO).
Never raises out of main(): any exception becomes a failure email. Exit code 0 = published/nothing to publish, 1 = degraded, 2 = problem.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import notify  # noqa: E402

WEEKLY_DIR = ROOT / "data" / "weekly"
BACKUP_KEEP = 6
REVIEW_REMINDER_DAYS = 30
CLONE = Path.home() / "solr-publish"
CURATED = ROOT / "data" / "data_curated"
STAGING = ROOT / "data" / "agent_review" / "staging.xlsx"
SCAN_STATE = ROOT / "data" / "agent_review" / "runs" / "scan_state.json"
ORCH_LATEST = ROOT / "data" / "agent_review" / "runs" / "latest.json"
SITE_STATS = ROOT / "data" / "site_build" / "docs_data" / "stats.json"
PY = sys.executable


class Log:
    def __init__(self):
        WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
        self.path = WEEKLY_DIR / f"{dt.datetime.now():%Y-%m-%d_%H%M%S}.log"
        self._f = open(self.path, "a", encoding="utf-8", buffering=1)

    def __call__(self, msg: str):
        line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self._f.write(line + "\n")

    def stream(self, cmd, env=None) -> tuple:
        """Run cmd, copy its output into the log; returns (returncode, last 40 output lines)."""
        self(f"$ {' '.join(str(c) for c in cmd)}")
        p = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env={**os.environ, **(env or {})})
        tail = []
        for line in p.stdout:
            self._f.write("  | " + line if line.endswith("\n") else "  | " + line + "\n")
            tail.append(line.rstrip("\n"))
            tail = tail[-40:]
        p.wait()
        return p.returncode, tail


def acquire_lock() -> bool:
    lock = WEEKLY_DIR / "weekly.lock"
    WEEKLY_DIR.mkdir(parents=True, exist_ok=True)
    if lock.exists():
        try:
            os.kill(int(lock.read_text().split()[0]), 0)
            return False
        except (ValueError, IndexError, ProcessLookupError):
            pass
        except PermissionError:
            return False
    lock.write_text(f"{os.getpid()} {dt.datetime.now().isoformat()}\n")
    return True


def release_lock():
    try:
        (WEEKLY_DIR / "weekly.lock").unlink()
    except OSError:
        pass


def backup(log: Log) -> Path:
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    dest = WEEKLY_DIR / "backups" / stamp
    dest.mkdir(parents=True)
    files = [f for f in CURATED.glob("*.xlsx") if not f.name.startswith("~$")]
    files += [STAGING, SCAN_STATE]
    for f in files:
        if f.exists():
            shutil.copy2(f, dest / f.name)
    olds = sorted(p for p in (WEEKLY_DIR / "backups").iterdir() if p.is_dir())
    for old in olds[:-BACKUP_KEEP]:
        shutil.rmtree(old, ignore_errors=True)
    log(f"backup -> {dest.relative_to(ROOT)} ({len(list(dest.iterdir()))} files)")
    return dest


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def staged_candidates() -> int:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(STAGING, read_only=True)
        return max(0, sum(ws.max_row - 1 for ws in wb.worksheets))
    except Exception:  # noqa: BLE001
        return 0


def decide(orch_rc: int) -> tuple:
    """(build_allowed, label). The orchestrator reports 0 ok / 1 degraded / 2 failed."""
    if orch_rc == 2:
        return False, "FAILED"
    return True, "OK" if orch_rc == 0 else "DEGRADED"


def review_age_days(stats: dict):
    try:
        return (dt.date.today() - dt.date.fromisoformat(stats.get("last_human_review", ""))).days
    except ValueError:
        return None


def build_digest(status: str, ctx: dict) -> tuple:
    before, after = ctx.get("before", {}), ctx.get("after", {})
    lines = [f"SOLR weekly run - {status}", f"{dt.datetime.now():%Y-%m-%d %H:%M}", ""]
    if ctx.get("problem"):
        lines += ["PROBLEM: " + ctx["problem"], ""]
    outcome = ctx.get("publish_outcome")
    if outcome:
        lines += [f"Publish: {outcome}", ""]
    if after:
        d = lambda k: after.get(k, 0) - before.get(k, 0)  # noqa: E731
        lines += [f"Papers:   {after.get('total_papers')} ({d('total_papers'):+d} vs published)",
                  f"Datasets: {after.get('total_datasets')} ({d('total_datasets'):+d} vs published)",
                  f"Not yet manually reviewed: {after.get('unreviewed_papers')} papers, {after.get('unreviewed_datasets')} datasets", ""]
    lines.append(f"Candidates waiting for your next merge (staging.xlsx): {ctx.get('staged', 0)}")
    age = review_age_days(after or before)
    if age is not None and age > REVIEW_REMINDER_DAYS:
        lines.append(f"REMINDER: your last manual review was {age} days ago ({(after or before).get('last_human_review')}). "
                     f"Time to run the merge and review.")
    elif age is not None:
        lines.append(f"Last manual review: {age} days ago.")
    if ctx.get("pending_autoreview"):
        lines.append(f"An autoreview file is waiting for promotion: {ctx['pending_autoreview']}")
    if ctx.get("orch_stages"):
        lines += ["", "Orchestrator stages:"] + [f"  {s['stage']:<12} {s['status']:<9} {s.get('counts') or ''} {'; '.join(s.get('notes') or [])}"[:200]
                                                  for s in ctx["orch_stages"]]
    if ctx.get("build_tail"):
        lines += ["", "Build / gate output (last lines):"] + ["  " + l for l in ctx["build_tail"][-18:]]
    lines += ["", f"Full log: {ctx.get('log_path')}"]
    return f"[SOLR] weekly run {status}", "\n".join(lines)


def run(a, log: Log) -> int:
    ctx = {"log_path": str(log.path), "before": _json(CLONE / "docs" / "data" / "stats.json")}
    if not acquire_lock():
        log("another weekly run holds the lock - exiting")
        return 2
    status, rc = "FAILED", 2
    try:
        backup(log)
        orch_rc = 0
        if not a.no_orchestrate:
            cmd = [PY, "-m", "src.agents.orchestrate"] + (["--max-papers", str(a.max_papers)] if a.max_papers else []) + a.orchestrate_args.split()
            orch_rc, _ = log.stream(cmd)
            rep = _json(ORCH_LATEST)
            ctx["orch_stages"] = rep.get("stages", [])
            log(f"orchestrator exit {orch_rc} ({rep.get('overall')})")
        ok, status = decide(orch_rc)
        ctx["staged"] = staged_candidates()
        if not ok:
            ctx["problem"] = "the orchestrator FAILED, so nothing was built or published (see the stage table below)."
        else:
            cmd = [PY, "-m", "src.publish.build_site"] + (["--push"] if a.push else []) + (["--skip-embeddings"] if a.skip_embeddings else [])
            brc, tail = log.stream(cmd)
            ctx["build_tail"] = tail
            ctx["after"] = _json(SITE_STATS)
            text = "\n".join(tail)
            if brc != 0:
                status = "FAILED"
                ctx["problem"] = ("the site build or the publish gate REFUSED - nothing was published. "
                                  "Details below; the published site is unchanged.")
            elif "PUBLISHED" in text:
                ctx["publish_outcome"] = "pushed to GitHub as solr-bot"
            elif "Nothing to publish" in text:
                ctx["publish_outcome"] = "nothing new to publish"
            else:
                ctx["publish_outcome"] = "DRY RUN (not pushed)"
        pend = [f.name for f in CURATED.glob("*autoreview*.xlsx") if not f.name.startswith("~$")]
        ctx["pending_autoreview"] = ", ".join(sorted(pend)[-1:]) if pend else ""
        rc = {"OK": 0, "DEGRADED": 1}.get(status, 2)
    except Exception:  # noqa: BLE001
        log(traceback.format_exc())
        status, rc = "FAILED", 2
        ctx["problem"] = "unexpected error:\n" + traceback.format_exc()[-1500:]
    finally:
        release_lock()
    subject, body = build_digest(status, ctx)
    log("digest:\n" + body)
    # Always leave the digest on disk too (the only notification while email is not set up):
    #   data/weekly/STATUS.txt       one line - OK / DEGRADED / FAILED + time
    #   data/weekly/last_digest.txt  the full digest of the latest run
    (WEEKLY_DIR / "last_digest.txt").write_text(body + "\n", encoding="utf-8")
    (WEEKLY_DIR / "STATUS.txt").write_text(f"{status} {dt.datetime.now():%Y-%m-%d %H:%M} {log.path.name}\n", encoding="utf-8")
    if a.no_email:
        log("email skipped (--no-email)")
    else:
        sent, why = notify.send(subject, body)
        log(f"email: {why}")
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="really publish (default: the publish step is a dry run)")
    ap.add_argument("--no-orchestrate", action="store_true")
    ap.add_argument("--orchestrate-args", default="", help='extra args for the orchestrator, e.g. "--dry-run"')
    ap.add_argument("--max-papers", type=int, default=None)
    ap.add_argument("--skip-embeddings", action="store_true")
    ap.add_argument("--no-email", action="store_true")
    a = ap.parse_args()
    os.environ["PATH"] = "/usr/local/bin:/usr/bin:/bin:" + os.environ.get("PATH", "")   # cron's PATH is minimal (git, ssh)
    log = Log()
    sys.exit(run(a, log))


if __name__ == "__main__":
    main()
