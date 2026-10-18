"""
orchestrate.py
One command for a whole review cycle, meant to be run by hand or by cron/systemd:

    python -m src.agents.orchestrate                  # preflight -> snapshot -> scrape -> agents (staging only)
    python -m src.agents.orchestrate --merge          # ... then merge_candidates (new *_autoreview_* file)
    python -m src.agents.orchestrate --refresh-site   # ... then 01->02->03->04 on the CURRENT master (docs/ updated)
    python -m src.agents.orchestrate --dry-run        # preflight + show what would run, change nothing

Stages (each is timed, caught, and reported as ok / degraded / failed / skipped):
    preflight  local LLM models reachable, exactly one master, no other run holding the lock. FAILED = abort.
    snapshot   data/processed CSVs older than the master (or missing) -> run 01 + 02 first. Agents must never run on a stale snapshot.
    scrape     PubMed + bioRxiv + medRxiv since the last successful scan (state file), through the existing funnel into staging.xlsx.
               Health: an exception, or zero records from a source over a multi-day window, = FAILED; far fewer records than last time = DEGRADED.
               A failed scrape does NOT stop the agents (they still have a queue); the run is just marked degraded/failed at the end.
    agents     `run_pipeline` as a subprocess (papers, applications, datasets), hop depth = config.MAX_HOPS (1).
    manual_list  refreshes data/manual_papers/NEEDED.html (papers still without full text; drop PDFs in that folder, any file name).
    merge      OPTIONAL (--merge). Refused when an unreviewed autoreview file newer than the master already exists.
    site       OPTIONAL (--refresh-site): 01 -> 02 -> 03 -> 04 on the CURRENT master. Not automatic - the site only changes
               when the master changes (after you promote a reviewed autoreview file).

Never writes to the master, never touches git. Agents write to data/agent_review/staging.xlsx only; promotion of a reviewed
autoreview file to master stays a manual step (then `--refresh-site`).

Outputs, per run, in data/agent_review/runs/<YYYY-MM-DD_HHMMSS>/: run.log, report.json, report.md.
Plus data/agent_review/runs/latest.json and STATUS.txt (one line: OK / DEGRADED / FAILED + time) for a cron/monitoring check.
Exit code: 0 ok, 1 degraded, 2 failed.
"""

import argparse
import contextlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

from .common import config

ROOT = config.ROOT
RUNS_DIR = config.STAGING_DIR / "runs"
STATE_FILE = RUNS_DIR / "scan_state.json"
LOCK_FILE = RUNS_DIR / "orchestrate.lock"
PREPROC = ROOT / "src" / "preprocessing"

OK, DEGRADED, FAILED, SKIPPED = "ok", "degraded", "failed", "skipped"
_RANK = {SKIPPED: 0, OK: 0, DEGRADED: 1, FAILED: 2}

SCAN_SOURCES = [("pubmed", None), ("biorxiv", "biorxiv"), ("medrxiv", "medrxiv")]
SCAN_OVERLAP_DAYS = 3          # re-scan a few days back: indexing lag; the seen-ledger dedups anyway
SHORT_WINDOW_DAYS = 3          # a window shorter than this may legitimately return nothing
DEGRADED_RATIO = 0.2           # fewer than 20% of the previous run's records from a source = suspicious


class Run:
    def __init__(self, args):
        self.args = args
        self.start = datetime.now()
        self.dir = RUNS_DIR / self.start.strftime("%Y-%m-%d_%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "run.log"
        self._log = open(self.log_path, "a", encoding="utf-8", buffering=1)
        self.stages = []

    def log(self, msg: str):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self._log.write(line + "\n")

    def raw(self, line: str):
        self._log.write(line if line.endswith("\n") else line + "\n")

    def add(self, name, status, seconds, counts=None, notes=None):
        self.stages.append({"stage": name, "status": status, "seconds": round(seconds, 1),
                            "counts": counts or {}, "notes": notes or []})
        self.log(f"--- {name}: {status.upper()} ({seconds:.0f}s) {counts or ''}")
        for n in notes or []:
            self.log(f"    {n}")

    @property
    def overall(self):
        worst = max((_RANK[s["status"]] for s in self.stages), default=0)
        return {0: OK, 1: DEGRADED, 2: FAILED}[worst]


class _Tee:
    def __init__(self, run):
        self.run = run

    def write(self, s):
        if s:
            self.run._log.write(s)
        return len(s)

    def flush(self):
        self.run._log.flush()


def run_cmd(run: Run, cmd: list, label: str) -> tuple:
    """Run a subprocess, streaming its output into run.log; returns (returncode, [lines])."""
    run.log(f"$ {' '.join(str(c) for c in cmd)}")
    lines = []
    proc = subprocess.Popen([str(c) for c in cmd], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=os.environ.copy())
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        run.raw(f"  | {line.rstrip()}")
    proc.wait()
    return proc.returncode, lines


# ── helpers ──────────────────────────────────────────────────────────────────
def find_master():
    cands = [f for f in config.CURATED_DIR.glob("*.xlsx")
             if "autoreview" not in f.name.lower() and not f.name.startswith("~$")]
    return cands[0] if len(cands) == 1 else None, cands


def snapshot_state(master: Path) -> tuple:
    """(is_fresh, reasons). Fresh = the three CSVs for this master's date tag exist and are newer than the master."""
    tag = master.stem.replace("datasets_curated_", "")
    need = [f"methods_{tag}.csv", f"datasets_{tag}.csv", f"methods_metadata_{tag}.csv"]
    reasons = []
    for n in need:
        p = ROOT / "data" / "processed" / n
        if not p.exists():
            reasons.append(f"missing {n}")
        elif p.stat().st_mtime < master.stat().st_mtime:
            reasons.append(f"{n} is older than the master")
    return not reasons, reasons


def pending_autoreview(master: Path) -> list:
    return [f.name for f in config.CURATED_DIR.glob("*autoreview*.xlsx")
            if not f.name.startswith("~$") and f.stat().st_mtime > master.stat().st_mtime]


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def acquire_lock(run: Run) -> bool:
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().split()[0])
            os.kill(pid, 0)
            run.log(f"Another run holds the lock (pid {pid}, {LOCK_FILE}).")
            return False
        except (ValueError, ProcessLookupError, IndexError):
            run.log("Stale lock file (process gone) - taking over.")
        except PermissionError:
            run.log("Lock held by a process we cannot signal - treating as alive.")
            return False
    LOCK_FILE.write_text(f"{os.getpid()} {datetime.now().isoformat()}\n")
    return True


# ── stages ───────────────────────────────────────────────────────────────────
def stage_preflight(run: Run, master) -> bool:
    t = time.time()
    notes, status = [], OK
    from .common import llm_client
    if master is None:
        _, cands = find_master()
        notes.append(f"expected exactly one non-autoreview .xlsx in {config.CURATED_DIR}, found {len(cands)}: {[c.name for c in cands]}")
        status = FAILED
    else:
        notes.append(f"master: {master.name}")
        pend = pending_autoreview(master)
        if pend:
            notes.append(f"NOTE: unreviewed autoreview file(s) newer than the master: {pend}")
    problems = llm_client.check_local_models()
    for p in problems:
        notes.append(f"LOCAL LLM PROBLEM: {p}")
    if problems:
        status = FAILED
    notes.append(f"LLM mode: {config.LLM_MODE}")
    run.add("preflight", status, time.time() - t, notes=notes)
    return status != FAILED


def stage_snapshot(run: Run, master: Path) -> bool:
    t = time.time()
    fresh, reasons = snapshot_state(master)
    if fresh:
        run.add("snapshot", SKIPPED, time.time() - t, notes=["processed CSVs are newer than the master"])
        return True
    if run.args.dry_run:
        run.add("snapshot", SKIPPED, time.time() - t, notes=["dry-run; would rebuild 01 -> 02: " + "; ".join(reasons)])
        return True
    run.log("Snapshot stale (" + "; ".join(reasons) + ") - rebuilding 01 -> 02.")
    for script in ("01_parse_excel.py", "02_fetch_metadata.py"):
        rc, _ = run_cmd(run, [sys.executable, PREPROC / script], script)
        if rc != 0:
            run.add("snapshot", FAILED, time.time() - t, notes=[f"{script} exited {rc}; agents will NOT run on a stale snapshot"])
            return False
    run.add("snapshot", OK, time.time() - t, notes=["rebuilt 01 -> 02"])
    return True


def stage_scrape(run: Run):
    t = time.time()
    if run.args.dry_run or run.args.no_scrape:
        run.add("scrape", SKIPPED, time.time() - t, notes=["dry-run / --no-scrape"])
        return
    from .living_ingestion.literature_search import scan
    state = load_state()
    today = date.today()
    status, notes, counts = OK, [], {}
    advanced = dict(state)
    for source, server in SCAN_SOURCES:
        if source in run.args.skip_source:
            notes.append(f"{source}: skipped (--skip-source)")
            continue
        last_end = state.get(source, {}).get("last_end")
        start = (date.fromisoformat(last_end) - timedelta(days=SCAN_OVERLAP_DAYS)) if last_end \
            else today - timedelta(days=run.args.first_lookback_days)
        window_days = (today - start).days
        try:
            with contextlib.redirect_stdout(_Tee(run)):
                if source == "pubmed":
                    res = scan.scan_pubmed(mindate=start.strftime("%Y/%m/%d"), maxdate=today.strftime("%Y/%m/%d"),
                                           max_new=run.args.max_new)
                else:
                    res = scan.scan_biorxiv(start.isoformat(), today.isoformat(), server=server, max_new=run.args.max_new)
        except Exception as e:
            run.raw(traceback.format_exc())
            notes.append(f"{source}: EXCEPTION {type(e).__name__}: {e}")
            status = FAILED
            continue
        total = sum(res.values())
        counts[source] = {"records": total, **res}
        prev_total = state.get(source, {}).get("last_total")
        if total == 0 and window_days >= SHORT_WINDOW_DAYS:
            notes.append(f"{source}: ZERO records for {start}..{today} - the source or our client is probably broken")
            status = FAILED
            continue
        if prev_total and prev_total >= 20 and total < DEGRADED_RATIO * prev_total:
            notes.append(f"{source}: only {total} records vs {prev_total} last run - suspicious")
            status = max(status, DEGRADED, key=_RANK.get)
        advanced[source] = {"last_end": today.isoformat(), "last_total": total}
    if run.args.max_new is None:
        save_state(advanced)   # a failed source never advanced its own window, so it is retried from the old one
    else:
        notes.append("--max-new set: scan state NOT advanced (the scan stopped early, the window is not fully processed)")
    run.add("scrape", status, time.time() - t, counts=counts, notes=notes)


_RX = {
    "papers": re.compile(r"Paper queue done: (\d+) papers processed, (\d+) new method entries, (\d+) dataset entries"),
    "applications": re.compile(r"Application queue done: (\d+) AP_pub papers processed, (\d+) dataset entries"),
    "datasets": re.compile(r"Data pool done: (\d+) entries filled, (\d+) skipped"),
}


def stage_agents(run: Run):
    t = time.time()
    if run.args.dry_run or run.args.no_agents:
        run.add("agents", SKIPPED, time.time() - t, notes=["dry-run / --no-agents"])
        return
    cmd = [sys.executable, "-m", "src.agents.run_pipeline", "--stages", run.args.stages]
    if run.args.max_papers:
        cmd += ["--max-papers", str(run.args.max_papers)]
    rc, lines = run_cmd(run, cmd, "run_pipeline")
    text = "\n".join(lines)
    counts, notes, status = {}, [], OK
    m = _RX["papers"].search(text)
    if m:
        counts["papers_processed"], counts["new_method_entries"], counts["datasets_linked"] = map(int, m.groups())
    m = _RX["applications"].search(text)
    if m:
        counts["applications_processed"], counts["app_datasets_linked"] = map(int, m.groups())
    m = _RX["datasets"].search(text)
    if m:
        counts["datasets_filled"], counts["datasets_skipped"] = map(int, m.groups())
    if rc != 0:
        status = FAILED
        notes.append(f"run_pipeline exited {rc}")
    if "LOCAL LLM PROBLEM" in text:
        status = FAILED
        notes.append("run_pipeline stopped at its own LLM preflight")
    if "fallback chain was fully exhausted" in text or "ended early" in text:
        status = max(status, DEGRADED, key=_RANK.get)
        notes.append("run ended early: LLM chain exhausted - queued work remains")
    # Fetch health: papers whose text could not be retrieved are silently useless (both desks skip them), so a run where most papers
    # fail to fetch is NOT a healthy run even though nothing crashed.
    n_nofetch = text.count("no text could be fetched from any source")
    n_done = counts.get("papers_processed", 0) + counts.get("applications_processed", 0)
    counts["papers_without_text"] = n_nofetch
    if n_done >= 3 and n_nofetch >= 0.5 * n_done:
        status = max(status, DEGRADED, key=_RANK.get)
        notes.append(f"{n_nofetch}/{n_done} papers had no retrievable text (paywall, or fetch sources down) - desks skipped them")
    # In local mode every LLM call must go to a local_<stage> provider; anything else means an API call slipped in.
    if config.LLM_MODE == "local":
        providers = set(re.findall(r"\[llm_client\] (\S+?)/", text))
        non_local = sorted(p for p in providers if not p.startswith("local_"))
        counts["llm_calls_local_providers"] = len(providers)
        if non_local:
            status = max(status, DEGRADED, key=_RANK.get)
            notes.append(f"NON-LOCAL LLM provider(s) used in local mode: {non_local}")
    n_tb = text.count("Traceback (most recent call last)")
    if n_tb:
        status = max(status, DEGRADED, key=_RANK.get)
        notes.append(f"{n_tb} traceback(s) in the agent output - see run.log")
    if status == OK and not counts:
        status = DEGRADED
        notes.append("no stage summary found in the agent output - something is off, see run.log")
    run.add("agents", status, time.time() - t, counts=counts, notes=notes)


def stage_manual_list(run: Run):
    """Refresh data/manual_papers/NEEDED.html|csv (papers still without full text) so each run ends with an up-to-date to-do list."""
    t = time.time()
    if run.args.dry_run:
        run.add("manual_list", SKIPPED, time.time() - t, notes=["dry-run"])
        return
    rc, lines = run_cmd(run, [sys.executable, "-m", "src.agents.common.manual_papers", "needed"], "manual_papers")
    summary = next((l for l in lines if "need a manual PDF" in l), "")
    m = re.search(r"(\d+) papers need a manual PDF", summary)
    run.add("manual_list", OK if rc == 0 else DEGRADED, time.time() - t,
            counts={"papers_needing_manual_pdf": int(m.group(1))} if m else {},
            notes=["data/manual_papers/NEEDED.html"] if rc == 0 else [f"manual_papers needed exited {rc}"])


def stage_merge(run: Run, master: Path):
    t = time.time()
    if not run.args.merge:
        run.add("merge", SKIPPED, time.time() - t, notes=["not requested (--merge); staging.xlsx keeps accumulating"])
        return
    pend = pending_autoreview(master)
    if pend:
        run.add("merge", SKIPPED, time.time() - t,
                notes=[f"refused: unreviewed autoreview file(s) already newer than the master: {pend}. Review/promote first."])
        return
    rc, _ = run_cmd(run, [sys.executable, "-m", "src.agents.living_ingestion.merge_candidates"], "merge_candidates")
    run.add("merge", OK if rc == 0 else FAILED, time.time() - t,
            notes=[] if rc == 0 else [f"merge_candidates exited {rc}"])


def stage_site(run: Run):
    t = time.time()
    if not run.args.refresh_site:
        run.add("site", SKIPPED, time.time() - t, notes=["not requested (--refresh-site); the site only changes when the master changes"])
        return
    if run.args.dry_run:
        run.add("site", SKIPPED, time.time() - t, notes=["dry-run"])
        return
    for script in ("01_parse_excel.py", "02_fetch_metadata.py", "03_export_json.py", "04_compute_embeddings.py"):
        rc, _ = run_cmd(run, [sys.executable, PREPROC / script], script)
        if rc != 0:
            run.add("site", FAILED, time.time() - t, notes=[f"{script} exited {rc}; docs/ may be partly updated"])
            return
    run.add("site", OK, time.time() - t, notes=["01 -> 02 -> 03 -> 04 done; docs/data/* updated - commit/push is manual"])


# ── reporting ────────────────────────────────────────────────────────────────
def write_report(run: Run):
    end = datetime.now()
    rep = {"started": run.start.isoformat(timespec="seconds"), "finished": end.isoformat(timespec="seconds"),
           "seconds": round((end - run.start).total_seconds()), "overall": run.overall, "stages": run.stages,
           "args": {k: v for k, v in vars(run.args).items()}}
    (run.dir / "report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    md = [f"# SOLR run {run.start:%Y-%m-%d %H:%M} - {run.overall.upper()}", "",
          f"Duration {rep['seconds']}s. Log: `{run.log_path.relative_to(ROOT)}`", "",
          "| stage | status | time | details |", "|---|---|---|---|"]
    for s in run.stages:
        det = "; ".join([f"{k}={v}" for k, v in s["counts"].items()] + s["notes"]).replace("|", "/")
        md.append(f"| {s['stage']} | {s['status']} | {s['seconds']}s | {det} |")
    (run.dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    RUNS_DIR.joinpath("latest.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    RUNS_DIR.joinpath("STATUS.txt").write_text(f"{run.overall.upper()} {end:%Y-%m-%d %H:%M} {run.dir.name}\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", default="papers,applications,datasets", help="passed to run_pipeline --stages")
    ap.add_argument("--max-papers", type=int, default=None, help="passed to run_pipeline --max-papers (default: config.MAX_PAPERS_PER_RUN)")
    ap.add_argument("--max-new", type=int, default=None, help="cap on newly staged scrape candidates per source")
    ap.add_argument("--first-lookback-days", type=int, default=14, help="scan window when a source has no state yet")
    ap.add_argument("--skip-source", action="append", default=[], choices=[s for s, _ in SCAN_SOURCES])
    ap.add_argument("--no-scrape", action="store_true")
    ap.add_argument("--no-agents", action="store_true")
    ap.add_argument("--merge", action="store_true", help="after the agents, run merge_candidates (creates an autoreview file)")
    ap.add_argument("--refresh-site", action="store_true", help="after everything, run 01->02->03->04 on the current master")
    ap.add_argument("--dry-run", action="store_true", help="preflight and report only")
    args = ap.parse_args(argv)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run = Run(args)
    if not acquire_lock(run):
        run.add("preflight", FAILED, 0, notes=["another run is active"])
        write_report(run)
        return 2
    try:
        run.log(f"Run {run.dir.name}: stages={args.stages} hops={config.MAX_HOPS} max_papers={args.max_papers or config.MAX_PAPERS_PER_RUN}")
        master, _ = find_master()
        if stage_preflight(run, master):
            if stage_snapshot(run, master):
                stage_scrape(run)
                stage_agents(run)
                stage_manual_list(run)
                stage_merge(run, master)
            stage_site(run)
    except Exception:
        run.raw(traceback.format_exc())
        run.add("orchestrator", FAILED, 0, notes=["unhandled exception - see run.log"])
    finally:
        write_report(run)
        with contextlib.suppress(OSError):
            LOCK_FILE.unlink()
    run.log(f"RUN {run.overall.upper()} - report: {(run.dir / 'report.md').relative_to(ROOT)}")
    return {OK: 0, DEGRADED: 1, FAILED: 2}[run.overall]


if __name__ == "__main__":
    sys.exit(main())
