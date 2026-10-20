"""
notify.py - send the weekly digest / failure notice by email (stdlib smtplib, STARTTLS).

Credentials live ONLY in ~/.solr_secrets (outside every repo, mode 600), as KEY=VALUE lines:
    SMTP_HOST=smtp.gmail.com        (default)
    SMTP_PORT=587                   (default)
    SMTP_USER=you@gmail.com
    SMTP_PASSWORD=<a Gmail *app password*, not the account password>
    MAIL_TO=you@gmail.com           (default: SMTP_USER)
Never raises: if the secrets are missing/incomplete or sending fails, it returns (False, reason) and the caller just logs it.
"""
import smtplib
import ssl
import stat
from email.message import EmailMessage
from pathlib import Path

SECRETS = Path.home() / ".solr_secrets"


def _read_secrets() -> dict:
    out = {}
    for line in SECRETS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def send(subject: str, body: str) -> tuple:
    try:
        if not SECRETS.exists():
            return False, f"{SECRETS} not found"
        if SECRETS.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            return False, f"{SECRETS} is readable by others - run: chmod 600 {SECRETS}"
        cfg = _read_secrets()
        user, pw = cfg.get("SMTP_USER", ""), cfg.get("SMTP_PASSWORD", "")
        if not user or not pw:
            return False, "SMTP_USER / SMTP_PASSWORD missing in the secrets file"
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = cfg.get("MAIL_TO") or user
        msg.set_content(body)
        with smtplib.SMTP(cfg.get("SMTP_HOST", "smtp.gmail.com"), int(cfg.get("SMTP_PORT", "587")), timeout=30) as s:
            s.starttls(context=ssl.create_default_context())
            s.login(user, pw)
            s.send_message(msg)
        return True, "sent"
    except Exception as e:  # noqa: BLE001 - notification problems must never break the run
        return False, f"{type(e).__name__}: {e}"


if __name__ == "__main__":
    # python -m src.publish.notify   -> sends a test email using ~/.solr_secrets
    ok, why = send("[SOLR] test email", "If you can read this, the SOLR weekly digest email works.")
    print("sent" if ok else f"NOT sent: {why}")
    raise SystemExit(0 if ok else 1)
