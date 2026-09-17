"""Small shared progress/status writer used by the GA worker and stage scripts."""
import json
import os
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATUS_FILE = os.path.join(BASE_DIR, "ga_status.json")


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def write_progress(**kwargs):
    current = {}
    try:
        if os.path.exists(STATUS_FILE):
            with open(STATUS_FILE, "r", encoding="utf-8") as fh:
                current = json.load(fh)
    except Exception:
        current = {}

    message = kwargs.get("message")
    current.update(kwargs)
    current["updated_at"] = now_iso()

    if message:
        history = current.get("activity", [])
        if not isinstance(history, list):
            history = []
        stamp = datetime.now().strftime("%H:%M:%S")
        entry = f"{stamp} · {message}"
        if not history or history[-1] != entry:
            history.append(entry)
        current["activity"] = history[-20:]

    tmp = STATUS_FILE + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(current, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, STATUS_FILE)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
