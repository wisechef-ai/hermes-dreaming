"""Cognee sink for hermes-dreaming (Option 2 re-plumb, 2026-07-14).

Routes durable FACTS distilled by dreaming to the Layer-2 Cognee brain
(unbounded, auto-recalled per turn by the memory-router hook) instead of
the 2,200-char MEMORY.md injection window.

Design rules:
- Vendored REST client, no SDK dep — same multipart /api/v1/add shape as
  the former gbrain_to_cognee_facts.py bridge (retired with GBrain 2026-09-23).
- FAIL-SAFE, never FAIL-OPEN-SILENT: when Cognee is unreachable the fact
  is spooled to ~/.hermes/dreaming/cognee-spool/ and retried on the next
  apply/cron run via flush_spool(). A distilled fact is never dropped.
- Config from ~/.hermes/config.yaml memory.cognee.{host,dataset_name};
  tiny line parser, no PyYAML hard dep.

CLI: python3 -m hermes_dreaming.cognee_client --flush-spool [--status]
"""

from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

HERMES_HOME = Path.home() / ".hermes"
CONFIG_YAML = HERMES_HOME / "config.yaml"
SPOOL_DIR = HERMES_HOME / "dreaming" / "cognee-spool"

DEFAULT_COGNEE_HOST = "http://100.106.15.177:8001"
DEFAULT_COGNEE_DATASET = "hermes-tori"


def load_cognee_config() -> Tuple[str, str]:
    """Return (host, dataset) from config.yaml memory.cognee block."""
    host = DEFAULT_COGNEE_HOST
    dataset = DEFAULT_COGNEE_DATASET
    if not CONFIG_YAML.exists():
        return host, dataset
    text = CONFIG_YAML.read_text(encoding="utf-8", errors="replace")
    in_memory = False
    in_cognee = False
    memory_indent = -1
    cognee_indent = -1
    for line in text.splitlines():
        stripped = line.rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if stripped.lstrip() == "memory:":
            in_memory = True
            memory_indent = indent
            in_cognee = False
            continue
        if in_memory and indent <= memory_indent and stripped.lstrip() != "memory:":
            in_memory = False
            in_cognee = False
        if in_memory and stripped.lstrip() == "cognee:":
            in_cognee = True
            cognee_indent = indent
            continue
        if in_cognee and indent <= cognee_indent and stripped.lstrip() != "cognee:":
            in_cognee = False
        if in_cognee:
            m = re.match(r"\s*([a-zA-Z_]+)\s*:\s*(.+?)\s*$", line)
            if m:
                k, v = m.group(1), m.group(2)
                v = v.strip().strip('"').strip("'")
                if k == "host":
                    host = v
                elif k == "dataset_name":
                    dataset = v
    return host, dataset


def cognee_health_ok(host: str, timeout: float = 8.0) -> bool:
    try:
        req = Request(host.rstrip("/") + "/health", method="GET")
        with urlopen(req, timeout=timeout) as resp:
            return resp.status == 200 and "healthy" in resp.read().decode(
                "utf-8", errors="replace"
            )
    except Exception:
        return False


def _cognee_add(host: str, dataset: str, content: str, timeout: float = 30.0) -> Dict[str, Any]:
    """POST content to /api/v1/add as a small text file. Raises on transport error."""
    url = f"{host.rstrip('/')}/api/v1/add"
    boundary = f"----dreaming2cognee{int(time.time() * 1000)}"
    parts: List[bytes] = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="datasetName"\r\n\r\n')
    parts.append(dataset.encode("utf-8"))
    parts.append(b"\r\n")
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        b'Content-Disposition: form-data; name="data"; filename="dreaming-fact.txt"\r\n'
    )
    parts.append(b"Content-Type: text/plain\r\n\r\n")
    parts.append(content.encode("utf-8"))
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    req = Request(
        url,
        data=b"".join(parts),
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"status": "ok", "raw": raw}


def _spool(content: str, kind: str) -> Path:
    SPOOL_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = SPOOL_DIR / f"{stamp}-{kind}.txt"
    path.write_text(content, encoding="utf-8")
    return path


def send_fact(content: str, *, kind: str = "fact", provenance: str = "") -> Dict[str, Any]:
    """Send one distilled fact to Cognee. NEVER raises.

    Returns {"sent": True, ...} on success, or {"sent": False,
    "spooled": "<path>"} when Cognee is unreachable — the fact is
    preserved on disk and flushed on a later run.
    """
    stamped = content.strip()
    if not stamped:
        return {"sent": False, "skipped": "empty"}
    header = f"[hermes-dreaming {kind} {datetime.now(timezone.utc).date().isoformat()}]"
    if provenance:
        header += f" (source: {provenance})"
    payload = f"{header}\n{stamped}\n"
    host, dataset = load_cognee_config()
    try:
        result = _cognee_add(host, dataset, payload)
        if "error" in result:
            raise URLError(str(result["error"]))
        return {"sent": True, "dataset": dataset, "result": result}
    except Exception as exc:
        path = _spool(payload, kind)
        return {"sent": False, "error": str(exc), "spooled": str(path)}


def flush_spool() -> Dict[str, Any]:
    """Retry every spooled fact. Sent files are deleted; failures stay."""
    if not SPOOL_DIR.exists():
        return {"pending": 0, "flushed": 0}
    files = sorted(SPOOL_DIR.glob("*.txt"))
    if not files:
        return {"pending": 0, "flushed": 0}
    host, dataset = load_cognee_config()
    if not cognee_health_ok(host):
        return {"pending": len(files), "flushed": 0, "note": "cognee down"}
    flushed = 0
    for f in files:
        try:
            result = _cognee_add(host, dataset, f.read_text(encoding="utf-8"))
            if "error" in result:
                raise URLError(str(result["error"]))
            f.unlink()
            flushed += 1
        except Exception:
            break  # keep order; retry next run
    return {"pending": len(files) - flushed, "flushed": flushed}


def main(argv: List[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if "--flush-spool" in args:
        print(json.dumps(flush_spool()))
        return 0
    if "--status" in args:
        host, dataset = load_cognee_config()
        pending = len(list(SPOOL_DIR.glob("*.txt"))) if SPOOL_DIR.exists() else 0
        print(
            json.dumps(
                {
                    "host": host,
                    "dataset": dataset,
                    "healthy": cognee_health_ok(host),
                    "spool_pending": pending,
                }
            )
        )
        return 0
    print("usage: python3 -m hermes_dreaming.cognee_client --flush-spool | --status")
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
