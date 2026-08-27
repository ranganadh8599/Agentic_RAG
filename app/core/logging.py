# Agentic RAG - central logging setup.
#
# One clear, consistent format for every module (api, agents, retrieval,
# rerank, llm, ingest, ...) so you can follow a single request end-to-end
# from the server console:
#
#   2026-08-26 10:45:12 | INFO    | api.chat_endpoint      | chat in | user=testuser conv=None stream=True collection=None query='what is RAGAS'
#   2026-08-26 10:45:13 | DEBUG   | agents.run_stream      | rewrite 'what does it do?' -> 'What does RAGAS do?'
#   2026-08-26 10:45:13 | INFO    | agents.run_stream      | route=rag | collection=None | filters=None
#   2026-08-26 10:45:14 | INFO    | retrieval.retrieve     | retrieval cache HIT (3.2 ms)
#   2026-08-26 10:45:16 | INFO    | rerank.rerank          | rerank 20 candidates -> top 5 in 1950 ms on cuda:0
#   2026-08-26 10:45:20 | INFO    | agents.run_stream      | answer: chars=412 sources=3 | total 8.1 s
#
# Enable verbose tracing with LOG_LEVEL=DEBUG (per-call LLM/embedding timing,
# per-channel retrieval counts, critic feedback, ...).
#
# It is idempotent: safe to call from api startup, cli.py and test scripts.

import datetime
import logging
import os
import sys

_FORMAT = "%(asctime)s | %(levelname)-5s | %(message)s"
_DATEFMT = "%H:%M:%S"

# Every log line is mirrored to a file under LOG_DIR/<date>/app_<timestamp>.log
# for post-mortem inspection (disable with LOG_TO_FILE=0). LOG_DIR defaults to
# the repository root's logs/ directory (two levels up from app/core/).
LOG_DIR = os.getenv(
    "LOG_DIR",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "logs")))
_log_file_path: str | None = None


def current_log_file() -> str | None:
    """Absolute path of the active log file (None when file logging is off)."""
    return _log_file_path


def fmt_table(headers, rows, max_col_width=60):
    """Render rows as a simple aligned ASCII table for readable multi-line logs.

    headers: list[str]  rows: iterable of row tuples (cells are truncated to
    max_col_width so wide fields like snippets stay tidy).
    """
    widths = [len(str(h)) for h in headers]
    str_rows = []
    for r in rows:
        cells = []
        for i in range(len(headers)):
            s = str(r[i] if i < len(r) and r[i] is not None else "")
            if len(s) > max_col_width:
                s = s[: max_col_width - 3] + "..."
            cells.append(s)
            widths[i] = max(widths[i], len(s))
        str_rows.append(cells)

    def line(cells):
        return "  |" + "|".join(f" {str(c).ljust(widths[i])} "
                                for i, c in enumerate(cells)) + "|"

    border = "  +" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [border, line(headers), border]
    out += [line(c) for c in str_rows]
    out.append(border)
    return "\n".join(out)

# Reduce third-party noise BEFORE those libs get imported (this module is
# imported very early by api.py / cli.py).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("LITELLM_LOG", "WARNING")


def setup_logging(level: str | None = None) -> None:
    """Configure a single, clear console logger for the whole app.

    Replaces whatever handlers are already on the root logger (e.g. uvicorn's)
    so messages never double-print, and routes uvicorn's own loggers through
    the same handler for a unified look.

    level: LOG_LEVEL env var wins unless a level string is passed explicitly.
    """
    level = (level or os.getenv("LOG_LEVEL", "INFO")).strip().upper()
    numeric = getattr(logging, level, None)
    if not isinstance(numeric, int):
        numeric = logging.INFO

    root = logging.getLogger()
    root.setLevel(numeric)
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    root.addHandler(handler)

    # Mirror every log line to a date/time-based file for later inspection.
    global _log_file_path
    _log_file_path = None
    if os.getenv("LOG_TO_FILE", "1") == "1":
        try:
            now = datetime.datetime.now()
            day_dir = os.path.join(LOG_DIR, now.strftime("%Y-%m-%d"))
            os.makedirs(day_dir, exist_ok=True)
            path = os.path.join(day_dir,
                                f"app_{now.strftime('%Y-%m-%d_%H-%M-%S')}.log")
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
            root.addHandler(fh)
            _log_file_path = path
            root.info("📁 Log file: %s", path)
        except Exception:  # noqa: BLE001
            _log_file_path = None

    # Route uvicorn's own loggers through our root handler for a unified look.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    # Quiet noisy third-party loggers so only OUR pipeline steps show at INFO.
    # LOG_LEVEL=DEBUG reveals per-call LLM/embedding timings + litellm internals.
    for noisy in ("LiteLLM", "litellm", "httpx", "httpcore", "urllib3",
                  "sentence_transformers", "transformers", "tokenizers",
                  "PIL", "datasets", "filelock", "ragas", "huggingface",
                  "huggingface_hub", "huggingface_hub.utils._http",
                  "openai", "anthropic", "google", "google.genai",
                  "pymongo", "urllib3.connectionpool"):
        logging.getLogger(noisy).setLevel(
            logging.DEBUG if numeric <= logging.DEBUG else logging.ERROR)
