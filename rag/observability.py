import json
import logging
import sys
import time
from contextlib import contextmanager

_RESERVED_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    root = logging.getLogger("rag")
    root.setLevel(level)
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


@contextmanager
def log_duration(logger: logging.Logger, event: str, **extra):
    start = time.perf_counter()
    status = "ok"
    error = None
    try:
        yield extra
    except Exception as exc:
        status = "error"
        error = str(exc)
        raise
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            event,
            extra={"event": event, "status": status, "duration_ms": duration_ms, "error": error, **extra},
        )
