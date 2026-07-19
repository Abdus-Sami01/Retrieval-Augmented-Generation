import json
import logging

import pytest

from rag.observability import JsonFormatter, log_duration


def _make_record(logger_name="rag.test", **extra):
    record = logging.LogRecord(
        name=logger_name, level=logging.INFO, pathname="x.py", lineno=1, msg="hello", args=(), exc_info=None
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_produces_valid_json():
    record = _make_record()
    formatted = JsonFormatter().format(record)
    parsed = json.loads(formatted)
    assert parsed["message"] == "hello"
    assert parsed["level"] == "INFO"
    assert parsed["logger"] == "rag.test"


def test_json_formatter_includes_extra_fields():
    record = _make_record(event="query", duration_ms=12.5, chunks_retrieved=3)
    parsed = json.loads(JsonFormatter().format(record))
    assert parsed["event"] == "query"
    assert parsed["duration_ms"] == 12.5
    assert parsed["chunks_retrieved"] == 3


def test_log_duration_emits_ok_status_and_duration(caplog):
    logger = logging.getLogger("rag.test.duration")
    logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="rag.test.duration"):
        with log_duration(logger, "test_event", question_len=5) as ctx:
            ctx["chunks_retrieved"] = 3

    record = caplog.records[0]
    assert record.event == "test_event"
    assert record.status == "ok"
    assert record.duration_ms >= 0
    assert record.question_len == 5
    assert record.chunks_retrieved == 3


def test_log_duration_emits_error_status_on_exception(caplog):
    logger = logging.getLogger("rag.test.duration.error")
    logger.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="rag.test.duration.error"):
        with pytest.raises(ValueError):
            with log_duration(logger, "test_event"):
                raise ValueError("boom")

    record = caplog.records[0]
    assert record.status == "error"
    assert record.error == "boom"
