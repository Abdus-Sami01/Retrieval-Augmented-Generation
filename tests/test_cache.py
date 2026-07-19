import time

from rag.cache import TTLCache


def test_get_missing_key_returns_none():
    cache = TTLCache(ttl_seconds=60)
    assert cache.get("missing") is None


def test_set_then_get_returns_value():
    cache = TTLCache(ttl_seconds=60)
    cache.set("a", "value")
    assert cache.get("a") == "value"


def test_entry_expires_after_ttl():
    cache = TTLCache(ttl_seconds=0.01)
    cache.set("a", "value")
    time.sleep(0.03)
    assert cache.get("a") is None


def test_evicts_oldest_entry_when_max_size_exceeded():
    cache = TTLCache(ttl_seconds=60, max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert cache.size() == 2
    assert cache.get("a") is None
    assert cache.get("c") == 3


def test_updating_existing_key_does_not_evict():
    cache = TTLCache(ttl_seconds=60, max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("a", 99)
    assert cache.size() == 2
    assert cache.get("a") == 99
    assert cache.get("b") == 2
