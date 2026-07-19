import threading
import time
from typing import Any


class TTLCache:
    """In-memory TTL cache with simple FIFO eviction at max_size. Fine for a single
    process; a multi-worker deployment would need a shared backend (e.g. Redis)
    for cache hits to be shared across workers."""

    def __init__(self, ttl_seconds: float, max_size: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key not in self._store and len(self._store) >= self.max_size:
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
            self._store[key] = (time.monotonic() + self.ttl_seconds, value)

    def size(self) -> int:
        with self._lock:
            return len(self._store)
