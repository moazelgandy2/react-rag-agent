import time
from collections import OrderedDict
from threading import Lock
from typing import Generic, TypeVar

from .config import settings

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, max_entries: int, ttl_seconds: int):
        self._max_entries = max(1, max_entries)
        self._ttl_seconds = max(1, ttl_seconds)
        self._lock = Lock()
        self._data: OrderedDict[str, tuple[float, T]] = OrderedDict()

    def get(self, key: str) -> T | None:
        now = time.time()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None

            timestamp, value = entry
            if now - timestamp > self._ttl_seconds:
                self._data.pop(key, None)
                return None

            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: T) -> None:
        now = time.time()
        with self._lock:
            self._data[key] = (now, value)
            self._data.move_to_end(key)
            self._evict_expired_locked(now)
            while len(self._data) > self._max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def _evict_expired_locked(self, now: float) -> None:
        expired_keys = [
            key for key, (timestamp, _) in self._data.items() if now - timestamp > self._ttl_seconds
        ]
        for key in expired_keys:
            self._data.pop(key, None)


retrieval_cache: TTLCache[list[dict]] = TTLCache(
    max_entries=settings.kv_cache_max_entries,
    ttl_seconds=settings.retrieval_cache_ttl_seconds,
)

orchestrator_cache: TTLCache[dict[str, str]] = TTLCache(
    max_entries=settings.kv_cache_max_entries,
    ttl_seconds=settings.orchestrator_cache_ttl_seconds,
)

response_cache: TTLCache[str] = TTLCache(
    max_entries=settings.kv_cache_max_entries,
    ttl_seconds=settings.response_cache_ttl_seconds,
)
