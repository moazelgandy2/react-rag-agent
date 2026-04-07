from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from threading import Lock
from uuid import uuid4

from .config import settings


@dataclass
class SessionData:
    session_id: str
    created_at: datetime
    updated_at: datetime
    messages: list[tuple[str, str]] = field(default_factory=list)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionData] = {}
        self._lock = Lock()

    def create(self) -> SessionData:
        now = datetime.now(tz=timezone.utc)
        session = SessionData(
            session_id=str(uuid4()),
            created_at=now,
            updated_at=now,
            messages=[],
        )
        with self._lock:
            self._prune_locked(now)
            if len(self._sessions) >= settings.session_max_count:
                oldest = min(self._sessions.values(), key=lambda item: item.updated_at)
                self._sessions.pop(oldest.session_id, None)
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> SessionData | None:
        now = datetime.now(tz=timezone.utc)
        with self._lock:
            self._prune_locked(now)
            session = self._sessions.get(session_id)
            if session is None:
                return None
            session.updated_at = now
            return session

    def append_exchange(
        self, session_id: str, user_message: str, assistant_message: str
    ) -> SessionData | None:
        now = datetime.now(tz=timezone.utc)
        with self._lock:
            self._prune_locked(now)
            session = self._sessions.get(session_id)
            if session is None:
                return None

            session.messages.append(("user", user_message))
            session.messages.append(("assistant", assistant_message))
            limit = settings.session_max_messages
            if len(session.messages) > limit:
                session.messages = session.messages[-limit:]
            session.updated_at = now
            return session

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list_messages(self, session_id: str) -> list[tuple[str, str]] | None:
        now = datetime.now(tz=timezone.utc)
        with self._lock:
            self._prune_locked(now)
            session = self._sessions.get(session_id)
            if session is None:
                return None
            return list(session.messages)

    def _prune_locked(self, now: datetime) -> None:
        ttl = timedelta(minutes=settings.session_ttl_minutes)
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.updated_at > ttl
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)


session_store = SessionStore()
