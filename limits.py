"""Limits that protect the Groq key on a public site: requests per minute per address, and a daily budget."""
import time
from collections import defaultdict, deque
from datetime import datetime, timezone


class RateLimiter:
    """At most `per_minute` requests per key in any 60 seconds, held in memory (so use a single worker)."""

    def __init__(self, per_minute=10, max_keys=10000):
        self.per_minute, self.max_keys, self.hits = per_minute, max_keys, defaultdict(deque)

    def allow(self, key, now=None):
        now = time.monotonic() if now is None else now
        q = self.hits[key]
        while q and now - q[0] >= 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        if len(self.hits) > self.max_keys:
            for k in [k for k, v in self.hits.items() if not v or now - v[-1] >= 60]:
                del self.hits[k]
        return True


class DailyBudget:
    """At most `limit` model-backed requests per UTC day across all visitors."""

    def __init__(self, limit=1000):
        self.limit, self.day, self.used = limit, None, 0

    def take(self, today=None):
        today = today or datetime.now(timezone.utc).date()
        if today != self.day:
            self.day, self.used = today, 0
        if self.used >= self.limit:
            return False
        self.used += 1
        return True
