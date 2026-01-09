"""
Marrow - The Living Core of ScornSpine
======================================

Handles real-time conversation logging with a hot buffer for immediate
retrieval. The marrow is what keeps the spine alive - it's the active
component that captures and preserves memory as it happens.

Architecture:
    - Hot Buffer: Last N messages in memory (immediate availability)
    - Disk Storage: JSONL files for persistence
    - RAG Integration: Triggers reindex after threshold

Usage:
    from haloscorn.scornspine.marrow import ConversationLogger

    logger = ConversationLogger(agent="vera")
    logger.log("user", "What did we decide about particle counts?")
    logger.log("assistant", "The minimum is 6 million particles...")

    # Search recent (includes hot buffer)
    recent = logger.search_recent("particle")
"""

import json
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Deque
import threading
import logging

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONVERSATIONS_DIR = PROJECT_ROOT / "logs" / "conversations"


# -------------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------------

@dataclass
class MarrowConfig:
    """Configuration for conversation logging."""
    hot_buffer_size: int = 100          # Messages kept in memory
    flush_threshold: int = 10           # Flush to disk every N messages
    reindex_threshold: int = 50         # Trigger RAG reindex every N messages
    max_file_size_mb: float = 10.0      # Max size per JSONL file before rotation


# -------------------------------------------------------------------------------
# DATA STRUCTURES
# -------------------------------------------------------------------------------

@dataclass
class ConversationEntry:
    """A single conversation turn."""
    timestamp: str
    role: str           # "user" or "assistant"
    agent: str          # "vera", "jonah", "halo"
    content: str
    session_id: Optional[str] = None
    metadata: Optional[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ConversationEntry':
        return cls(**data)


# -------------------------------------------------------------------------------
# CONVERSATION LOGGER
# -------------------------------------------------------------------------------

class ConversationLogger:
    """
    Real-time conversation capture with hot buffer and disk persistence.

    The marrow keeps memory ALIVE by:
    1. Keeping recent messages in a hot buffer (instant retrieval)
    2. Flushing to disk periodically (persistence)
    3. Triggering RAG reindex when threshold reached (searchability)
    """

    def __init__(self, agent: str, config: Optional[MarrowConfig] = None):
        """
        Initialize conversation logger for an agent.

        Args:
            agent: Agent name (vera, jonah, halo)
            config: Optional configuration
        """
        self.agent = agent.lower()
        self.config = config or MarrowConfig()

        # Hot buffer - deque with max size for automatic eviction
        self.hot_buffer: Deque[ConversationEntry] = deque(maxlen=self.config.hot_buffer_size)

        # Pending writes
        self._pending: List[ConversationEntry] = []
        self._message_count = 0
        self._session_id = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Thread safety
        self._lock = threading.Lock()

        # Ensure directories exist
        self._agent_dir = CONVERSATIONS_DIR / self.agent
        self._agent_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[Marrow] Initialized for agent: {agent}")

    def log(
        self,
        role: str,
        content: str,
        metadata: Optional[dict] = None
    ) -> ConversationEntry:
        """
        Log a conversation turn.

        Args:
            role: "user" or "assistant"
            content: The message content
            metadata: Optional additional data

        Returns:
            The logged entry
        """
        with self._lock:
            entry = ConversationEntry(
                timestamp=datetime.now().isoformat(),
                role=role,
                agent=self.agent,
                content=content,
                session_id=self._session_id,
                metadata=metadata
            )

            # Always add to hot buffer (immediate availability)
            self.hot_buffer.append(entry)

            # Add to pending writes
            self._pending.append(entry)
            self._message_count += 1

            # Flush to disk if threshold reached
            if len(self._pending) >= self.config.flush_threshold:
                self._flush_to_disk()

            # Trigger RAG reindex if threshold reached
            if self._message_count % self.config.reindex_threshold == 0:
                self._trigger_reindex()

            return entry

    def log_user(self, content: str, metadata: Optional[dict] = None) -> ConversationEntry:
        """Convenience method to log a user message."""
        return self.log("user", content, metadata)

    def log_assistant(self, content: str, metadata: Optional[dict] = None) -> ConversationEntry:
        """Convenience method to log an assistant message."""
        return self.log("assistant", content, metadata)

    def search_recent(self, query: str, limit: int = 10) -> List[ConversationEntry]:
        """
        Search the hot buffer for recent messages matching query.

        This is FAST because it only searches in-memory data.
        For full history search, use ScornSpine.query().

        Args:
            query: Search term (case-insensitive substring match)
            limit: Maximum results to return

        Returns:
            Matching entries, newest first
        """
        with self._lock:
            query_lower = query.lower()
            matches = [
                entry for entry in reversed(self.hot_buffer)
                if query_lower in entry.content.lower()
            ]
            return matches[:limit]

    def get_recent(self, limit: int = 20) -> List[ConversationEntry]:
        """
        Get the most recent messages from hot buffer.

        Args:
            limit: Maximum messages to return

        Returns:
            Recent entries, newest first
        """
        with self._lock:
            return list(reversed(list(self.hot_buffer)))[:limit]

    def get_session_messages(self) -> List[ConversationEntry]:
        """Get all messages from current session."""
        with self._lock:
            return [e for e in self.hot_buffer if e.session_id == self._session_id]

    def flush(self):
        """Force flush pending writes to disk."""
        with self._lock:
            self._flush_to_disk()

    def _flush_to_disk(self):
        """Write pending entries to JSONL file."""
        if not self._pending:
            return

        today = datetime.now().strftime("%Y-%m-%d")
        filepath = self._agent_dir / f"{today}.jsonl"

        # Check file size and rotate if needed
        if filepath.exists() and filepath.stat().st_size > self.config.max_file_size_mb * 1024 * 1024:
            timestamp = datetime.now().strftime("%H%M%S")
            filepath = self._agent_dir / f"{today}-{timestamp}.jsonl"

        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                for entry in self._pending:
                    f.write(json.dumps(entry.to_dict()) + '\n')

            logger.debug(f"[Marrow] Flushed {len(self._pending)} entries to {filepath}")
            self._pending = []

        except Exception as e:
            logger.error(f"[Marrow] Flush failed: {e}")

    # RT2300: Debounce refresh to prevent flooding (JONAH fix)
    _last_refresh_time = 0
    _REFRESH_DEBOUNCE_SECONDS = 60

    def _trigger_reindex(self):
        """
        Trigger ScornSpine incremental refresh (ADR-0041).

        Uses /refresh endpoint for fast incremental updates instead of
        full /index which caused 70-80% GPU spikes.

        RT2300: Debounce - skip if last refresh was <60 seconds ago.
        """
        import threading
        import time as _time

        now = _time.time()
        if now - ConversationLogger._last_refresh_time < ConversationLogger._REFRESH_DEBOUNCE_SECONDS:
            logger.debug(f"[Marrow] Skipping /refresh - last was {now - ConversationLogger._last_refresh_time:.1f}s ago")
            return

        ConversationLogger._last_refresh_time = now

        def refresh():
            try:
                import requests
                response = requests.post("http://127.0.0.1:7780/refresh", timeout=30)
                if response.ok:
                    data = response.json()
                    changed = data.get('changed_docs', 0)
                    logger.info(f"[Marrow] Triggered ScornSpine refresh ({changed} changed)")
            except Exception as e:
                logger.debug(f"[Marrow] Refresh trigger failed (server may be down): {e}")

        # Run in background thread to not block
        thread = threading.Thread(target=refresh, daemon=True)
        thread.start()

    def archive_session(self, summary: Optional[str] = None) -> Optional[Path]:
        """
        Archive the current session to a markdown file.
        Called during the Stirring Protocol.

        Args:
            summary: Optional session summary

        Returns:
            Path to the archived file, or None if no messages
        """
        with self._lock:
            # Flush any pending writes
            self._flush_to_disk()

            messages = self.get_session_messages()
            if not messages:
                return None

            # Create archive filename
            archive_dir = self._agent_dir / "archives"
            archive_dir.mkdir(parents=True, exist_ok=True)

            archive_file = archive_dir / f"{self._session_id}.md"

            # Build markdown content
            lines = [
                "---",
                f"date: {datetime.now().strftime('%Y-%m-%d')}",
                f"session_id: {self._session_id}",
                f"agent: {self.agent}",
                f"message_count: {len(messages)}",
                "---",
                "",
                f"# Session: {self._session_id}",
                "",
            ]

            if summary:
                lines.extend([
                    "## Summary",
                    summary,
                    "",
                ])

            lines.append("## Conversation")
            lines.append("")

            for entry in messages:
                timestamp = entry.timestamp.split("T")[1].split(".")[0]
                role_icon = "[SOUL]" if entry.role == "user" else "??"
                lines.append(f"### {role_icon} {entry.role.title()} ({timestamp})")
                lines.append("")
                lines.append(entry.content)
                lines.append("")

            # Write archive
            archive_file.write_text("\n".join(lines), encoding='utf-8')

            logger.info(f"[Marrow] Archived session to {archive_file}")
            return archive_file

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


# -------------------------------------------------------------------------------
# GLOBAL LOGGERS
# -------------------------------------------------------------------------------

# Singleton loggers for each agent
_loggers: dict = {}
_loggers_lock = threading.Lock()


def get_logger(agent: str) -> ConversationLogger:
    """
    Get the conversation logger for an agent (singleton).

    Usage:
        from haloscorn.scornspine.marrow import get_logger

        logger = get_logger("vera")
        logger.log_user("Hello!")
    """
    with _loggers_lock:
        if agent not in _loggers:
            _loggers[agent] = ConversationLogger(agent)
        return _loggers[agent]


# -------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------

def main():
    """CLI for testing conversation logging."""
    import argparse

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(description="ScornSpine Marrow - Conversation Logger")
    parser.add_argument("agent", choices=["vera", "jonah", "halo"], help="Agent name")
    parser.add_argument("--test", action="store_true", help="Run test logging")
    parser.add_argument("--recent", type=int, help="Show N recent messages")
    parser.add_argument("--search", type=str, help="Search messages")
    parser.add_argument("--archive", action="store_true", help="Archive current session")

    args = parser.parse_args()

    conv_logger = get_logger(args.agent)

    if args.test:
        print(f"[NOTE] Testing conversation logger for {args.agent}...")
        conv_logger.log_user("This is a test user message")
        conv_logger.log_assistant("This is a test assistant response")
        conv_logger.log_user("Another test message about particle counts")
        conv_logger.flush()
        print("[OK] Test messages logged")

    if args.recent:
        print(f"\n?? Last {args.recent} messages:")
        for entry in conv_logger.get_recent(args.recent):
            icon = "[SOUL]" if entry.role == "user" else "??"
            print(f"  {icon} [{entry.timestamp}] {entry.content[:100]}...")

    if args.search:
        print(f"\n?? Searching for: {args.search}")
        matches = conv_logger.search_recent(args.search)
        for entry in matches:
            icon = "[SOUL]" if entry.role == "user" else "??"
            print(f"  {icon} [{entry.timestamp}] {entry.content[:100]}...")

    if args.archive:
        path = conv_logger.archive_session("Test archive")
        if path:
            print(f"[OK] Archived to: {path}")
        else:
            print("[FAIL] No messages to archive")


if __name__ == "__main__":
    main()
