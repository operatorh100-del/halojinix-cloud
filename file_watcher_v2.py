"""
ScornSpine File Watcher v2 - RT16300
Event-driven file watching using watchdog library.

Replaces polling with native OS file system events.
Based on JONAH RT16100 research: watchdog + 2s debounce = 1-3s latency.
"""

import os
import sys
import time
import hashlib
import json
import threading
import requests
from pathlib import Path
from typing import Set, Dict, Optional
from datetime import datetime

# Config
SPINE_URL = "http://127.0.0.1:7782"
WORKSPACE = Path("F:/primewave-engine")
WATCH_DIRS = [
    "docs",
    "agent-skills",
    "knowledge-base",
    "protocols",
    "agent-handoff",
    ".github/instructions",
]
WATCH_EXTENSIONS = {".md", ".txt", ".json", ".yaml", ".yml"}
IGNORE_PATTERNS = [
    "node_modules",
    ".git",
    "archive",
    "__pycache__",
    ".venv",
]
DEBOUNCE_SECONDS = 2  # Reduced from 5s per JONAH research
HASH_FILE = WORKSPACE / "data" / "spine-file-hashes.json"

# Try to import watchdog, fall back to polling if not available
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("[Watcher] watchdog not installed, falling back to polling mode")
    print("[Watcher] Install with: pip install watchdog")


class SpineEventHandler(FileSystemEventHandler):
    """Handle file system events and trigger Spine reindex."""

    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.pending_files: Set[str] = set()
        self.debounce_timer: Optional[threading.Timer] = None
        self.lock = threading.Lock()

    def _should_process(self, path: str) -> bool:
        """Check if file should be processed."""
        path_obj = Path(path)

        # Check extension
        if path_obj.suffix.lower() not in WATCH_EXTENSIONS:
            return False

        # Check ignore patterns
        path_str = str(path_obj)
        for pattern in IGNORE_PATTERNS:
            if pattern in path_str:
                return False

        return True

    def _schedule_reindex(self):
        """Schedule debounced reindex."""
        with self.lock:
            if self.debounce_timer:
                self.debounce_timer.cancel()
            self.debounce_timer = threading.Timer(DEBOUNCE_SECONDS, self._do_reindex)
            self.debounce_timer.start()

    def _do_reindex(self):
        """Execute the reindex after debounce."""
        with self.lock:
            if self.pending_files:
                files = list(self.pending_files)
                self.pending_files.clear()
                self.debounce_timer = None

        if files:
            self.callback(files)

    def on_modified(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            rel_path = str(Path(event.src_path).relative_to(WORKSPACE))
            print(f"[Watcher] MODIFIED: {rel_path}")
            with self.lock:
                self.pending_files.add(rel_path)
            self._schedule_reindex()

    def on_created(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            rel_path = str(Path(event.src_path).relative_to(WORKSPACE))
            print(f"[Watcher] CREATED: {rel_path}")
            with self.lock:
                self.pending_files.add(rel_path)
            self._schedule_reindex()

    def on_deleted(self, event):
        if event.is_directory:
            return
        if self._should_process(event.src_path):
            rel_path = str(Path(event.src_path).relative_to(WORKSPACE))
            print(f"[Watcher] DELETED: {rel_path}")
            with self.lock:
                self.pending_files.add(rel_path)
            self._schedule_reindex()


class SpineFileWatcherV2:
    """Event-driven file watcher using watchdog."""

    def __init__(self):
        self.file_hashes: Dict[str, str] = {}
        self._load_hashes()
        self.observer: Optional[Observer] = None
        self.stats = {
            "events_received": 0,
            "reindexes_triggered": 0,
            "last_reindex": None,
            "start_time": datetime.utcnow().isoformat(),
        }

    def _load_hashes(self):
        """Load previous file hashes from disk."""
        if HASH_FILE.exists():
            try:
                with open(HASH_FILE, 'r') as f:
                    self.file_hashes = json.load(f)
                print(f"[Watcher] Loaded {len(self.file_hashes)} file hashes")
            except Exception as e:
                print(f"[Watcher] Failed to load hashes: {e}")

    def _save_hashes(self):
        """Save file hashes to disk."""
        HASH_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(HASH_FILE, 'w') as f:
            json.dump(self.file_hashes, f, indent=2)

    def _hash_file(self, path: Path) -> str:
        """Get MD5 hash of file content."""
        try:
            content = path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return ""

    def _on_changes(self, changed_files: list):
        """Callback when files change."""
        self.stats["events_received"] += len(changed_files)

        # Update hashes for changed files
        for rel_path in changed_files:
            full_path = WORKSPACE / rel_path
            if full_path.exists():
                self.file_hashes[rel_path] = self._hash_file(full_path)
            elif rel_path in self.file_hashes:
                del self.file_hashes[rel_path]

        self._save_hashes()
        self._trigger_reindex(changed_files)

    def _trigger_reindex(self, changed_files: list):
        """Trigger Spine to reindex."""
        try:
            print(f"[Watcher] Triggering Spine refresh for {len(changed_files)} files...")
            response = requests.post(f"{SPINE_URL}/refresh", timeout=30)
            if response.status_code == 200:
                result = response.json()
                print(f"[Watcher] Spine refreshed: {result.get('documents', '?')} docs")
                self.stats["reindexes_triggered"] += 1
                self.stats["last_reindex"] = datetime.utcnow().isoformat()
                return True
            else:
                print(f"[Watcher] Spine refresh failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"[Watcher] Spine connection failed: {e}")
            return False

    def run(self):
        """Start the file watcher."""
        if not WATCHDOG_AVAILABLE:
            print("[Watcher] ERROR: watchdog not available, cannot run event-driven mode")
            print("[Watcher] Use file_watcher.py for polling fallback")
            return

        print("[Watcher] === ScornSpine File Watcher v2 (Event-Driven) ===")
        print(f"[Watcher] Workspace: {WORKSPACE}")
        print(f"[Watcher] Debounce: {DEBOUNCE_SECONDS}s")
        print(f"[Watcher] Extensions: {WATCH_EXTENSIONS}")
        print(f"[Watcher] Watching directories:")

        handler = SpineEventHandler(self._on_changes)
        self.observer = Observer()

        for watch_dir in WATCH_DIRS:
            watch_path = WORKSPACE / watch_dir
            if watch_path.exists():
                self.observer.schedule(handler, str(watch_path), recursive=True)
                print(f"[Watcher]   - {watch_dir}/")
            else:
                print(f"[Watcher]   - {watch_dir}/ (not found, skipping)")

        # Also watch root for AGENTS.md
        self.observer.schedule(handler, str(WORKSPACE), recursive=False)

        self.observer.start()
        print("[Watcher] Event-driven watcher started. Press Ctrl+C to stop.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("[Watcher] Shutting down...")
            self.observer.stop()

        self.observer.join()
        print(f"[Watcher] Stats: {self.stats}")


if __name__ == "__main__":
    watcher = SpineFileWatcherV2()
    watcher.run()
