"""
ScornSpine File Watcher - RT16000
Real-time file change detection for instant RAG ingestion.

Watches key directories and triggers Spine re-indexing on changes.
"""

import os
import sys
import time
import hashlib
import json
import requests
from pathlib import Path
from typing import Set, Dict, Optional
from datetime import datetime

# Config
SPINE_URL = "http://127.0.0.1:7782"
WORKSPACE = Path("F:/primewave-engine")
WATCH_PATTERNS = [
    "docs/**/*.md",
    "agent-skills/**/*.md",
    "knowledge-base/**/*.md",
    "protocols/**/*.md",
    "docs/decisions/**/*.md",
    "agent-handoff/**/*.md",
    "AGENTS.md",
    ".github/instructions/*.md",
]
IGNORE_PATTERNS = [
    "**/node_modules/**",
    "**/.git/**",
    "**/archive/**",
    "**/__pycache__/**",
]
DEBOUNCE_SECONDS = 5  # Wait for writes to settle
POLL_INTERVAL = 10  # How often to check for changes

class SpineFileWatcher:
    def __init__(self):
        self.file_hashes: Dict[str, str] = {}
        self.pending_changes: Set[str] = set()
        self.last_change_time: Optional[float] = None
        self.hash_file = WORKSPACE / "data" / "spine-file-hashes.json"
        self._load_hashes()

    def _load_hashes(self):
        """Load previous file hashes from disk."""
        if self.hash_file.exists():
            try:
                with open(self.hash_file, 'r') as f:
                    self.file_hashes = json.load(f)
                print(f"[Watcher] Loaded {len(self.file_hashes)} file hashes")
            except Exception as e:
                print(f"[Watcher] Failed to load hashes: {e}")
                self.file_hashes = {}

    def _save_hashes(self):
        """Save file hashes to disk."""
        self.hash_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.hash_file, 'w') as f:
            json.dump(self.file_hashes, f, indent=2)

    def _hash_file(self, path: Path) -> str:
        """Get MD5 hash of file content."""
        try:
            content = path.read_bytes()
            return hashlib.md5(content).hexdigest()
        except Exception:
            return ""

    def _should_ignore(self, path: Path) -> bool:
        """Check if path matches ignore patterns."""
        path_str = str(path)
        for pattern in IGNORE_PATTERNS:
            # Simple glob-like matching
            if "**" in pattern:
                parts = pattern.split("**")
                if all(p in path_str for p in parts if p):
                    return True
            elif pattern in path_str:
                return True
        return False

    def _get_watched_files(self) -> Set[Path]:
        """Get all files matching watch patterns."""
        files = set()
        for pattern in WATCH_PATTERNS:
            if "**" in pattern:
                # Recursive glob
                base, rest = pattern.split("**", 1)
                base_path = WORKSPACE / base.rstrip("/")
                if base_path.exists():
                    for f in base_path.rglob(rest.lstrip("/")):
                        if f.is_file() and not self._should_ignore(f):
                            files.add(f)
            else:
                # Direct path
                full_path = WORKSPACE / pattern
                if full_path.exists() and full_path.is_file():
                    files.add(full_path)
        return files

    def check_for_changes(self) -> Set[str]:
        """Check all watched files for changes. Returns set of changed paths."""
        changed = set()
        current_files = self._get_watched_files()

        for path in current_files:
            rel_path = str(path.relative_to(WORKSPACE))
            current_hash = self._hash_file(path)

            if rel_path not in self.file_hashes:
                # New file
                print(f"[Watcher] NEW: {rel_path}")
                changed.add(rel_path)
                self.file_hashes[rel_path] = current_hash
            elif self.file_hashes[rel_path] != current_hash:
                # Modified file
                print(f"[Watcher] MODIFIED: {rel_path}")
                changed.add(rel_path)
                self.file_hashes[rel_path] = current_hash

        # Check for deleted files
        known_paths = set(self.file_hashes.keys())
        current_rel_paths = {str(p.relative_to(WORKSPACE)) for p in current_files}
        deleted = known_paths - current_rel_paths
        for rel_path in deleted:
            print(f"[Watcher] DELETED: {rel_path}")
            del self.file_hashes[rel_path]
            changed.add(rel_path)

        if changed:
            self._save_hashes()

        return changed

    def trigger_reindex(self, changed_files: Set[str]):
        """Trigger Spine to reindex changed files."""
        try:
            # For now, trigger full refresh since Spine doesn't have incremental
            print(f"[Watcher] Triggering Spine refresh for {len(changed_files)} files...")
            response = requests.post(f"{SPINE_URL}/refresh", timeout=30)
            if response.status_code == 200:
                result = response.json()
                print(f"[Watcher] Spine refreshed: {result.get('documents', '?')} docs")
                return True
            else:
                print(f"[Watcher] Spine refresh failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"[Watcher] Spine connection failed: {e}")
            return False

    def run(self):
        """Main watch loop with debouncing."""
        print(f"[Watcher] Starting file watcher...")
        print(f"[Watcher] Watching: {len(WATCH_PATTERNS)} patterns")
        print(f"[Watcher] Poll interval: {POLL_INTERVAL}s, Debounce: {DEBOUNCE_SECONDS}s")

        # Initial scan
        initial = self.check_for_changes()
        if initial:
            print(f"[Watcher] Initial changes detected: {len(initial)} files")
            self.trigger_reindex(initial)

        while True:
            try:
                time.sleep(POLL_INTERVAL)
                changes = self.check_for_changes()

                if changes:
                    self.pending_changes.update(changes)
                    self.last_change_time = time.time()

                # Debounce: only trigger if we have pending changes and they've settled
                if self.pending_changes and self.last_change_time:
                    elapsed = time.time() - self.last_change_time
                    if elapsed >= DEBOUNCE_SECONDS:
                        print(f"[Watcher] Debounce complete, triggering reindex...")
                        self.trigger_reindex(self.pending_changes)
                        self.pending_changes.clear()
                        self.last_change_time = None

            except KeyboardInterrupt:
                print("[Watcher] Shutting down...")
                break
            except Exception as e:
                print(f"[Watcher] Error: {e}")
                time.sleep(5)


if __name__ == "__main__":
    watcher = SpineFileWatcher()
    watcher.run()
