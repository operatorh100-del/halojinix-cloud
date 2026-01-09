"""
ScornSpine Write-Ahead Log (WAL) - RT17200
Durability layer for crash-safe index operations.

Based on JONAH RT17100 research:
- Log operations BEFORE executing them
- Replay unapplied entries on crash recovery
- Periodic compaction to prevent unbounded growth
"""

import sqlite3
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class WALOperation(Enum):
    """Types of WAL operations."""
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    COMPACT = "compact"


@dataclass
class WALEntry:
    """A single WAL entry."""
    id: int
    timestamp: str
    operation: str
    doc_id: str
    payload: Dict[str, Any]
    checksum: str
    applied: bool


class SpineWAL:
    """
    Write-Ahead Log for ScornSpine.
    
    Guarantees:
    - Atomicity: Operations complete fully or not at all
    - Durability: Committed changes survive crashes
    - Recovery: Replay log to restore consistent state
    
    Usage:
        wal = SpineWAL()
        
        # Before modifying index:
        entry_id = wal.log_operation("add", "doc/path.md", {"content": "..."})
        
        # Apply the change to index
        index.add(vectors)
        
        # Mark as applied
        wal.mark_applied(entry_id)
    """
    
    DB_VERSION = 1
    
    def __init__(self, wal_path: Optional[Path] = None):
        self.wal_path = wal_path or Path("F:/primewave-engine/data/spine-wal.db")
        self.wal_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.wal_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        self._init_schema()
        print(f"[SpineWAL] Initialized: {self.wal_path}")
    
    def _init_schema(self):
        """Create tables if not exist."""
        cursor = self.conn.cursor()
        
        # WAL entries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wal_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                operation TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                payload TEXT,
                checksum TEXT NOT NULL,
                applied INTEGER DEFAULT 0,
                applied_at TEXT
            )
        """)
        
        # Indexes for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_wal_applied ON wal_entries(applied)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_wal_doc ON wal_entries(doc_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_wal_timestamp ON wal_entries(timestamp)")
        
        # Schema version
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wal_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # Check/set version
        cursor.execute("SELECT value FROM wal_meta WHERE key = 'version'")
        row = cursor.fetchone()
        if not row:
            cursor.execute("INSERT INTO wal_meta (key, value) VALUES ('version', ?)", 
                          (str(self.DB_VERSION),))
        
        self.conn.commit()
    
    def _compute_checksum(self, operation: str, doc_id: str, payload: Dict) -> str:
        """Compute checksum for entry verification."""
        data = f"{operation}:{doc_id}:{json.dumps(payload, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def log_operation(
        self,
        operation: str,
        doc_id: str,
        payload: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Log operation BEFORE executing it.
        
        Args:
            operation: One of 'add', 'update', 'delete', 'compact'
            doc_id: Document path/identifier
            payload: Additional data (content hash, chunk info, etc.)
            
        Returns:
            Entry ID for marking as applied
        """
        if payload is None:
            payload = {}
        
        timestamp = datetime.now(timezone.utc).isoformat()
        checksum = self._compute_checksum(operation, doc_id, payload)
        payload_json = json.dumps(payload, sort_keys=True)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO wal_entries (timestamp, operation, doc_id, payload, checksum)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, operation, doc_id, payload_json, checksum))
        
        self.conn.commit()
        entry_id = cursor.lastrowid
        
        print(f"[SpineWAL] Logged: {operation} {doc_id} (id={entry_id})")
        return entry_id
    
    def mark_applied(self, entry_id: int) -> bool:
        """
        Mark WAL entry as successfully applied.
        Call this AFTER the operation succeeds.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE wal_entries 
            SET applied = 1, applied_at = ?
            WHERE id = ?
        """, (datetime.now(timezone.utc).isoformat(), entry_id))
        
        self.conn.commit()
        return cursor.rowcount > 0
    
    def get_unapplied(self) -> List[WALEntry]:
        """
        Get all unapplied entries for crash recovery.
        Ordered by ID (chronological).
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, operation, doc_id, payload, checksum, applied
            FROM wal_entries
            WHERE applied = 0
            ORDER BY id ASC
        """)
        
        entries = []
        for row in cursor.fetchall():
            entries.append(WALEntry(
                id=row['id'],
                timestamp=row['timestamp'],
                operation=row['operation'],
                doc_id=row['doc_id'],
                payload=json.loads(row['payload']) if row['payload'] else {},
                checksum=row['checksum'],
                applied=bool(row['applied']),
            ))
        
        return entries
    
    def verify_entry(self, entry: WALEntry) -> bool:
        """Verify entry checksum hasn't been corrupted."""
        expected = self._compute_checksum(entry.operation, entry.doc_id, entry.payload)
        return entry.checksum == expected
    
    def get_entry(self, entry_id: int) -> Optional[WALEntry]:
        """Get a specific WAL entry."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, operation, doc_id, payload, checksum, applied
            FROM wal_entries WHERE id = ?
        """, (entry_id,))
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return WALEntry(
            id=row['id'],
            timestamp=row['timestamp'],
            operation=row['operation'],
            doc_id=row['doc_id'],
            payload=json.loads(row['payload']) if row['payload'] else {},
            checksum=row['checksum'],
            applied=bool(row['applied']),
        )
    
    def get_entries_for_doc(self, doc_id: str) -> List[WALEntry]:
        """Get all entries for a document."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, operation, doc_id, payload, checksum, applied
            FROM wal_entries
            WHERE doc_id = ?
            ORDER BY id DESC
        """, (doc_id,))
        
        return [
            WALEntry(
                id=row['id'],
                timestamp=row['timestamp'],
                operation=row['operation'],
                doc_id=row['doc_id'],
                payload=json.loads(row['payload']) if row['payload'] else {},
                checksum=row['checksum'],
                applied=bool(row['applied']),
            )
            for row in cursor.fetchall()
        ]
    
    def compact(self, keep_last_n: int = 1000) -> Dict[str, int]:
        """
        Remove old applied entries to prevent unbounded growth.
        
        Args:
            keep_last_n: Number of recent applied entries to keep
            
        Returns:
            Stats about compaction
        """
        cursor = self.conn.cursor()
        
        # Count before
        cursor.execute("SELECT COUNT(*) as total FROM wal_entries")
        before = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as applied FROM wal_entries WHERE applied = 1")
        applied_count = cursor.fetchone()['applied']
        
        if applied_count <= keep_last_n:
            return {"before": before, "after": before, "removed": 0}
        
        # Delete old applied entries, keeping the most recent
        cursor.execute("""
            DELETE FROM wal_entries
            WHERE applied = 1
            AND id NOT IN (
                SELECT id FROM wal_entries
                WHERE applied = 1
                ORDER BY id DESC
                LIMIT ?
            )
        """, (keep_last_n,))
        
        removed = cursor.rowcount
        self.conn.commit()
        
        # VACUUM to reclaim space
        self.conn.execute("VACUUM")
        
        print(f"[SpineWAL] Compacted: removed {removed} entries, kept {keep_last_n}")
        
        return {
            "before": before,
            "after": before - removed,
            "removed": removed,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get WAL statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM wal_entries")
        total = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as unapplied FROM wal_entries WHERE applied = 0")
        unapplied = cursor.fetchone()['unapplied']
        
        cursor.execute("""
            SELECT operation, COUNT(*) as count 
            FROM wal_entries 
            GROUP BY operation
        """)
        by_operation = {row['operation']: row['count'] for row in cursor.fetchall()}
        
        cursor.execute("SELECT MIN(timestamp) as oldest, MAX(timestamp) as newest FROM wal_entries")
        row = cursor.fetchone()
        
        return {
            "total_entries": total,
            "unapplied": unapplied,
            "applied": total - unapplied,
            "by_operation": by_operation,
            "oldest_entry": row['oldest'],
            "newest_entry": row['newest'],
            "wal_path": str(self.wal_path),
        }
    
    def clear_all(self) -> int:
        """Clear all WAL entries. USE WITH CAUTION."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM wal_entries")
        count = cursor.rowcount
        self.conn.commit()
        print(f"[SpineWAL] Cleared {count} entries")
        return count
    
    def close(self):
        """Close database connection."""
        self.conn.close()


if __name__ == "__main__":
    wal = SpineWAL()
    
    # Test logging
    entry_id = wal.log_operation(
        "add",
        "docs/test.md",
        {"content_hash": "abc123", "chunks": 3}
    )
    print(f"Logged entry: {entry_id}")
    
    # Check unapplied
    unapplied = wal.get_unapplied()
    print(f"Unapplied entries: {len(unapplied)}")
    
    # Mark applied
    wal.mark_applied(entry_id)
    
    # Stats
    print(f"Stats: {wal.get_stats()}")
    
    wal.close()
