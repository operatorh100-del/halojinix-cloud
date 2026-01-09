"""
ScornSpine Truth Registry - RT17100
Central registry for cross-agent memory coherence.

Based on JONAH RT16900 research:
- SQLite storage for ACID transactions
- Vector clocks for conflict detection
- Automatic roundtable triggering on conflicts
- History tracking for audit trail
"""

import sqlite3
import json
import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from .crdts import VectorClock, ClockComparison


class TruthStatus(Enum):
    """Status of a truth entry."""
    PROPOSED = "proposed"  # In flux, can be modified
    ACTIVE = "active"      # Live, but can be updated
    FROZEN = "frozen"      # Immutable (accepted ADR)
    SUPERSEDED = "superseded"  # Replaced by newer version


class TruthType(Enum):
    """Type of truth entry."""
    ADR = "adr"
    SKILL = "skill"
    PROTOCOL = "protocol"
    DECISION = "decision"
    CONFIG = "config"


class RoundtableStatus(Enum):
    """Status of a roundtable request."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    TIMEOUT = "timeout"


@dataclass
class TruthEntry:
    """A truth registry entry."""
    id: str
    type: str
    status: str
    content_hash: str
    vector_clock: Dict[str, int]
    last_modified_by: str
    last_modified_at: str
    content_path: Optional[str] = None


@dataclass
class HistoryEntry:
    """History/audit entry for a truth."""
    entry_id: str
    timestamp: str
    agent: str
    action: str
    content_hash: str
    vector_clock: Dict[str, int]


@dataclass
class RoundtableRequest:
    """Automatic roundtable request for conflict resolution."""
    id: str
    type: str
    priority: str
    created_at: str
    timeout_at: str
    subject_id: str
    subject_type: str
    existing_clock: Dict[str, int]
    existing_hash: str
    incoming_clock: Dict[str, int]
    incoming_hash: str
    incoming_agent: str
    status: str = "pending"
    resolution: Optional[str] = None
    resolved_at: Optional[str] = None


class TruthRegistry:
    """
    Central registry for tracking ground truth across agents.
    
    Responsibilities:
    1. Track what is "true" (frozen decisions, accepted ADRs)
    2. Detect conflicts using vector clocks
    3. Trigger roundtables automatically on conflicts
    4. Maintain audit trail
    """
    
    DB_VERSION = 1
    ROUNDTABLE_TIMEOUT_MINUTES = 30
    
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or Path("F:/primewave-engine/data/truth-registry.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        self._init_schema()
        print(f"[TruthRegistry] Initialized: {self.db_path}")
    
    def _init_schema(self):
        """Create tables if not exist."""
        cursor = self.conn.cursor()
        
        # Truth entries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS truth_entries (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                vector_clock TEXT NOT NULL,
                last_modified_by TEXT NOT NULL,
                last_modified_at TIMESTAMP NOT NULL,
                content_path TEXT
            )
        """)
        
        # History/audit table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS truth_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                agent TEXT NOT NULL,
                action TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                vector_clock TEXT NOT NULL,
                FOREIGN KEY (entry_id) REFERENCES truth_entries(id)
            )
        """)
        
        # Roundtable requests
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS roundtable_requests (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                priority TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                timeout_at TIMESTAMP NOT NULL,
                subject_id TEXT NOT NULL,
                subject_type TEXT NOT NULL,
                existing_clock TEXT NOT NULL,
                existing_hash TEXT NOT NULL,
                incoming_clock TEXT NOT NULL,
                incoming_hash TEXT NOT NULL,
                incoming_agent TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                resolution TEXT,
                resolved_at TIMESTAMP
            )
        """)
        
        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_truth_type ON truth_entries(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_truth_status ON truth_entries(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_entry ON truth_history(entry_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_roundtable_status ON roundtable_requests(status)")
        
        self.conn.commit()
    
    @staticmethod
    def hash_content(content: str) -> str:
        """Generate content hash."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def register(
        self,
        entry_id: str,
        entry_type: str,
        content: str,
        agent: str,
        clock: Optional[VectorClock] = None,
        content_path: Optional[str] = None,
        status: str = "active"
    ) -> Tuple[bool, Optional[RoundtableRequest]]:
        """
        Register or update a truth entry.
        
        Returns:
            (success, roundtable_request if conflict detected)
        """
        cursor = self.conn.cursor()
        content_hash = self.hash_content(content)
        now = datetime.utcnow().isoformat()
        
        if clock is None:
            clock = VectorClock()
        clock.increment(agent)
        
        # Check for existing entry
        cursor.execute("SELECT * FROM truth_entries WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        
        if row:
            existing = TruthEntry(
                id=row['id'],
                type=row['type'],
                status=row['status'],
                content_hash=row['content_hash'],
                vector_clock=json.loads(row['vector_clock']),
                last_modified_by=row['last_modified_by'],
                last_modified_at=row['last_modified_at'],
                content_path=row['content_path'],
            )
            
            # Check if frozen
            if existing.status == TruthStatus.FROZEN.value:
                print(f"[TruthRegistry] Cannot modify frozen entry: {entry_id}")
                return (False, None)
            
            # Check for conflict using vector clocks
            existing_clock = VectorClock.from_dict(existing.vector_clock)
            comparison = existing_clock.compare(clock)
            
            if comparison == ClockComparison.CONCURRENT:
                # CONFLICT DETECTED!
                # Check if content is actually different
                if existing.content_hash == content_hash:
                    # Same content, just merge clocks
                    clock.merge(existing_clock)
                else:
                    # True conflict - trigger roundtable
                    roundtable = self._create_roundtable_request(
                        existing, content_hash, clock, agent
                    )
                    return (False, roundtable)
            
            elif comparison == ClockComparison.B_BEFORE_A:
                # Incoming is stale, reject
                print(f"[TruthRegistry] Stale update rejected for {entry_id}")
                return (False, None)
            
            # Valid update - merge clocks
            clock.merge(existing_clock)
            
            # Update entry
            cursor.execute("""
                UPDATE truth_entries 
                SET content_hash = ?, vector_clock = ?, last_modified_by = ?, 
                    last_modified_at = ?, status = ?, content_path = ?
                WHERE id = ?
            """, (content_hash, json.dumps(clock.to_dict()), agent, now, 
                  status, content_path, entry_id))
            
            # Add history entry
            self._add_history(entry_id, agent, "modified", content_hash, clock)
            
            self.conn.commit()
            return (True, None)
        
        else:
            # New entry
            cursor.execute("""
                INSERT INTO truth_entries 
                (id, type, status, content_hash, vector_clock, last_modified_by, last_modified_at, content_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (entry_id, entry_type, status, content_hash, 
                  json.dumps(clock.to_dict()), agent, now, content_path))
            
            # Add history entry
            self._add_history(entry_id, agent, "created", content_hash, clock)
            
            self.conn.commit()
            return (True, None)
    
    def _add_history(
        self,
        entry_id: str,
        agent: str,
        action: str,
        content_hash: str,
        clock: VectorClock
    ):
        """Add history entry."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO truth_history (entry_id, timestamp, agent, action, content_hash, vector_clock)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (entry_id, datetime.utcnow().isoformat(), agent, action, 
              content_hash, json.dumps(clock.to_dict())))
    
    def _create_roundtable_request(
        self,
        existing: TruthEntry,
        incoming_hash: str,
        incoming_clock: VectorClock,
        incoming_agent: str
    ) -> RoundtableRequest:
        """Create a roundtable request for conflict resolution."""
        cursor = self.conn.cursor()
        now = datetime.utcnow()
        timeout = now + timedelta(minutes=self.ROUNDTABLE_TIMEOUT_MINUTES)
        
        # Generate unique ID
        request_id = f"RT{now.strftime('%Y%m%d%H%M%S')}-{existing.id}"
        
        request = RoundtableRequest(
            id=request_id,
            type="truth_conflict",
            priority="high" if existing.type in ["adr", "protocol"] else "medium",
            created_at=now.isoformat(),
            timeout_at=timeout.isoformat(),
            subject_id=existing.id,
            subject_type=existing.type,
            existing_clock=existing.vector_clock,
            existing_hash=existing.content_hash,
            incoming_clock=incoming_clock.to_dict(),
            incoming_hash=incoming_hash,
            incoming_agent=incoming_agent,
        )
        
        cursor.execute("""
            INSERT INTO roundtable_requests
            (id, type, priority, created_at, timeout_at, subject_id, subject_type,
             existing_clock, existing_hash, incoming_clock, incoming_hash, incoming_agent, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')
        """, (
            request.id, request.type, request.priority, request.created_at,
            request.timeout_at, request.subject_id, request.subject_type,
            json.dumps(request.existing_clock), request.existing_hash,
            json.dumps(request.incoming_clock), request.incoming_hash,
            request.incoming_agent
        ))
        
        self.conn.commit()
        
        print(f"[TruthRegistry] CONFLICT DETECTED: {existing.id}")
        print(f"[TruthRegistry] Roundtable request created: {request.id}")
        
        # Notify via Signal (if available)
        self._notify_conflict(request)
        
        return request
    
    def _notify_conflict(self, request: RoundtableRequest):
        """Post conflict notification to Signal bulletin."""
        try:
            import requests
            requests.post(
                "http://127.0.0.1:7778/api/bulletin",
                json={
                    "author": "TRUTH-REGISTRY",
                    "content": f"CONFLICT DETECTED: {request.subject_id} ({request.subject_type}). "
                               f"Roundtable {request.id} created. "
                               f"Timeout: {request.timeout_at}. "
                               f"Conflicting agent: {request.incoming_agent}",
                    "topic": "conflict",
                },
                timeout=5
            )
        except Exception as e:
            print(f"[TruthRegistry] Warning: Could not notify Signal: {e}")
    
    def freeze(self, entry_id: str, agent: str) -> bool:
        """
        Freeze an entry (make immutable).
        Only proposed/active entries can be frozen.
        """
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT * FROM truth_entries WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        
        if not row:
            return False
        
        if row['status'] == TruthStatus.FROZEN.value:
            return True  # Already frozen
        
        if row['status'] not in [TruthStatus.PROPOSED.value, TruthStatus.ACTIVE.value]:
            return False
        
        clock = VectorClock.from_dict(json.loads(row['vector_clock']))
        clock.increment(agent)
        
        cursor.execute("""
            UPDATE truth_entries 
            SET status = ?, vector_clock = ?, last_modified_by = ?, last_modified_at = ?
            WHERE id = ?
        """, (TruthStatus.FROZEN.value, json.dumps(clock.to_dict()), 
              agent, datetime.utcnow().isoformat(), entry_id))
        
        self._add_history(entry_id, agent, "frozen", row['content_hash'], clock)
        
        self.conn.commit()
        print(f"[TruthRegistry] Entry frozen: {entry_id}")
        return True
    
    def supersede(self, entry_id: str, successor_id: str, agent: str) -> bool:
        """Mark an entry as superseded by another."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT * FROM truth_entries WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        
        if not row:
            return False
        
        clock = VectorClock.from_dict(json.loads(row['vector_clock']))
        clock.increment(agent)
        
        cursor.execute("""
            UPDATE truth_entries 
            SET status = ?, vector_clock = ?, last_modified_by = ?, last_modified_at = ?
            WHERE id = ?
        """, (TruthStatus.SUPERSEDED.value, json.dumps(clock.to_dict()),
              agent, datetime.utcnow().isoformat(), entry_id))
        
        self._add_history(entry_id, agent, f"superseded by {successor_id}", 
                          row['content_hash'], clock)
        
        self.conn.commit()
        return True
    
    def get(self, entry_id: str) -> Optional[TruthEntry]:
        """Get a truth entry by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM truth_entries WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return TruthEntry(
            id=row['id'],
            type=row['type'],
            status=row['status'],
            content_hash=row['content_hash'],
            vector_clock=json.loads(row['vector_clock']),
            last_modified_by=row['last_modified_by'],
            last_modified_at=row['last_modified_at'],
            content_path=row['content_path'],
        )
    
    def get_by_type(self, entry_type: str, status: Optional[str] = None) -> List[TruthEntry]:
        """Get all entries of a type."""
        cursor = self.conn.cursor()
        
        if status:
            cursor.execute(
                "SELECT * FROM truth_entries WHERE type = ? AND status = ?",
                (entry_type, status)
            )
        else:
            cursor.execute(
                "SELECT * FROM truth_entries WHERE type = ?",
                (entry_type,)
            )
        
        return [
            TruthEntry(
                id=row['id'],
                type=row['type'],
                status=row['status'],
                content_hash=row['content_hash'],
                vector_clock=json.loads(row['vector_clock']),
                last_modified_by=row['last_modified_by'],
                last_modified_at=row['last_modified_at'],
                content_path=row['content_path'],
            )
            for row in cursor.fetchall()
        ]
    
    def get_history(self, entry_id: str) -> List[HistoryEntry]:
        """Get history for an entry."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM truth_history WHERE entry_id = ? ORDER BY timestamp DESC",
            (entry_id,)
        )
        
        return [
            HistoryEntry(
                entry_id=row['entry_id'],
                timestamp=row['timestamp'],
                agent=row['agent'],
                action=row['action'],
                content_hash=row['content_hash'],
                vector_clock=json.loads(row['vector_clock']),
            )
            for row in cursor.fetchall()
        ]
    
    def get_pending_roundtables(self) -> List[RoundtableRequest]:
        """Get all pending roundtable requests."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM roundtable_requests WHERE status = 'pending' ORDER BY created_at"
        )
        
        return [self._row_to_roundtable(row) for row in cursor.fetchall()]
    
    def _row_to_roundtable(self, row) -> RoundtableRequest:
        """Convert DB row to RoundtableRequest."""
        return RoundtableRequest(
            id=row['id'],
            type=row['type'],
            priority=row['priority'],
            created_at=row['created_at'],
            timeout_at=row['timeout_at'],
            subject_id=row['subject_id'],
            subject_type=row['subject_type'],
            existing_clock=json.loads(row['existing_clock']),
            existing_hash=row['existing_hash'],
            incoming_clock=json.loads(row['incoming_clock']),
            incoming_hash=row['incoming_hash'],
            incoming_agent=row['incoming_agent'],
            status=row['status'],
            resolution=row['resolution'],
            resolved_at=row['resolved_at'],
        )
    
    def resolve_roundtable(
        self,
        request_id: str,
        resolution: str,
        agent: str
    ) -> bool:
        """
        Resolve a roundtable request.
        
        resolution: "accept_existing", "accept_incoming", "merge", "escalate"
        """
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT * FROM roundtable_requests WHERE id = ?", (request_id,))
        row = cursor.fetchone()
        
        if not row or row['status'] != 'pending':
            return False
        
        cursor.execute("""
            UPDATE roundtable_requests 
            SET status = 'resolved', resolution = ?, resolved_at = ?
            WHERE id = ?
        """, (resolution, datetime.utcnow().isoformat(), request_id))
        
        self.conn.commit()
        print(f"[TruthRegistry] Roundtable resolved: {request_id} -> {resolution}")
        return True
    
    def check_timeouts(self) -> List[RoundtableRequest]:
        """Check for timed-out roundtables and escalate them."""
        cursor = self.conn.cursor()
        now = datetime.utcnow().isoformat()
        
        cursor.execute("""
            SELECT * FROM roundtable_requests 
            WHERE status = 'pending' AND timeout_at < ?
        """, (now,))
        
        timed_out = []
        for row in cursor.fetchall():
            request = self._row_to_roundtable(row)
            
            cursor.execute("""
                UPDATE roundtable_requests 
                SET status = 'timeout'
                WHERE id = ?
            """, (request.id,))
            
            print(f"[TruthRegistry] Roundtable timed out: {request.id}")
            timed_out.append(request)
        
        self.conn.commit()
        return timed_out
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) as total FROM truth_entries")
        total = cursor.fetchone()['total']
        
        cursor.execute("""
            SELECT status, COUNT(*) as count 
            FROM truth_entries 
            GROUP BY status
        """)
        by_status = {row['status']: row['count'] for row in cursor.fetchall()}
        
        cursor.execute("""
            SELECT type, COUNT(*) as count 
            FROM truth_entries 
            GROUP BY type
        """)
        by_type = {row['type']: row['count'] for row in cursor.fetchall()}
        
        cursor.execute("SELECT COUNT(*) as total FROM roundtable_requests WHERE status = 'pending'")
        pending_roundtables = cursor.fetchone()['total']
        
        return {
            "total_entries": total,
            "by_status": by_status,
            "by_type": by_type,
            "pending_roundtables": pending_roundtables,
            "db_path": str(self.db_path),
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()


if __name__ == "__main__":
    registry = TruthRegistry()
    
    # Register an ADR
    success, conflict = registry.register(
        entry_id="ADR-0090",
        entry_type="adr",
        content="# ADR-0090: MARROW Architecture\nDecision content here...",
        agent="halo",
        content_path="docs/decisions/ADR-0090.md"
    )
    print(f"Registered ADR-0090: {success}")
    
    # Register a skill
    success, conflict = registry.register(
        entry_id="S145",
        entry_type="skill",
        content="# S145: RAG Single Source of Truth\nSkill content...",
        agent="halo",
        content_path="agent-skills/shared/S145.md"
    )
    print(f"Registered S145: {success}")
    
    # Simulate concurrent modification
    clock1 = VectorClock()
    clock1.increment("jonah")
    
    success, conflict = registry.register(
        entry_id="S145",
        entry_type="skill",
        content="# S145: MODIFIED by JONAH\nDifferent content...",
        agent="jonah",
        clock=clock1
    )
    
    if conflict:
        print(f"Conflict detected! Roundtable: {conflict.id}")
    
    # Get stats
    print(f"\nStats: {registry.get_stats()}")
    
    registry.close()
