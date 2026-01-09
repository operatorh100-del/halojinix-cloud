"""
ScornSpine Snapshot Manager - RT17300
Hourly FAISS index snapshots for fast recovery.

Based on JONAH RT17100 research:
- Atomic writes (temp file + rename)
- Metadata alongside each snapshot
- Prune old snapshots (keep 24 hours)
- Integrity validation on load
"""

import os
import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class SnapshotMetadata:
    """Metadata for a snapshot."""
    timestamp: str
    vector_count: int
    dimension: int
    index_type: str
    document_count: int
    wal_position: int  # Last applied WAL entry ID
    content_hash: str
    created_by: str


class SnapshotManager:
    """
    Manages FAISS index snapshots for crash recovery.

    Features:
    - Hourly snapshots with atomic writes
    - Metadata tracking (vector count, WAL position)
    - Automatic pruning of old snapshots
    - Integrity verification

    Usage:
        snapshots = SnapshotManager()

        # Create snapshot
        path = snapshots.create_snapshot(index, metadata)

        # Load latest
        index, metadata = snapshots.load_latest()
    """

    def __init__(
        self,
        snapshot_dir: Optional[Path] = None,
        max_snapshots: int = 24
    ):
        self.snapshot_dir = snapshot_dir or Path("F:/primewave-engine/data/snapshots")
        self.max_snapshots = max_snapshots
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        print(f"[SnapshotManager] Initialized: {self.snapshot_dir} (max={max_snapshots})")

    def create_snapshot(
        self,
        index,
        document_count: int = 0,
        wal_position: int = 0,
        agent: str = "spine"
    ) -> Optional[Path]:
        """
        Create timestamped snapshot of FAISS index.

        Args:
            index: FAISS index to snapshot
            document_count: Number of documents indexed
            wal_position: Last applied WAL entry ID
            agent: Agent creating the snapshot

        Returns:
            Path to created snapshot, or None on failure
        """
        if not FAISS_AVAILABLE:
            print("[SnapshotManager] FAISS not available, cannot create snapshot")
            return None

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")[:20]  # Include milliseconds
        snapshot_name = f"spine-{timestamp}"
        snapshot_path = self.snapshot_dir / f"{snapshot_name}.index"
        temp_path = self.snapshot_dir / f"{snapshot_name}.tmp"
        meta_path = self.snapshot_dir / f"{snapshot_name}.meta.json"

        # Handle collision: if file exists, add suffix
        suffix = 0
        while snapshot_path.exists() or temp_path.exists():
            suffix += 1
            snapshot_name = f"spine-{timestamp}-{suffix}"
            snapshot_path = self.snapshot_dir / f"{snapshot_name}.index"
            temp_path = self.snapshot_dir / f"{snapshot_name}.tmp"
            meta_path = self.snapshot_dir / f"{snapshot_name}.meta.json"

        try:
            # Convert GPU index to CPU if needed
            if hasattr(index, 'index') and faiss.get_num_gpus() > 0:
                # It's a GPU index wrapper
                try:
                    cpu_index = faiss.index_gpu_to_cpu(index)
                except (RuntimeError, AttributeError):
                    cpu_index = index
            else:
                cpu_index = index

            # Write index to temp file first
            faiss.write_index(cpu_index, str(temp_path))

            # Compute checksum of index file
            content_hash = self._compute_file_hash(temp_path)

            # Create metadata
            metadata = SnapshotMetadata(
                timestamp=timestamp,
                vector_count=cpu_index.ntotal,
                dimension=cpu_index.d,
                index_type=type(cpu_index).__name__,
                document_count=document_count,
                wal_position=wal_position,
                content_hash=content_hash,
                created_by=agent,
            )

            # Write metadata
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": metadata.timestamp,
                    "vector_count": metadata.vector_count,
                    "dimension": metadata.dimension,
                    "index_type": metadata.index_type,
                    "document_count": metadata.document_count,
                    "wal_position": metadata.wal_position,
                    "content_hash": metadata.content_hash,
                    "created_by": metadata.created_by,
                    "created_at": datetime.utcnow().isoformat(),
                }, f, indent=2)

            # Atomic rename (temp -> final)
            temp_path.rename(snapshot_path)

            print(f"[SnapshotManager] Created: {snapshot_name} ({metadata.vector_count} vectors)")

            # Prune old snapshots
            self._prune_old_snapshots()

            return snapshot_path

        except Exception as e:
            print(f"[SnapshotManager] Error creating snapshot: {e}")
            # Cleanup temp file if exists
            if temp_path.exists():
                temp_path.unlink()
            return None

    def _compute_file_hash(self, path: Path) -> str:
        """Compute SHA-256 hash of file."""
        hasher = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_snapshots(self) -> List[Tuple[Path, Optional[SnapshotMetadata]]]:
        """Get all snapshots with metadata, sorted newest first."""
        snapshots = []

        for path in sorted(self.snapshot_dir.glob("spine-*.index"), reverse=True):
            meta_path = path.with_suffix(".meta.json")
            metadata = None

            if meta_path.exists():
                try:
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        metadata = SnapshotMetadata(
                            timestamp=data.get("timestamp", ""),
                            vector_count=data.get("vector_count", 0),
                            dimension=data.get("dimension", 0),
                            index_type=data.get("index_type", ""),
                            document_count=data.get("document_count", 0),
                            wal_position=data.get("wal_position", 0),
                            content_hash=data.get("content_hash", ""),
                            created_by=data.get("created_by", ""),
                        )
                except Exception:
                    pass

            snapshots.append((path, metadata))

        return snapshots

    def get_latest_snapshot(self) -> Optional[Tuple[Path, Optional[SnapshotMetadata]]]:
        """Get most recent valid snapshot."""
        for path, metadata in self.get_snapshots():
            if self.validate_snapshot(path):
                return (path, metadata)
        return None

    def validate_snapshot(self, path: Path) -> bool:
        """
        Verify snapshot integrity.

        Checks:
        1. File exists and is readable
        2. FAISS can load it
        3. Checksum matches (if metadata exists)
        """
        if not path.exists():
            return False

        if not FAISS_AVAILABLE:
            # Can't validate without FAISS
            return path.stat().st_size > 0

        try:
            # Try to load index
            index = faiss.read_index(str(path))

            if index.ntotal < 0:
                return False

            # Check checksum if metadata exists
            meta_path = path.with_suffix(".meta.json")
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    data = json.load(f)
                    expected_hash = data.get("content_hash")
                    if expected_hash:
                        actual_hash = self._compute_file_hash(path)
                        if actual_hash != expected_hash:
                            print(f"[SnapshotManager] Checksum mismatch: {path.name}")
                            return False

            return True

        except Exception as e:
            print(f"[SnapshotManager] Validation failed for {path.name}: {e}")
            return False

    def load_snapshot(self, path: Path) -> Optional[Tuple[Any, Optional[SnapshotMetadata]]]:
        """
        Load a snapshot and its metadata.

        Returns:
            (index, metadata) or None on failure
        """
        if not FAISS_AVAILABLE:
            print("[SnapshotManager] FAISS not available, cannot load snapshot")
            return None

        if not self.validate_snapshot(path):
            print(f"[SnapshotManager] Invalid snapshot: {path.name}")
            return None

        try:
            index = faiss.read_index(str(path))

            # Load metadata
            metadata = None
            meta_path = path.with_suffix(".meta.json")
            if meta_path.exists():
                with open(meta_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metadata = SnapshotMetadata(
                        timestamp=data.get("timestamp", ""),
                        vector_count=data.get("vector_count", 0),
                        dimension=data.get("dimension", 0),
                        index_type=data.get("index_type", ""),
                        document_count=data.get("document_count", 0),
                        wal_position=data.get("wal_position", 0),
                        content_hash=data.get("content_hash", ""),
                        created_by=data.get("created_by", ""),
                    )

            print(f"[SnapshotManager] Loaded: {path.name} ({index.ntotal} vectors)")
            return (index, metadata)

        except Exception as e:
            print(f"[SnapshotManager] Error loading {path.name}: {e}")
            return None

    def load_latest(self) -> Optional[Tuple[Any, Optional[SnapshotMetadata]]]:
        """Load the most recent valid snapshot."""
        latest = self.get_latest_snapshot()
        if latest:
            return self.load_snapshot(latest[0])
        return None

    def _prune_old_snapshots(self):
        """Keep only max_snapshots most recent."""
        snapshots = self.get_snapshots()

        if len(snapshots) <= self.max_snapshots:
            return

        # Delete old snapshots
        for path, _ in snapshots[self.max_snapshots:]:
            try:
                path.unlink(missing_ok=True)
                path.with_suffix(".meta.json").unlink(missing_ok=True)
                print(f"[SnapshotManager] Pruned: {path.name}")
            except Exception as e:
                print(f"[SnapshotManager] Error pruning {path.name}: {e}")

    def delete_snapshot(self, path: Path) -> bool:
        """Delete a specific snapshot."""
        try:
            path.unlink(missing_ok=True)
            path.with_suffix(".meta.json").unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get snapshot statistics."""
        snapshots = self.get_snapshots()

        total_size = sum(
            path.stat().st_size for path, _ in snapshots if path.exists()
        )

        latest = None
        oldest = None
        if snapshots:
            latest = snapshots[0][1].timestamp if snapshots[0][1] else None
            oldest = snapshots[-1][1].timestamp if snapshots[-1][1] else None

        return {
            "snapshot_count": len(snapshots),
            "max_snapshots": self.max_snapshots,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "latest": latest,
            "oldest": oldest,
            "snapshot_dir": str(self.snapshot_dir),
        }


if __name__ == "__main__":
    manager = SnapshotManager()

    # List existing snapshots
    snapshots = manager.get_snapshots()
    print(f"Found {len(snapshots)} snapshots")

    for path, meta in snapshots[:5]:
        valid = manager.validate_snapshot(path)
        print(f"  {path.name}: valid={valid}, vectors={meta.vector_count if meta else '?'}")

    # Stats
    print(f"\nStats: {manager.get_stats()}")
