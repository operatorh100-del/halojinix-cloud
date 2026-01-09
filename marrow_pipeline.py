"""
MARROW Pipeline Integration - RT17500
Wires together all MARROW components into a unified pipeline.

Components:
- file_watcher_v2.py: Event-driven file watching (Phase 1)
- incremental_index.py: FAISS IndexIDMap with soft deletes (Phase 2)
- document_store.py: SQLite document versioning (Phase 2)
- semantic_chunker.py: Language-aware chunking (Phase 2)
- crdts.py: Conflict-free data types (Phase 4)
- truth_registry.py: Cross-agent coherence (Phase 4)
- wal.py: Write-ahead logging (Phase 5)
- snapshots.py: FAISS snapshots (Phase 5)
- recovery.py: Crash recovery (Phase 5)

Architecture (ADR-0090):
```
File Event ? WAL.append ? DocumentStore.check ? Chunker ? Index.add ? Snapshot
                ?                                           ?
           Recovery Point                           TruthRegistry.update
```
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# MARROW components
from .document_store import DocumentStore, detect_language
from .semantic_chunker import SemanticChunker, Chunk
from .incremental_index import IncrementalSpineIndex
from .wal import SpineWAL, WALEntry, WALOperation
from .snapshots import SnapshotManager
from .recovery import CrashRecoveryManager
from .truth_registry import TruthRegistry


@dataclass
class PipelineConfig:
    """MARROW pipeline configuration."""
    workspace: Path = Path("F:/primewave-engine")
    db_path: Path = Path("F:/primewave-engine/data/spine-documents.db")
    index_path: Path = Path("F:/primewave-engine/haloscorn/scornspine/index_incremental")
    wal_path: Path = Path("F:/primewave-engine/data/spine-wal.db")
    snapshot_dir: Path = Path("F:/primewave-engine/data/spine-snapshots")
    truth_db_path: Path = Path("F:/primewave-engine/data/spine-truth.db")
    
    embedding_model: str = "intfloat/multilingual-e5-base"
    use_gpu: bool = True
    auto_snapshot_interval: int = 100  # Operations between snapshots
    
    def __post_init__(self):
        # Ensure all paths are Path objects
        self.workspace = Path(self.workspace)
        self.db_path = Path(self.db_path)
        self.index_path = Path(self.index_path)
        self.wal_path = Path(self.wal_path)
        self.snapshot_dir = Path(self.snapshot_dir)
        self.truth_db_path = Path(self.truth_db_path)


@dataclass
class IndexResult:
    """Result of indexing a file."""
    path: str
    success: bool
    is_update: bool
    chunks_added: int
    error: Optional[str] = None
    wal_id: Optional[int] = None


class MARROWPipeline:
    """
    Unified MARROW indexing pipeline.
    
    Combines all phases into a single cohesive system:
    1. Write operations go through WAL first (durability)
    2. DocumentStore tracks versions and change detection
    3. SemanticChunker breaks documents into meaningful pieces
    4. IncrementalIndex stores embeddings
    5. TruthRegistry tracks coherence across agents
    6. SnapshotManager persists state periodically
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        print("[MARROW] Initializing pipeline...")
        
        # Phase 2: Document tracking (needed by recovery)
        self.doc_store = DocumentStore(db_path=self.config.db_path)
        self.chunker = SemanticChunker()
        
        # Phase 5: Write-ahead logging (needed by recovery)
        self.wal = SpineWAL(wal_path=self.config.wal_path)
        
        # Phase 5: Snapshots (needed by recovery)
        self.snapshots = SnapshotManager(
            snapshot_dir=self.config.snapshot_dir,
            max_snapshots=24
        )
        
        # Phase 2: Incremental indexing
        self.index = IncrementalSpineIndex(
            model_name=self.config.embedding_model,
            index_path=self.config.index_path,
            use_gpu=self.config.use_gpu
        )
        
        # Phase 5: Recovery manager (uses wal, snapshots, doc_store)
        self.recovery = CrashRecoveryManager(
            wal=self.wal,
            snapshots=self.snapshots,
            doc_store=self.doc_store,
            embedder=self.index.embedder,
            dimension=self.index.dim
        )
        
        # Phase 4: Coherence tracking
        self.truth = TruthRegistry(db_path=self.config.truth_db_path)
        
        # Stats
        self._ops_since_snapshot = 0
        self.stats = {
            "files_indexed": 0,
            "files_updated": 0,
            "files_deleted": 0,
            "chunks_total": 0,
            "errors": 0,
            "last_snapshot": None,
            "pipeline_started": datetime.utcnow().isoformat(),
        }
        
        # Attempt recovery on startup
        self._attempt_recovery()
        
        print("[MARROW] Pipeline ready")
    
    def _attempt_recovery(self):
        """Try to recover from crash if needed."""
        try:
            # Check if there are unapplied WAL entries
            unapplied = self.wal.get_unapplied()
            if unapplied:
                print(f"[MARROW] Found {len(unapplied)} unapplied WAL entries, attempting recovery...")
                result = self.recovery.recover()
                if result.success:
                    print(f"[MARROW] Recovery successful: method={result.method}, vectors={result.vector_count}")
                    # Update index if recovery produced one
                    if result.index is not None:
                        self.index.index = result.index
                else:
                    print(f"[MARROW] Recovery had issues: {result.errors}")
            else:
                print("[MARROW] No unapplied WAL entries, starting fresh")
        except Exception as e:
            print(f"[MARROW] Recovery check failed (fresh start): {e}")
    
    def index_file(self, file_path: str, force: bool = False) -> IndexResult:
        """
        Index a single file through the full pipeline.
        
        Args:
            file_path: Relative path from workspace root
            force: Force reindex even if unchanged
            
        Returns:
            IndexResult with success status and details
        """
        full_path = self.config.workspace / file_path
        
        # Step 1: Read file
        try:
            content = full_path.read_text(encoding='utf-8')
        except Exception as e:
            return IndexResult(
                path=file_path,
                success=False,
                is_update=False,
                chunks_added=0,
                error=f"Read failed: {e}"
            )
        
        # Step 2: Check if changed (skip if unchanged and not forced)
        changed, existing_doc = self.doc_store.has_changed(file_path, content)
        if not changed and not force:
            return IndexResult(
                path=file_path,
                success=True,
                is_update=False,
                chunks_added=0,
                error="unchanged"
            )
        
        # Step 3: Write to WAL FIRST (durability)
        wal_entry_id = self.wal.log_operation(
            operation="update" if existing_doc else "add",
            doc_id=file_path,
            payload={"content_hash": self.doc_store.hash_content(content), "file_size": len(content), "language": detect_language(file_path)}
        )
        
        try:
            # Step 4: Soft-delete old chunks if update
            if existing_doc:
                old_vector_ids = self.doc_store.get_vector_ids(existing_doc.id)
                for vid in old_vector_ids:
                    self.index._deleted_ids.add(vid)
                    self.index.stats["soft_deletes"] += 1
                self.doc_store.delete_chunks(existing_doc.id)
            
            # Step 5: Register document
            doc_id, is_update = self.doc_store.register_document(
                path=file_path,
                content=content,
                language=detect_language(file_path),
                file_size=len(content)
            )
            
            # Step 6: Chunk document
            chunks = self.chunker.chunk(content, file_path)
            
            # Step 7: Embed and index each chunk
            chunks_added = 0
            for i, chunk in enumerate(chunks):
                embedding_text = chunk.to_embedding_string(file_path)
                
                # Add to FAISS
                success, msg = self.index.add_document(
                    path=f"{file_path}#chunk{i}",
                    text=embedding_text,
                    force=True  # Already checked change
                )
                
                if success:
                    # Get the vector ID (it's the last added)
                    vector_id = self.index._next_id - 1
                    
                    # Track in DocumentStore
                    self.doc_store.add_chunk(
                        doc_id=doc_id,
                        chunk_index=i,
                        vector_id=vector_id,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        chunk_type=chunk.chunk_type,
                        content_preview=chunk.content[:200],
                        embedding_model=self.config.embedding_model
                    )
                    chunks_added += 1
            
            # Step 8: Mark WAL entry as applied
            self.wal.mark_applied(wal_entry_id)
            
            # Step 9: Update TruthRegistry
            self.truth.register(
                entry_id=f"doc:{file_path}",
                entry_type="document",
                content=content[:1000],  # Truncate for hash
                agent="spine",
                content_path=file_path
            )
            
            # Step 10: Maybe snapshot
            self._ops_since_snapshot += 1
            if self._ops_since_snapshot >= self.config.auto_snapshot_interval:
                self._create_snapshot()
            
            # Update stats
            if is_update:
                self.stats["files_updated"] += 1
            else:
                self.stats["files_indexed"] += 1
            self.stats["chunks_total"] += chunks_added
            
            return IndexResult(
                path=file_path,
                success=True,
                is_update=is_update,
                chunks_added=chunks_added,
                wal_id=wal_entry_id
            )
            
        except Exception as e:
            self.stats["errors"] += 1
            return IndexResult(
                path=file_path,
                success=False,
                is_update=bool(existing_doc),
                chunks_added=0,
                error=str(e),
                wal_id=wal_entry_id
            )
    
    def delete_file(self, file_path: str) -> IndexResult:
        """Remove a file from the index."""
        # WAL first
        wal_entry_id = self.wal.log_operation(
            operation="delete",
            doc_id=file_path
        )
        
        try:
            # Get document
            doc = self.doc_store.get_document_by_path(file_path)
            if not doc:
                return IndexResult(
                    path=file_path,
                    success=True,
                    is_update=False,
                    chunks_added=0,
                    error="not found"
                )
            
            # Soft-delete vectors
            vector_ids = self.doc_store.get_vector_ids(doc.id)
            for vid in vector_ids:
                self.index._deleted_ids.add(vid)
            
            # Delete from DocumentStore
            self.doc_store.soft_delete_document(file_path)
            
            # Remove from index mappings
            self.index.remove_document(file_path)
            
            # Mark WAL applied
            self.wal.mark_applied(wal_entry_id)
            
            self.stats["files_deleted"] += 1
            
            return IndexResult(
                path=file_path,
                success=True,
                is_update=False,
                chunks_added=0,
                wal_id=wal_entry_id
            )
            
        except Exception as e:
            self.stats["errors"] += 1
            return IndexResult(
                path=file_path,
                success=False,
                is_update=False,
                chunks_added=0,
                error=str(e),
                wal_id=wal_entry_id
            )
    
    def query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Query the index.
        
        Returns results with document metadata from DocumentStore.
        """
        results = self.index.query(query_text, top_k=top_k)
        
        # Enrich with DocumentStore metadata
        for result in results:
            # Extract base path from chunk path
            path = result.get("path", "").split("#chunk")[0]
            doc = self.doc_store.get_document_by_path(path)
            if doc:
                result["version"] = doc.version
                result["last_indexed"] = str(doc.last_indexed)
        
        return results
    
    def _create_snapshot(self):
        """Create a snapshot of current state."""
        try:
            # Save index first
            self.index._save()
            
            # Create snapshot
            snapshot_path = self.snapshots.create_snapshot(
                self.index.index,
                document_count=self.doc_store.get_stats()["total_documents"],
                wal_position=0,  # TODO: Track WAL position
                agent="marrow"
            )
            
            # Compact WAL
            self.wal.compact()
            
            self._ops_since_snapshot = 0
            self.stats["last_snapshot"] = datetime.utcnow().isoformat()
            
            print(f"[MARROW] Snapshot created: {snapshot_path}")
            
        except Exception as e:
            print(f"[MARROW] Snapshot failed: {e}")
    
    def compact(self) -> Dict[str, Any]:
        """
        Run full compaction:
        - Remove soft-deleted vectors from FAISS
        - Clean up DocumentStore
        - Compact WAL
        - Create fresh snapshot
        """
        print("[MARROW] Starting compaction...")
        started = datetime.utcnow()
        
        docs_before = self.doc_store.get_stats()["total_documents"]
        vectors_before = self.index.index.ntotal
        
        # Compact FAISS index
        compact_result = self.index.compact()
        
        # Clean DocumentStore
        deleted_docs = self.doc_store.cleanup_deleted()
        
        # Compact WAL
        self.wal.compact()
        
        # Create snapshot
        self._create_snapshot()
        
        completed = datetime.utcnow()
        
        # Log compaction
        self.doc_store.log_compaction(
            started_at=started,
            completed_at=completed,
            docs_before=docs_before,
            docs_after=self.doc_store.get_stats()["total_documents"],
            vectors_removed=compact_result.get("removed", 0)
        )
        
        result = {
            "duration_seconds": (completed - started).total_seconds(),
            "vectors_removed": compact_result.get("removed", 0),
            "vectors_remaining": self.index.index.ntotal,
            "docs_cleaned": deleted_docs,
        }
        
        print(f"[MARROW] Compaction complete: {result}")
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return {
            "pipeline": self.stats,
            "index": self.index.get_stats(),
            "document_store": self.doc_store.get_stats(),
            "wal": {
                "unapplied": len(self.wal.get_unapplied()),
            },
            "snapshots": {
                "count": len(self.snapshots.get_snapshots()),
                "latest": (lambda s: s[1] if s else None)(self.snapshots.get_latest_snapshot()),
            },
            "truth_registry": self.truth.get_stats(),
        }
    
    def shutdown(self):
        """Graceful shutdown with final snapshot."""
        print("[MARROW] Shutting down...")
        
        # Save everything
        self.index._save()
        self._create_snapshot()
        self.doc_store.close()
        
        print("[MARROW] Shutdown complete")


# Integration with file_watcher_v2
def create_watcher_callback(pipeline: MARROWPipeline):
    """
    Create a callback for SpineFileWatcherV2 that uses MARROW pipeline.
    
    Usage in file_watcher_v2.py:
        pipeline = MARROWPipeline()
        callback = create_watcher_callback(pipeline)
        handler = SpineEventHandler(callback)
    """
    def on_changes(changed_files: List[str]):
        """Process batch of changed files."""
        print(f"[MARROW] Processing {len(changed_files)} file changes...")
        
        results = []
        for file_path in changed_files:
            full_path = pipeline.config.workspace / file_path
            
            if full_path.exists():
                result = pipeline.index_file(file_path)
            else:
                result = pipeline.delete_file(file_path)
            
            results.append(result)
            
            if result.success:
                action = "updated" if result.is_update else "indexed"
                if result.error == "unchanged":
                    action = "unchanged"
                print(f"[MARROW] {file_path}: {action} ({result.chunks_added} chunks)")
            else:
                print(f"[MARROW] {file_path}: FAILED - {result.error}")
        
        # Summary
        success = sum(1 for r in results if r.success)
        failed = len(results) - success
        print(f"[MARROW] Batch complete: {success} success, {failed} failed")
    
    return on_changes


if __name__ == "__main__":
    # Test pipeline
    print("=== MARROW Pipeline Test ===")
    
    pipeline = MARROWPipeline()
    
    # Index a test file
    result = pipeline.index_file("agent-skills/shared/S001-SKILL-TEMPLATE.md")
    print(f"Index result: {result}")
    
    # Query
    results = pipeline.query("skill template", top_k=3)
    print(f"Query results: {len(results)}")
    for r in results:
        print(f"  {r['path']}: {r['score']:.3f}")
    
    # Stats
    print(f"\nPipeline stats:")
    for key, value in pipeline.get_stats().items():
        print(f"  {key}: {value}")
    
    # Shutdown
    pipeline.shutdown()
