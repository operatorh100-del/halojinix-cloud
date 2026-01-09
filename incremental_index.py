"""
ScornSpine Incremental Index - RT16400
IndexIDMap wrapper for surgical add/remove operations.

Based on JONAH RT16100 research:
- Use IndexIDMap for ID tracking
- Soft deletes for performance (avoid slow remove_ids)
- Track doc_path -> faiss_id mapping
"""

import os
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


class IncrementalSpineIndex:
    """
    Incremental FAISS index with ID tracking.
    Supports surgical add/update/remove without full rebuild.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        index_path: Optional[Path] = None,
        use_gpu: bool = True
    ):
        self.model_name = model_name
        self.index_path = index_path or Path("F:/primewave-engine/haloscorn/scornspine/index_incremental")
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.use_gpu = use_gpu

        # Load embedding model
        import torch
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        print(f"[IncrementalSpine] Loading {model_name} on {device}...")
        self.embedder = SentenceTransformer(model_name, device=device)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        print(f"[IncrementalSpine] Dimension: {self.dim}")

        # ID counter for unique IDs
        self._next_id = 0

        # Mappings
        self._path_to_ids: Dict[str, List[int]] = {}  # doc_path -> [chunk_ids]
        self._id_to_doc: Dict[int, Dict] = {}  # id -> {path, text, hash, deleted}
        self._content_hashes: Dict[str, int] = {}  # hash -> id (for dedup)
        self._deleted_ids: Set[int] = set()  # Soft-deleted IDs

        # Stats
        self.stats = {
            "adds": 0,
            "updates": 0,
            "deletes": 0,
            "soft_deletes": 0,
            "last_modified": None,
        }

        # Initialize FAISS with IDMap
        self._init_index()

        # Load existing data
        self._load()

    def _init_index(self):
        """Initialize FAISS IndexIDMap."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS required for IncrementalSpineIndex")

        # Base index (flat L2)
        base_index = faiss.IndexFlatL2(self.dim)

        # Wrap with IDMap for custom IDs
        self.index = faiss.IndexIDMap(base_index)

        # GPU acceleration
        self.gpu_resources = None
        if self.use_gpu and faiss.get_num_gpus() > 0 and hasattr(faiss, 'StandardGpuResources'):
            self.gpu_resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, self.index)
            print("[IncrementalSpine] Using GPU FAISS")
        else:
            print("[IncrementalSpine] Using CPU FAISS")

    def _load(self):
        """Load persisted state."""
        meta_file = self.index_path / "incremental_meta.json"
        index_file = self.index_path / "incremental.index"

        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self._next_id = data.get("next_id", 0)
            self._path_to_ids = data.get("path_to_ids", {})
            self._id_to_doc = {int(k): v for k, v in data.get("id_to_doc", {}).items()}
            self._content_hashes = data.get("content_hashes", {})
            self._deleted_ids = set(data.get("deleted_ids", []))
            self.stats = data.get("stats", self.stats)

            print(f"[IncrementalSpine] Loaded metadata: {len(self._id_to_doc)} docs, {len(self._deleted_ids)} soft-deleted")

        if index_file.exists():
            cpu_index = faiss.read_index(str(index_file))
            if self.use_gpu and faiss.get_num_gpus() > 0 and self.gpu_resources is not None:
                self.index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, cpu_index)
            else:
                self.index = cpu_index
            print(f"[IncrementalSpine] Loaded index with {self.index.ntotal} vectors")

    def _save(self):
        """Persist state to disk."""
        meta_file = self.index_path / "incremental_meta.json"
        index_file = self.index_path / "incremental.index"

        data = {
            "next_id": self._next_id,
            "path_to_ids": self._path_to_ids,
            "id_to_doc": self._id_to_doc,
            "content_hashes": self._content_hashes,
            "deleted_ids": list(self._deleted_ids),
            "stats": self.stats,
            "saved_at": datetime.utcnow().isoformat(),
        }

        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        # Save FAISS index (convert from GPU if needed)
        if self.index.ntotal > 0:
            if self.use_gpu and faiss.get_num_gpus() > 0:
                cpu_index = faiss.index_gpu_to_cpu(self.index)
            else:
                cpu_index = self.index
            faiss.write_index(cpu_index, str(index_file))

        print(f"[IncrementalSpine] Saved: {len(self._id_to_doc)} docs, {self.index.ntotal} vectors")

    def _hash_content(self, text: str) -> str:
        """Generate content hash for dedup."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def add_document(self, path: str, text: str, force: bool = False) -> Tuple[bool, str]:
        """
        Add or update a document.

        Args:
            path: Document path (relative)
            text: Document content
            force: Skip dedup check

        Returns:
            (success, message)
        """
        content_hash = self._hash_content(text)

        # Dedup check
        if not force and content_hash in self._content_hashes:
            existing_id = self._content_hashes[content_hash]
            if existing_id not in self._deleted_ids:
                return (False, "duplicate")

        # Check if this path already exists (update case)
        if path in self._path_to_ids:
            old_ids = self._path_to_ids[path]
            # Soft-delete old chunks
            for old_id in old_ids:
                self._deleted_ids.add(old_id)
                self.stats["soft_deletes"] += 1
            self.stats["updates"] += 1
        else:
            self.stats["adds"] += 1

        # Embed
        embedding = self.embedder.encode(text, convert_to_numpy=True).astype('float32')

        # Assign ID
        doc_id = self._next_id
        self._next_id += 1

        # Add to FAISS
        self.index.add_with_ids(
            embedding.reshape(1, -1),
            np.array([doc_id], dtype='int64')
        )

        # Update mappings
        self._path_to_ids[path] = [doc_id]  # Could be multiple for chunks
        self._id_to_doc[doc_id] = {
            "path": path,
            "text": text[:2000],  # Truncate for storage
            "hash": content_hash,
            "added_at": datetime.utcnow().isoformat(),
        }
        self._content_hashes[content_hash] = doc_id

        self.stats["last_modified"] = datetime.utcnow().isoformat()

        return (True, "added")

    def remove_document(self, path: str) -> bool:
        """
        Soft-delete a document by path.

        Note: We use soft deletes because FAISS remove_ids is slow.
        Periodic compaction will clean up deleted vectors.
        """
        if path not in self._path_to_ids:
            return False

        for doc_id in self._path_to_ids[path]:
            self._deleted_ids.add(doc_id)
            # Remove from content hash dedup
            doc = self._id_to_doc.get(doc_id)
            if doc and doc.get("hash") in self._content_hashes:
                del self._content_hashes[doc["hash"]]

        del self._path_to_ids[path]
        self.stats["deletes"] += 1
        self.stats["last_modified"] = datetime.utcnow().isoformat()

        return True

    def query(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """
        Query the index, filtering out soft-deleted results.
        """
        if self.index.ntotal == 0:
            return []

        # Embed query
        query_embedding = self.embedder.encode(query_text, convert_to_numpy=True).astype('float32')

        # Search (get extra results to account for deleted)
        search_k = min(top_k * 3, self.index.ntotal)  # 3x to handle deletions
        distances, ids = self.index.search(query_embedding.reshape(1, -1), search_k)

        results = []
        for dist, doc_id in zip(distances[0], ids[0]):
            if doc_id == -1:  # FAISS returns -1 for empty slots
                continue
            if doc_id in self._deleted_ids:
                continue  # Skip soft-deleted

            doc = self._id_to_doc.get(int(doc_id))
            if doc:
                results.append({
                    "id": int(doc_id),
                    "path": doc["path"],
                    "text": doc["text"],
                    "score": float(1 / (1 + dist)),  # Convert distance to similarity
                })

            if len(results) >= top_k:
                break

        return results

    def compact(self) -> Dict:
        """
        Remove soft-deleted vectors and rebuild index.
        Call periodically (e.g., daily) to reclaim space.
        """
        if not self._deleted_ids:
            return {"removed": 0, "remaining": len(self._id_to_doc)}

        print(f"[IncrementalSpine] Compacting: removing {len(self._deleted_ids)} deleted vectors...")

        # Get all non-deleted documents
        active_docs = [
            (doc_id, doc)
            for doc_id, doc in self._id_to_doc.items()
            if doc_id not in self._deleted_ids
        ]

        # Re-embed and rebuild index
        self._init_index()  # Fresh index

        new_id_to_doc = {}
        new_path_to_ids = {}
        new_content_hashes = {}

        for old_id, doc in active_docs:
            text = doc["text"]
            embedding = self.embedder.encode(text, convert_to_numpy=True).astype('float32')

            new_id = len(new_id_to_doc)
            self.index.add_with_ids(
                embedding.reshape(1, -1),
                np.array([new_id], dtype='int64')
            )

            new_id_to_doc[new_id] = doc
            path = doc["path"]
            if path not in new_path_to_ids:
                new_path_to_ids[path] = []
            new_path_to_ids[path].append(new_id)
            if doc.get("hash"):
                new_content_hashes[doc["hash"]] = new_id

        removed_count = len(self._deleted_ids)

        # Update state
        self._id_to_doc = new_id_to_doc
        self._path_to_ids = new_path_to_ids
        self._content_hashes = new_content_hashes
        self._deleted_ids = set()
        self._next_id = len(new_id_to_doc)

        self._save()

        return {"removed": removed_count, "remaining": len(self._id_to_doc)}

    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            **self.stats,
            "total_docs": len(self._id_to_doc),
            "active_docs": len(self._id_to_doc) - len(self._deleted_ids),
            "deleted_docs": len(self._deleted_ids),
            "index_vectors": self.index.ntotal,
            "unique_paths": len(self._path_to_ids),
        }


# Convenience function for migration
def migrate_from_spine(old_spine, new_index: IncrementalSpineIndex):
    """Migrate documents from old ScornSpine to IncrementalSpineIndex."""
    print(f"[Migration] Migrating {len(old_spine.docs)} documents...")

    for doc in old_spine.docs:
        new_index.add_document(
            path=doc.get("path", "unknown"),
            text=doc.get("text", ""),
            force=True  # Skip dedup during migration
        )

    new_index._save()
    print(f"[Migration] Complete: {new_index.get_stats()}")


if __name__ == "__main__":
    # Test
    idx = IncrementalSpineIndex()

    # Add test doc
    success, msg = idx.add_document("test/doc1.md", "This is a test document about RAG systems.")
    print(f"Add: {success}, {msg}")

    # Query
    results = idx.query("RAG", top_k=5)
    print(f"Query results: {len(results)}")
    for r in results:
        print(f"  {r['path']}: {r['score']:.3f}")

    # Stats
    print(f"Stats: {idx.get_stats()}")

    # Save
    idx._save()
