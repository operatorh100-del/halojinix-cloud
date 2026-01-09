"""
ScornSpine - Halojinix RAG Engine
RT9600: Renamed from MinimalSpine per Human directive

Clean, fast RAG implementation: ~200 lines, local FAISS, GPU-accelerated.

Usage:
    from spine import ScornSpine
    spine = ScornSpine()
    spine.load_workspace("F:/primewave-engine")
    results = spine.query("agent protocol")
"""

import os
import json
import hashlib
import fnmatch  # RT12070: For .ragignore pattern matching
from pathlib import Path
from typing import List, Dict, Any, Optional, Set

# RT20260105: Set cache directories (cloud-compatible)
if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/runpod-volume/cache/huggingface")
if not os.environ.get("TRANSFORMERS_CACHE"):
    os.environ["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE", "/runpod-volume/cache/huggingface")
if not os.environ.get("SENTENCE_TRANSFORMERS_HOME"):
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.environ.get("SENTENCE_TRANSFORMERS_HOME", "/runpod-volume/cache/sentence_transformers")

# Core dependencies only
import numpy as np
import torch  # RT22700: For GPU cleanup after batch operations

# Try multiple import paths for embeddings_cache
try:
    from haloscorn.scornspine.embeddings_cache import init_cache, get_cached_embedding, cache_embeddings_batch
except ImportError:
    try:
        from scornspine.embeddings_cache import init_cache, get_cached_embedding, cache_embeddings_batch
    except ImportError:
        try:
            from embeddings_cache import init_cache, get_cached_embedding, cache_embeddings_batch
        except ImportError:
            # Fallback: define stubs if cache not available
            def init_cache(*args, **kwargs): pass
            def get_cached_embedding(*args, **kwargs): return None
            def cache_embeddings_batch(*args, **kwargs): pass
            print("WARNING: embeddings_cache not available, caching disabled")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("WARNING: faiss not available, using numpy fallback")

# RT31210: HNSW index configuration (O(log n) vs O(n) brute force)
# Set SCORNSPINE_INDEX_TYPE=hnsw to enable (default: flat for compatibility)
SCORNSPINE_INDEX_TYPE = os.environ.get("SCORNSPINE_INDEX_TYPE", "flat").lower()

def create_faiss_index(dimension: int, force_flat: bool = False) -> Any:
    """
    RT31210: Factory function for FAISS index creation.
    
    - HNSW: O(log n) search, 10x faster on large indices, 95%+ recall
    - Flat: O(n) brute force, 100% recall, supports deletion
    
    Set SCORNSPINE_INDEX_TYPE=hnsw for performance, =flat for compatibility.
    """
    if not FAISS_AVAILABLE:
        return None
    
    use_hnsw = (SCORNSPINE_INDEX_TYPE == "hnsw") and not force_flat
    
    if use_hnsw:
        # HNSW parameters: M=32 (connections per node), efConstruction=200 (build quality)
        index = faiss.IndexHNSWFlat(dimension, 32)
        index.hnsw.efConstruction = 200  # Higher = better recall, slower build
        index.hnsw.efSearch = 64  # Higher = better recall, slower search
        print(f"[ScornSpine] RT31210: Using HNSW index (M=32, efConstruction=200, efSearch=64)")
        return index
    else:
        print(f"[ScornSpine] Using Flat index (brute force, 100% recall)")
        return faiss.IndexFlatL2(dimension)

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False
    print("ERROR: sentence_transformers required")


# ═══════════════════════════════════════════════════════════════════════════════
# RT12070: .ragignore support (imported from vertebrae.py pattern)
# ═══════════════════════════════════════════════════════════════════════════════

def load_ragignore(workspace_path: Path) -> List[str]:
    """Load exclusion patterns from .ragignore file."""
    ragignore_file = workspace_path / ".ragignore"
    patterns = []
    if ragignore_file.exists():
        with open(ragignore_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    # Skip negation patterns (we don't support !pattern yet)
                    if not line.startswith('!'):
                        patterns.append(line)
    return patterns


def should_exclude(path: Path, workspace: Path, patterns: List[str]) -> bool:
    """Check if a path should be excluded based on ragignore patterns."""
    try:
        rel_path = path.relative_to(workspace)
        rel_str = str(rel_path).replace('\\', '/')
    except ValueError:
        return False  # Path not relative to workspace, don't exclude

    for pattern in patterns:
        # Handle directory patterns (ending with /)
        if pattern.endswith('/'):
            dir_pattern = pattern.rstrip('/')
            if rel_str.startswith(dir_pattern + '/') or rel_str == dir_pattern:
                return True
        # Handle glob patterns
        elif fnmatch.fnmatch(rel_str, pattern):
            return True
        # Handle ** patterns that could match anywhere in path
        elif '**' in pattern:
            if fnmatch.fnmatch(rel_str, pattern):
                return True
        # Simple filename match
        elif fnmatch.fnmatch(path.name, pattern):
            return True
        # Simple path contains check for directory patterns without trailing slash
        elif '/' not in pattern and pattern in rel_str.split('/'):
            return True

    return False


class ScornSpine:  # RT9600: Renamed from MinimalSpine
    """
    ScornSpine RAG engine. Local FAISS + GPU-accelerated embeddings.
    Clean, fast, reliable.
    
    RT31800: Now supports Qdrant Cloud backend for serverless deployment.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",  # RT12002: Upgrade from MiniLM (384-dim) to e5-base (768-dim)
        index_path: Optional[Path] = None,
        # RT31800: Qdrant Cloud support
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        collection: str = "halojinix-spine"
    ):
        self.model_name = model_name
        self.index_path = index_path or Path("/runpod-volume/index" if qdrant_url else "F:/primewave-engine/haloscorn/scornspine/index_minimal")
        self.collection = collection
        
        # RT31800: Qdrant Cloud backend
        self.qdrant_client = None
        self.use_qdrant = bool(qdrant_url and qdrant_api_key)
        
        if self.use_qdrant:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams
                print(f"[ScornSpine] RT31800: Connecting to Qdrant Cloud: {qdrant_url[:50]}...")
                self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
                print(f"[ScornSpine] RT31800: Qdrant Cloud connected")
            except Exception as e:
                print(f"[ScornSpine] RT31800: Qdrant connection failed: {e}")
                self.use_qdrant = False
        
        if not self.use_qdrant:
            self.index_path.mkdir(parents=True, exist_ok=True)

        # Load embedding model
        print(f"Loading model: {model_name}")
        import os
        import torch
        # Only set offline mode if not in cloud (model is pre-downloaded in Dockerfile)
        if not self.use_qdrant:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # RT20260104-GPU-FIX: Check SCORNSPINE_DEVICE env var for CPU mode
        # Set SCORNSPINE_DEVICE=cpu to force CPU-only operation
        force_device = os.environ.get("SCORNSPINE_DEVICE", "").lower()
        if force_device == "cpu":
            device = "cpu"
            print("[ScornSpine] RT20260104: SCORNSPINE_DEVICE=cpu - forcing CPU mode")
        else:
            # RT9500: Enable GPU for SentenceTransformer (main compute)
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # RT9600: Removed local_files_only - incompatible with newer sentence-transformers
        self.embedder = SentenceTransformer(model_name, device=device)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        print(f"Model loaded on {device.upper()}. Dimension: {self.dim}")

        # Initialize FAISS index
        # RT12100: Store GPU resources as instance variable to prevent GC collection
        # RT22500: PERMANENT GPU FIX - Keep FAISS on CPU always. Only embedder needs GPU.
        # This prevents GPU memory fragmentation and keeps VRAM usage stable.
        self._gpu_res = None  # Deprecated - not using GPU FAISS anymore
        if FAISS_AVAILABLE:
            # RT31210: Use factory function for index creation (supports HNSW)
            self.index = create_faiss_index(self.dim)
            # RT22500: FAISS always on CPU now - GPU FAISS caused memory leaks
        else:
            self.index = None
            self.embeddings = []  # Numpy fallback

        # Document storage (simple list)
        self.docs: List[Dict[str, Any]] = []

        # RT970: Hash-based deduplication tracking
        self._content_hashes: Set[str] = set()

        # RT26300: Incremental indexing metadata (path -> {mtime, size})
        self._file_metadata: Dict[str, Dict[str, Any]] = {}

        # RT937: Initialize embedding cache
        init_cache()

        # Load existing index if available
        self._load_index()

    def _load_index(self):
        """
        Load persisted index from disk with error handling.
        RT12102: Added error handling and corruption recovery.
        """
        docs_file = self.index_path / "docs.json"
        index_file = self.index_path / "faiss.index"
        embeddings_file = self.index_path / "embeddings.npy"  # RT8700: numpy fallback

        if docs_file.exists():
            try:
                with open(docs_file, 'r', encoding='utf-8-sig') as f:  # RT9700: utf-8-sig handles BOM
                    self.docs = json.load(f)
                print(f"Loaded {len(self.docs)} documents from cache")
            except (json.JSONDecodeError, IOError) as e:
                # RT12102: Corrupted docs file - start fresh
                print(f"[ScornSpine] WARNING: Corrupted docs.json: {e}. Starting fresh.")
                self.docs = []
                docs_file.unlink()  # Delete corrupt file
                if index_file.exists():
                    index_file.unlink()  # Delete matching index too
                return  # Exit early, no point loading index

            # RT970: Rebuild hash set from loaded docs
            self._content_hashes = set()
            self._file_metadata = {}  # RT26300: Rebuild file metadata
            for doc in self.docs:
                if 'hash' in doc:
                    self._content_hashes.add(doc['hash'])
                else:
                    # Backward compat: compute hash from text
                    content_hash = hashlib.sha256(doc['text'].encode('utf-8')).hexdigest()[:16]
                    self._content_hashes.add(content_hash)
                    doc['hash'] = content_hash

                # RT26300: Track file metadata for incremental indexing
                # Strip #chunkN from path to get base filename
                path_key = doc['path'].split('#chunk')[0]
                if 'mtime' in doc and 'size' in doc:
                    self._file_metadata[path_key] = {
                        'mtime': doc['mtime'],
                        'size': doc['size']
                    }
            print(f"Rebuilt {len(self._content_hashes)} content hashes and {len(self._file_metadata)} file metadata entries")

            if FAISS_AVAILABLE and index_file.exists():
                try:
                    cpu_index = faiss.read_index(str(index_file))

                    # RT12102: Verify consistency
                    if cpu_index.ntotal != len(self.docs):
                        print(f"[ScornSpine] WARNING: Index mismatch ({cpu_index.ntotal} vectors vs {len(self.docs)} docs). Rebuilding required.")

                    # RT20260104-GPU-FIX: Check env var for CPU mode
                    force_cpu = os.environ.get("SCORNSPINE_DEVICE", "").lower() == "cpu"
                    if not force_cpu and faiss.get_num_gpus() > 0:
                        # RT12100: Reuse existing _gpu_res or create new persistent reference
                        if self._gpu_res is None:
                            self._gpu_res = faiss.StandardGpuResources()  # type: ignore[attr-defined]
                        self.index = faiss.index_cpu_to_gpu(  # type: ignore[attr-defined]
                            self._gpu_res,
                            0,
                            cpu_index
                        )
                    else:
                        self.index = cpu_index
                    print(f"Loaded FAISS index with {self.index.ntotal} vectors")
                except Exception as e:
                    # RT12102: Corrupted index file - start fresh
                    print(f"[ScornSpine] WARNING: Corrupted faiss.index: {e}. Rebuilding required.")
                    self.docs = []  # Clear docs too since index is gone
                    self._content_hashes = set()
                    if index_file.exists():
                        index_file.unlink()
                    if docs_file.exists():
                        docs_file.unlink()
            elif not FAISS_AVAILABLE and embeddings_file.exists():
                try:
                    # RT8700: Load numpy embeddings for fallback
                    self.embeddings = list(np.load(embeddings_file))
                    print(f"Loaded {len(self.embeddings)} numpy embeddings from cache")
                except Exception as e:
                    print(f"[ScornSpine] WARNING: Corrupted embeddings: {e}. Starting fresh.")
                    self.embeddings = []

    def _save_index(self):
        """
        Persist index to disk with error handling and atomic writes.
        RT12101: Added error handling and atomic writes to prevent corruption.
        """
        docs_file = self.index_path / "docs.json"
        index_file = self.index_path / "faiss.index"
        embeddings_file = self.index_path / "embeddings.npy"  # RT8700: numpy fallback

        # RT12101: Use temp files for atomic writes
        docs_temp = self.index_path / "docs.json.tmp"
        index_temp = self.index_path / "faiss.index.tmp"

        try:
            # Write docs to temp file first
            with open(docs_temp, 'w', encoding='utf-8') as f:
                json.dump(self.docs, f)

            # Write FAISS index to temp file
            if FAISS_AVAILABLE and self.index.ntotal > 0:  # type: ignore[union-attr]
                # Convert GPU index to CPU for saving
                if faiss.get_num_gpus() > 0:
                    cpu_index = faiss.index_gpu_to_cpu(self.index)  # type: ignore[attr-defined]
                else:
                    cpu_index = self.index
                faiss.write_index(cpu_index, str(index_temp))
            elif not FAISS_AVAILABLE and len(self.embeddings) > 0:
                # RT8700: Save numpy embeddings for fallback
                np.save(embeddings_file, np.array(self.embeddings))

            # RT12101: Atomic rename - only if both writes succeeded
            if docs_temp.exists():
                docs_temp.replace(docs_file)
            if index_temp.exists():
                index_temp.replace(index_file)

            print(f"Saved {len(self.docs)} documents to {self.index_path}")

        except Exception as e:
            # RT12101: Log error but don't crash - leave old files intact
            print(f"[ScornSpine] ERROR saving index: {e}")
            # Clean up temp files on failure
            if docs_temp.exists():
                docs_temp.unlink()
            if index_temp.exists():
                index_temp.unlink()

    def add_document(self, path: str, text: str) -> bool:
        """
        Add a single document with deduplication.
        NOTE: For bulk loading, use add_documents_batch() for better GPU efficiency.

        Returns:
            True if document was added, False if it was a duplicate.
        """
        # RT970: Hash-based deduplication - skip exact duplicates
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        if content_hash in self._content_hashes:
            return False  # Skip duplicate
        self._content_hashes.add(content_hash)

        # Embed (single doc - for batch loading use add_documents_batch)
        embedding = self.embedder.encode(text, convert_to_numpy=True)

        # Add to index
        if FAISS_AVAILABLE:
            self.index.add(embedding.reshape(1, -1))  # type: ignore[union-attr]
        else:
            self.embeddings.append(embedding)

        # Store doc with hash for later dedup reference
        self.docs.append({
            'path': path,
            'text': text[:2000],  # Truncate for storage
            'idx': len(self.docs),
            'hash': content_hash
        })
        return True

    def add_documents_batch(self, documents: List[tuple]) -> int:
        """
        RT22700: BATCH add documents - prevents GPU memory fragmentation.
        RT26300: Added mtime/size tracking for incremental indexing.

        Args:
            documents: List of (path, text, mtime, size) tuples

        Returns:
            Number of documents actually added (excluding duplicates)
        """
        if not documents:
            return 0

        # Phase 1: Filter duplicates BEFORE any GPU work
        unique_docs = []
        cached_embeddings = []
        docs_to_embed = []

        for doc_tuple in documents:
            path = doc_tuple[0]
            text = doc_tuple[1]
            mtime = doc_tuple[2] if len(doc_tuple) > 2 else 0
            size = doc_tuple[3] if len(doc_tuple) > 3 else 0

            content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
            if content_hash not in self._content_hashes:
                # RT937: Check persistent cache
                cached = get_cached_embedding(path, text)
                if cached is not None:
                    cached_embeddings.append((path, text, content_hash, mtime, size, cached))
                    self._content_hashes.add(content_hash)
                else:
                    docs_to_embed.append((path, text, content_hash, mtime, size))

        if not cached_embeddings and not docs_to_embed:
            return 0

        # Phase 2: BATCH encode only what's not in cache
        new_embeddings = []
        if docs_to_embed:
            texts = [text for _, text, _, _, _ in docs_to_embed]
            new_embeddings = self.embedder.encode(
                texts,
                convert_to_numpy=True,
                batch_size=32,
                show_progress_bar=False
            )

            # RT937: Update persistent cache
            cache_items = []
            for (path, text, content_hash, mtime, size), emb in zip(docs_to_embed, new_embeddings):
                cache_items.append((path, text, emb))
                self._content_hashes.add(content_hash)
            cache_embeddings_batch(cache_items)

        # Phase 3: Combine and add to index
        all_embeddings = []
        all_metadata = []

        # Add cached ones
        for path, text, content_hash, mtime, size, emb in cached_embeddings:
            all_embeddings.append(emb)
            all_metadata.append((path, text, content_hash, mtime, size))

        # Add newly embedded ones
        for (path, text, content_hash, mtime, size), emb in zip(docs_to_embed, new_embeddings):
            all_embeddings.append(emb)
            all_metadata.append((path, text, content_hash, mtime, size))

        if FAISS_AVAILABLE:
            self.index.add(np.array(all_embeddings, dtype='float32'))  # type: ignore[union-attr]
        else:
            self.embeddings.extend(all_embeddings)

        # Phase 4: Store doc metadata
        for (path, text, content_hash, mtime, size) in all_metadata:
            self.docs.append({
                'path': path,
                'text': text[:2000],
                'idx': len(self.docs),
                'hash': content_hash,
                'mtime': mtime,
                'size': size
            })

        return len(all_metadata)

    def load_workspace(
        self,
        workspace_path: str,
        extensions: tuple = ('.md', '.py', '.txt', '.json', '.ts', '.tsx', '.js', '.jsx', '.ps1', '.bat', '.sh', '.yaml', '.yml', '.mjs', '.cjs', '.sql', '.psm1', '.psd1', '.xml', '.html', '.css', '.scss', '.less', '.config', '.ini', '.toml', '.jsonl', '.csv', '.mdx', '.log'),
        max_docs: int = 100000,  # RT20260105: Increased from 5000 to 100k for full workspace
        force_reindex: bool = False  # RT26300: Option to force full reindex
    ):
        """
        Load all documents from workspace with deduplication.
        RT22700: Uses batch encoding to prevent GPU memory fragmentation.
        RT26300: Uses incremental indexing - skips unchanged files.
        """
        workspace = Path(workspace_path)
        skipped_ragignore = 0
        skipped_incremental = 0  # RT26300: Track incremental skips

        # RT12070: Load .ragignore patterns
        ragignore_patterns = load_ragignore(workspace)
        if ragignore_patterns:
            print(f"Loaded {len(ragignore_patterns)} .ragignore patterns")

        # Skip patterns
        skip_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build', '.venv-1'}

        print(f"Scanning {workspace_path} (incremental={'OFF' if force_reindex else 'ON'})...")

        if force_reindex:
            print("[ScornSpine] RT26300: Force reindex requested - clearing in-memory index")
            self.docs = []
            self._content_hashes = set()
            self._file_metadata = {}
            if FAISS_AVAILABLE:
                self.index = create_faiss_index(self.dim)
            else:
                self.embeddings = []

        # RT22700: Collect documents first, then batch encode
        pending_docs: List[tuple] = []  # (path, text, mtime, size)

        for root, dirs, files in os.walk(workspace):
            # Skip unwanted directories
            dirs[:] = [d for d in dirs if d not in skip_dirs]

            # RT12070: Also filter directories by ragignore patterns
            if ragignore_patterns:
                dirs[:] = [d for d in dirs if not should_exclude(Path(root) / d, workspace, ragignore_patterns)]

            for file in files:
                if len(pending_docs) >= max_docs:
                    break

                if not file.endswith(extensions):
                    continue

                filepath = Path(root) / file
                rel_path = str(filepath.relative_to(workspace))

                # DEBUG: Print every 1000th file scanned
                if len(pending_docs) % 1000 == 0:
                    print(f"  Scanning: {rel_path}")

                # RT12070: Check ragignore patterns
                if ragignore_patterns and should_exclude(filepath, workspace, ragignore_patterns):
                    skipped_ragignore += 1
                    continue

                # RT26300: Incremental check
                try:
                    stat = filepath.stat()
                    mtime = stat.st_mtime
                    size = stat.st_size

                    if not force_reindex and rel_path in self._file_metadata:
                        meta = self._file_metadata[rel_path]
                        if meta['mtime'] == mtime and meta['size'] == size:
                            skipped_incremental += 1
                            continue
                except Exception:
                    mtime, size = 0, 0

                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()

                    if len(text) < 50:  # Skip tiny files
                        continue

                    # Chunk large files
                    if len(text) > 2000:
                        # RT970: Non-overlapping chunks to reduce duplicates
                        chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]
                        for i, chunk in enumerate(chunks[:20]):  # RT20260105: Increased from 10 to 20 chunks per file
                            pending_docs.append((f"{rel_path}#chunk{i}", chunk, mtime, size))
                    else:
                        pending_docs.append((rel_path, text, mtime, size))

                    if len(pending_docs) % 500 == 0:
                        print(f"  Collected {len(pending_docs)} documents...")

                except Exception as e:
                    pass  # Skip unreadable files

            if len(pending_docs) >= max_docs:
                break

        if not pending_docs:
            print(f"No new documents to index (skipped {skipped_incremental} unchanged files)")
            return 0

        print(f"Collected {len(pending_docs)} documents. Starting batch encoding...")

        # RT22700: BATCH ENCODE
        added = self.add_documents_batch(pending_docs)

        # RT22700: Single cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        print(f"Indexed {added} documents (skipped {len(pending_docs) - added} duplicates, {skipped_ragignore} by .ragignore, {skipped_incremental} unchanged)")
        self._save_index()
        return added

    def query(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the index. Returns top_k most similar documents.

        Args:
            question: The search query
            top_k: Number of results to return

        Returns:
            List of documents with path, text, and score
        """
        # Embed query - single encode call, minimal allocation
        # RT22700: PyTorch/CUDA memory is managed internally - no cleanup needed
        query_embedding = self.embedder.encode(question, convert_to_numpy=True)
        
        # RT31800: Use Qdrant Cloud if configured
        if self.use_qdrant and self.qdrant_client:
            try:
                from qdrant_client.models import PointStruct
                search_result = self.qdrant_client.search(
                    collection_name=self.collection,
                    query_vector=query_embedding.tolist(),
                    limit=top_k
                )
                results = []
                for i, hit in enumerate(search_result):
                    results.append({
                        'path': hit.payload.get('path', 'unknown'),
                        'text': hit.payload.get('text', '')[:2000],
                        'score': float(hit.score),
                        'rank': i + 1,
                        'idx': hit.id
                    })
                return results
            except Exception as e:
                print(f"[ScornSpine] Qdrant query failed: {e}")
                return []
        
        # Local FAISS fallback
        if len(self.docs) == 0:
            return []

        # Search
        if FAISS_AVAILABLE:
            distances, indices = self.index.search(  # type: ignore[union-attr]
                query_embedding.reshape(1, -1),
                min(top_k, len(self.docs))
            )

            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx >= 0 and idx < len(self.docs):
                    doc = self.docs[idx].copy()
                    doc['score'] = float(1 / (1 + dist))  # Convert distance to similarity
                    doc['rank'] = i + 1
                    results.append(doc)

            return results
        else:
            # Numpy fallback (slow but works)
            embeddings = np.array(self.embeddings)
            distances = np.linalg.norm(embeddings - query_embedding, axis=1)
            indices = np.argsort(distances)[:top_k]

            results = []
            for i, idx in enumerate(indices):
                doc = self.docs[idx].copy()
                doc['score'] = float(1 / (1 + distances[idx]))
                doc['rank'] = i + 1
                results.append(doc)

            return results

    def health(self) -> Dict[str, Any]:
        """Health check."""
        # RT20260104-GPU-FIX: Report actual device usage, not just availability
        force_cpu = os.environ.get("SCORNSPINE_DEVICE", "").lower() == "cpu"
        using_gpu = FAISS_AVAILABLE and faiss.get_num_gpus() > 0 and not force_cpu
        return {
            'status': 'healthy' if len(self.docs) > 0 else 'empty',
            'documents': len(self.docs),
            'unique_hashes': len(self._content_hashes),
            'model': self.model_name,
            'dimension': self.dim,
            'gpu': using_gpu,
            'index_path': str(self.index_path)
        }

    def warmup(self) -> Dict[str, Any]:
        """
        RT642: Pre-warm ScornSpine components at startup.

        Loads index and embedding model before first query
        to eliminate cold-start latency.

        Returns:
            Dict with warmup status and timings
        """
        import time
        result = {
            'index_loaded': False,
            'model_loaded': False,
            'index_load_ms': 0,
            'model_load_ms': 0,
        }

        # Load index from disk
        start = time.time()
        self._load_index()
        if len(self.docs) > 0:
            result['index_loaded'] = True
            result['index_load_ms'] = round((time.time() - start) * 1000, 2)
            print(f"[ScornSpine] Index pre-warmed: {len(self.docs)} docs in {result['index_load_ms']}ms")
        else:
            print("[ScornSpine] No index to pre-warm (will build on first index_documents call)")

        # Warm up embedding model with a test encode
        start = time.time()
        try:
            _ = self.embedder.encode("warmup test", convert_to_numpy=True)
            result['model_loaded'] = True
            result['model_load_ms'] = round((time.time() - start) * 1000, 2)
            print(f"[ScornSpine] Embedding model warmed in {result['model_load_ms']}ms")
        except Exception as e:
            print(f"[ScornSpine] Embedding model warmup failed: {e}")

        return result

    def dedupe_index(self, by_path: bool = True) -> Dict[str, Any]:
        """
        RT970: Remove duplicates from existing index.

        Args:
            by_path: If True, keep only newest version of each path.
                     If False, only remove exact content duplicates.

        Returns:
            Stats about the deduplication operation.
        """
        if len(self.docs) == 0:
            return {'status': 'empty', 'before': 0, 'after': 0, 'removed': 0}

        print(f"Starting deduplication of {len(self.docs)} documents (by_path={by_path})...")

        if by_path:
            # Keep only the last occurrence of each path (newest version)
            path_to_idx: Dict[str, int] = {}
            for i, doc in enumerate(self.docs):
                path_to_idx[doc['path']] = i  # Later indices overwrite earlier

            keep_indices = set(path_to_idx.values())
            print(f"Keeping {len(keep_indices)} unique paths from {len(self.docs)} docs")
        else:
            # Keep only unique content hashes
            seen_hashes: Set[str] = set()
            keep_indices = set()
            for i, doc in enumerate(self.docs):
                if 'hash' in doc:
                    content_hash = doc['hash']
                else:
                    content_hash = hashlib.sha256(doc['text'].encode('utf-8')).hexdigest()[:16]

                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    keep_indices.add(i)

        # GPU index can't reconstruct - need full reindex
        if FAISS_AVAILABLE and faiss.get_num_gpus() > 0:
            print("GPU index detected - full reindex required for dedup")
            return self._full_reindex_dedup(by_path=by_path)

        # Get embeddings for kept docs
        if FAISS_AVAILABLE:
            total = self.index.ntotal  # type: ignore[union-attr]
            all_embeddings = np.zeros((total, self.dim), dtype='float32')  # type: ignore[arg-type]
            for i in range(total):
                self.index.reconstruct(i, all_embeddings[i])  # type: ignore[union-attr]
        else:
            all_embeddings = np.array(self.embeddings)

        # Build new doc list and embeddings
        unique_docs = []
        unique_embeddings = []
        new_hashes = set()

        for i in sorted(keep_indices):
            doc = self.docs[i].copy()
            if 'hash' not in doc:
                doc['hash'] = hashlib.sha256(doc['text'].encode('utf-8')).hexdigest()[:16]
            doc['idx'] = len(unique_docs)
            unique_docs.append(doc)
            new_hashes.add(doc['hash'])
            if i < len(all_embeddings):
                unique_embeddings.append(all_embeddings[i])

        removed = len(self.docs) - len(unique_docs)
        print(f"Removed {removed} duplicates ({len(unique_docs)} unique docs remain)")

        # Rebuild index with unique docs
        self.docs = unique_docs
        self._content_hashes = new_hashes

        if FAISS_AVAILABLE:
            # RT31210: force_flat=True for rebuild (HNSW doesn't support deletion)
            self.index = create_faiss_index(self.dim, force_flat=True)
            if unique_embeddings:
                embeddings_array = np.array(unique_embeddings, dtype='float32')
                self.index.add(embeddings_array)  # type: ignore[union-attr]
        else:
            self.embeddings = unique_embeddings

        self._save_index()

        return {
            'status': 'success',
            'before': len(self.docs) + removed,
            'after': len(self.docs),
            'removed': removed
        }

    def _full_reindex_dedup(self, by_path: bool = True) -> Dict[str, Any]:
        """Rebuild entire index from docs with deduplication (for GPU indexes)."""
        print(f"Full reindex with deduplication (by_path={by_path})...")

        old_docs = self.docs

        if by_path:
            # Keep only newest version of each path
            path_to_doc: Dict[str, Dict[str, Any]] = {}
            for doc in old_docs:
                path_to_doc[doc['path']] = doc  # Later overwrites earlier
            docs_to_add = list(path_to_doc.values())
        else:
            docs_to_add = old_docs

        # Clear index
        if FAISS_AVAILABLE:
            # RT31210: Use factory, but HNSW rebuild needs fresh index
            self.index = create_faiss_index(self.dim)
            # RT22500: GPU FAISS deprecated - CPU only for stability
        else:
            self.embeddings = []

        self.docs = []
        self._content_hashes = set()

        added = 0
        skipped = 0

        for doc in docs_to_add:
            result = self.add_document(doc['path'], doc['text'])
            if result:
                added += 1
            else:
                skipped += 1

        self._save_index()

        return {
            'status': 'success',
            'before': len(old_docs),
            'after': added,
            'removed': skipped
        }


# CLI and Test
if __name__ == "__main__":
    import time
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="ScornSpine CLI")
    parser.add_argument("action", nargs="?", choices=["index", "query", "health"], default="query")
    parser.add_argument("--force", action="store_true", help="Force full reindex")
    parser.add_argument("--query", type=str, default="agent protocol", help="Query string")
    parser.add_argument("--path", type=str, default="F:/primewave-engine", help="Workspace path to index")

    args = parser.parse_args()

    print("=" * 50)
    print(f"ScornSpine CLI - Action: {args.action}")
    print("=" * 50)

    spine = ScornSpine()

    if args.action == "index":
        print(f"\nIndexing workspace: {args.path} (force={args.force})...")
        start = time.time()
        count = spine.load_workspace(args.path, force_reindex=args.force)
        print(f"Indexing took {time.time() - start:.2f}s. Added {count} documents.")
        sys.exit(0)

    if args.action == "health":
        print("\nHealth:", json.dumps(spine.health(), indent=2))
        sys.exit(0)

    # Default: Query
    # If no docs loaded, index workspace automatically
    if len(spine.docs) == 0:
        print("\nNo index found. Indexing workspace...")
        spine.load_workspace(args.path)

    print(f"\nQuery: '{args.query}'")
    start = time.time()
    results = spine.query(args.query, top_k=5)
    elapsed = time.time() - start

    print(f"Query took {elapsed*1000:.1f}ms")
    print(f"\nResults:")
    for r in results:
        print(f"  [{r['score']:.3f}] {r['path']}")
        # Safe print for Windows encoding
        preview = r['text'][:100].encode('ascii', 'replace').decode('ascii')
        print(f"    {preview}...")
