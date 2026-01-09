"""
ScornSpine Server - FastAPI RAG engine for Halojinix
RT9600: Renamed from ScornSpine per Human directive

Port: 7782 (Updated per RT1400/AGENTS.md)
Test: curl http://127.0.0.1:7782/query -d '{"query": "agent protocol"}'
"""

import os
import sys
import time

# RT8700: Winsock retry logic for Windows asyncio stability
# WinError 10106 = Winsock LSP catalog corruption - retry with backoff
def _safe_import_asyncio():
    """Import asyncio with retry for Windows Winsock issues."""
    for attempt in range(5):
        try:
            import asyncio
            return asyncio
        except OSError as e:
            if hasattr(e, 'winerror') and e.winerror == 10106:
                wait = 2 ** attempt
                print(f"[RT8700] Winsock error, retry {attempt+1}/5 in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("[RT8700] Failed to import asyncio after 5 attempts - Winsock corrupted. Run: netsh winsock reset")

# Pre-import asyncio with retry (uvicorn needs it)
_safe_import_asyncio()

import uvicorn
import numpy as np  # RT31200: For LatentGrid velocity calculations
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # RT8400: CORS for panel access
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

PORT = 7782  # Standard ScornSpine port (RT1400: Reconciled with AGENTS.md)

# Global stats tracking (RT9300: Moved up for earlier availability)
_query_count = 0
_total_query_time_ms = 0.0
_agent_stats: Dict[str, Dict[str, Any]] = {}
_startup_time = time.time()
_refreshing = False

# Add project root for imports
sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from haloscorn.scornspine.spine import ScornSpine  # RT9600: Renamed from spine_minimal
from haloscorn.scornspine.reranker import get_reranker  # RT970: Cross-encoder reranking
from haloscorn.scornspine.bm25_index import BM25Index, reciprocal_rank_fusion  # RT8600: PERFECTIFY
from haloscorn.scornspine.bloom_cache import global_cache  # RT5010: Synthesis Cache
from haloscorn.scornspine.metrics import metrics, timer  # RT31210: Telemetry infrastructure
from haloscorn.scornspine.qdrant_fallback import qdrant_fallback, needs_fallback, merge_results  # RT31701: Cloud fallback

# Create app
app = FastAPI(title="ScornSpine", version="2.0.0")  # RT9600: Version bump for rename

# RT8400: Add CORS middleware for Signal panel browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global spine instance
_spine: Optional[ScornSpine] = None  # RT9600: Renamed
_bm25: Optional[BM25Index] = None  # RT8600: BM25 index
_startup_time = time.time()
_indexing_thread = None
_indexing_status = "idle"


def get_spine() -> ScornSpine:
    """Get or create ScornSpine singleton."""
    global _spine, _indexing_status, _indexing_thread
    if _spine is None:
        print("[ScornSpine] Initializing instance...")
        _spine = ScornSpine()

        # If no docs, start indexing in background if not already started
        if len(_spine.docs) == 0 and _indexing_status == "idle":
            print("[ScornSpine] No cached index, starting background workspace scan...")

            def run_index():
                global _indexing_status
                _indexing_status = "indexing"
                try:
                    _spine.load_workspace("F:/primewave-engine")
                    _indexing_status = "ready"
                    print(f"[ScornSpine] Background indexing complete: {len(_spine.docs)} documents")

                    # Also build BM25 after main index is ready
                    get_bm25()
                except Exception as e:
                    _indexing_status = f"error: {str(e)}"
                    print(f"[ScornSpine] ERROR during background indexing: {e}")

            import threading
            _indexing_thread = threading.Thread(target=run_index, daemon=True)
            _indexing_thread.start()

    return _spine


def get_bm25() -> BM25Index:
    """RT8600: Get or create BM25 index singleton."""
    global _bm25
    if _bm25 is None:
        print("[BM25] Initializing...")
        _bm25 = BM25Index()
        # Try to load cached index first
        if not _bm25.load():
            print("[BM25] No cached index - building in background thread...")
            # Build asynchronously to not block server startup
            import threading
            def build_bm25():
                global _bm25
                spine = get_spine()
                if spine.docs:
                    print(f"[BM25] Building from {len(spine.docs)} docs...")
                    docs = [{"id": d.get("path", str(i)), "text": d.get("text", ""), "metadata": d}
                            for i, d in enumerate(spine.docs)]
                    _bm25.build_index(docs)
                    _bm25.save()
                    print(f"[BM25] [OK] Build complete: {_bm25.document_count} documents")
            threading.Thread(target=build_bm25, daemon=True).start()
        else:
            print(f"[BM25] Loaded from cache: {_bm25.document_count} documents")
    return _bm25


# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    agent: Optional[str] = None  # Ignored but accepted for compatibility


class Document(BaseModel):
    id: Optional[str] = None
    content: str
    metadata: Optional[Dict[str, Any]] = None


class IngestPayload(BaseModel):
    source: str
    category: str
    documents: List[Document]
    crawl_id: Optional[str] = None


class QueryResult(BaseModel):
    path: str
    text: str
    score: float
    rank: int


class QueryResponse(BaseModel):
    success: bool
    query: str
    results: List[Dict[str, Any]]
    elapsed_ms: float
    doc_count: int


class HealthResponse(BaseModel):
    status: str
    indexing_status: str
    documents: int
    model: str
    dimension: int
    gpu: bool
    uptime: float
    version: str


@app.get("/")
def root():
    """Root endpoint."""
    return {"service": "ScornSpine", "version": "2.0.0", "port": PORT}


@app.get("/health")
def health() -> HealthResponse:
    """Health check endpoint."""
    spine = get_spine()
    h = spine.health()
    return {
        "status": h['status'],
        "indexing_status": _indexing_status,
        "documents": h['documents'],
        "model": h['model'],
        "dimension": h['dimension'],
        "gpu": h['gpu'],
        "uptime": time.time() - _startup_time,
        "version": "2.0.0"
    }


@app.get("/health/live")
def health_live() -> Dict[str, Any]:
    """Live health check - used by triad-main-screen."""
    spine = get_spine()
    return {
        'status': 'healthy' if len(spine.docs) > 0 else 'degraded',
        'documents': len(spine.docs),
        'uptime': time.time() - _startup_time,
        'version': '2.0.0'
    }


@app.get("/health/ready")
def health_ready() -> Dict[str, Any]:
    """Readiness check."""
    spine = get_spine()
    return {
        'ready': len(spine.docs) > 0,
        'documents': len(spine.docs)
    }


@app.get("/metrics")
def get_metrics() -> Dict[str, Any]:
    """
    RT31210: Telemetry endpoint for P50/P95/P99 latency tracking.
    Returns metrics for all tracked operations including:
    - spine_search: Vector similarity search
    - bm25_search: BM25 keyword search
    - hybrid_search: Fusion of vector + BM25
    - embed: Embedding generation time
    - rerank: Cross-encoder reranking time
    """
    # Update health status before returning
    spine = get_spine()
    doc_count = len(spine.docs) if spine else 0
    metrics.update_health("scornspine", "healthy" if doc_count > 0 else "degraded", doc_count=doc_count)

    return metrics.get_all_metrics()


@app.post("/metrics/export")
def export_metrics() -> Dict[str, Any]:
    """Export metrics to file and return snapshot."""
    return metrics.export_metrics()


@app.post("/agent-context")
def agent_context(request: QueryRequest) -> Dict[str, Any]:
    """Agent context endpoint - used by Search-Chronicle.ps1."""
    start = time.time()
    spine = get_spine()

    results = spine.query(request.query, top_k=request.top_k)
    elapsed = (time.time() - start) * 1000

    # Format for agent consumption
    context_parts = []
    for r in results:
        context_parts.append(f"**[{r['path']}]**\n{r['text'][:800]}")

    return {
        'success': True,
        'agent': request.agent or 'unknown',
        'query': request.query,
        'context': "\n\n---\n\n".join(context_parts),
        'sources': [r['path'] for r in results],
        'elapsed_ms': round(elapsed, 2),
        'doc_count': len(spine.docs)
    }


@app.post("/query")
def query(request: QueryRequest) -> QueryResponse:
    """Query endpoint - compatible with old Spine API."""
    global _query_count, _total_query_time_ms, _agent_stats

    # RT8650: Validate query - reject empty
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # RT8650: Cap top_k to prevent abuse
    top_k = min(request.top_k, 50)

    # RT5010: Check Bloom Filter RAM Cache (<1ms)
    cache_key = f"q:{request.query}:{top_k}"
    cached_result = global_cache.get(cache_key)
    if cached_result and not request.query.startswith("!"): # Allow bypass with !
        print(f"[RT5010] Cache HIT for query: {request.query[:30]}...")
        return QueryResponse(
            success=True,
            query=request.query,
            results=cached_result,
            elapsed_ms=0.5,
            doc_count=0 # Status unknown from cache
        )

    try:
        start = time.time()
        spine = get_spine()
        reranker = get_reranker()

        # RT970: Get 3x candidates for reranking (improves relevance)
        candidate_k = min(top_k * 3, 50)  # More candidates for reranker
        candidates = spine.query(request.query.strip(), top_k=candidate_k)

        # RT970: Rerank candidates with cross-encoder
        if reranker.is_available and len(candidates) > top_k:
            results = reranker.rerank(request.query.strip(), candidates, top_k=top_k)
        else:
            results = candidates[:top_k]

        # RT31701: Cloud fallback when local results insufficient
        if qdrant_fallback.is_available and needs_fallback(results):
            print(f"[RT31701] Triggering cloud fallback for: {request.query[:40]}...")
            # Get embedding for cloud search
            vector = spine.embedder.encode([request.query.strip()])[0].tolist()
            cloud_results = qdrant_fallback.search(vector, top_k=top_k)
            if cloud_results:
                results = merge_results(results, cloud_results, top_k=top_k)
                print(f"[RT31701] Merged {len(cloud_results)} cloud + local results")

        elapsed = (time.time() - start) * 1000  # ms

        # RT9300: Track query stats
        _query_count += 1
        _total_query_time_ms += elapsed

        # RT31210: Record metrics for telemetry
        metrics.record("spine_search", elapsed)

        # Track per-agent stats
        agent = request.agent or 'anonymous'
        if agent not in _agent_stats:
            _agent_stats[agent] = {'queries': 0, 'total_ms': 0.0}
        _agent_stats[agent]['queries'] += 1
        _agent_stats[agent]['total_ms'] += elapsed

        # RT5010: Update cache
        global_cache.add(cache_key, results)

        return QueryResponse(
            success=True,
            query=request.query,
            results=results,
            elapsed_ms=round(elapsed, 2),
            doc_count=len(spine.docs)
        )
    except Exception as e:
        print(f"[ScornSpine] Error in /query: {e}")
        # Fallback to simple search if reranking/stats fail
        try:
            spine = get_spine()
            results = spine.query(request.query.strip(), top_k=top_k)
            return QueryResponse(
                success=True,
                query=request.query,
                results=results,
                elapsed_ms=0.0,
                doc_count=len(spine.docs)
            )
        except Exception as inner_e:
            raise HTTPException(status_code=500, detail=f"Query failed: {str(inner_e)}")


@app.post("/context")
def context(request: QueryRequest) -> Dict[str, Any]:
    """Context endpoint - returns formatted context string."""
    start = time.time()
    spine = get_spine()

    results = spine.query(request.query, top_k=request.top_k)
    elapsed = (time.time() - start) * 1000

    # Format as context string (matches old Spine behavior)
    context_parts = []
    for r in results:
        context_parts.append(f"[{r['path']}]\n{r['text'][:500]}")

    return {
        'success': True,
        'query': request.query,
        'context': "\n\n---\n\n".join(context_parts),
        'sources': [r['path'] for r in results],
        'elapsed_ms': round(elapsed, 2)
    }


@app.post("/search")
def search(request: QueryRequest) -> Dict[str, Any]:
    """RT2951: Fast embedding search - sub-100ms, no LLM synthesis.

    This is the endpoint referenced in critical.instructions.md.
    Returns raw results without LLM processing for speed.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    start = time.time()
    spine = get_spine()

    # Fast search - just embeddings + retrieval, no LLM
    results = spine.query(request.query.strip(), top_k=min(request.top_k, 20))
    elapsed = (time.time() - start) * 1000

    # RT31210: Record metrics
    metrics.record("fast_search", elapsed)

    # RT31200: Auto-update LatentGrid if agent is specified
    if request.agent and request.agent.lower() in ['halo', 'jonah', 'vera', 'halojinix']:
        try:
            grid = get_latent_grid()
            grid.update_position(request.agent.lower(), request.query, source="search")
        except Exception:
            pass  # Don't fail search if grid update fails

        # RT31210: Also track coherence telemetry
        try:
            telemetry = get_coherence_telemetry()
            telemetry.update(request.agent.lower(), request.query)
        except Exception:
            pass  # Don't fail search if telemetry fails

    return {
        'success': True,
        'query': request.query,
        'results': results,
        'count': len(results),
        'latency_ms': round(elapsed, 2),
        'agent': request.agent
    }


@app.post("/ingest")
async def ingest_documents(payload: IngestPayload):
    """
    RT1058: Ingest batch of documents from external sources.
    """
    start = time.time()
    spine = get_spine()

    docs_to_add = []
    for doc in payload.documents:
        docs_to_add.append({
            "path": doc.id or f"{payload.source}_{int(time.time())}_{os.urandom(4).hex()}",
            "text": doc.content,
            "category": payload.category,
            "metadata": {
                **(doc.metadata or {}),
                "source": payload.source,
                "ingested_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }
        })

    # RT9600: Add to spine
    spine.docs.extend(docs_to_add)

    # For small batches, update vector index and BM25 immediately
    if len(docs_to_add) < 100:
        spine.build_index()
        bm25 = get_bm25()
        if bm25.is_ready:
           bm25_docs = [{"id": d.get("path", str(i)), "text": d.get("text", ""), "metadata": d}
                            for i, d in enumerate(spine.docs)]
           bm25.build_index(bm25_docs)

    elapsed = (time.time() - start) * 1000

    return {
        "success": True,
        "indexed_count": len(docs_to_add),
        "total_documents": len(spine.docs),
        "latency_ms": round(elapsed, 2)
    }


@app.post("/refresh")
def refresh() -> Dict[str, Any]:
    """Refresh index from workspace - RT9200: Skip if already refreshing."""
    global _spine, _refreshing

    if _refreshing:
        return {
            'success': False,
            'message': 'Refresh already in progress',
            'documents': len(_spine.docs) if _spine else 0
        }

    _refreshing = True
    start = time.time()

    try:
        # RT8800: Invalidate BM25 cache to force rebuild after refresh
        global _bm25
        _bm25 = None

        # Create fresh instance
        _spine = ScornSpine()
        count = _spine.load_workspace("F:/primewave-engine")
        elapsed = time.time() - start

        return {
            'success': True,
            'documents': count,
            'elapsed_seconds': round(elapsed, 2)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'documents': len(_spine.docs) if _spine else 0
        }
    finally:
        _refreshing = False


@app.post("/index")
def index_all() -> Dict[str, Any]:
    """Alias for refresh - indexes all documents."""
    return refresh()


@app.post("/dedupe")
def dedupe() -> Dict[str, Any]:
    """
    RT970: Remove duplicate documents from the index.

    Uses content hash to identify exact duplicates and removes them,
    keeping only unique documents. This improves search quality by
    preventing the same content from appearing multiple times in results.
    """
    spine = get_spine()
    start = time.time()

    try:
        result = spine.dedupe_index()
        elapsed = time.time() - start

        return {
            'success': True,
            'before': result['before'],
            'after': result['after'],
            'removed': result['removed'],
            'elapsed_seconds': round(elapsed, 2)
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# -------------------------------------------------------------------------------
# RT9300: PARITY ENDPOINTS - Ported from old Spine for full compatibility
# -------------------------------------------------------------------------------

@app.get("/stats")
def stats() -> Dict[str, Any]:
    """RT9300: Detailed statistics with agent telemetry."""
    spine = get_spine()
    h = spine.health()

    avg_query_ms = (_total_query_time_ms / _query_count) if _query_count > 0 else 0

    return {
        'service': 'ScornSpine',
        'version': '2.1.0',  # RT31701: Bumped for cloud fallback
        'port': PORT,
        'status': h['status'],
        'documents': h['documents'],
        'model': h['model'],
        'dimension': h['dimension'],
        'gpu': h['gpu'],
        'uptime_seconds': round(time.time() - _startup_time, 2),
        'query_stats': {
            'total_queries': _query_count,
            'avg_query_ms': round(avg_query_ms, 2),
            'total_time_ms': round(_total_query_time_ms, 2)
        },
        'agent_stats': _agent_stats,
        'cache': {
            'type': 'faiss-flat',
            'indexed': len(spine.docs),
            'status': 'active'
        },
        'reranker': get_reranker().get_stats(),  # RT970: Reranker status
        'cloud_fallback': {  # RT31701: Cloud fallback stats
            'enabled': qdrant_fallback.is_available,
            'stats': qdrant_fallback.stats
        }
    }


@app.get("/agent-stats")
def agent_stats_endpoint() -> Dict[str, Any]:
    """RT9400: Agent-specific query statistics."""
    # Calculate per-agent averages
    detailed = {}
    for agent, data in _agent_stats.items():
        queries = data.get('queries', 0)
        total_ms = data.get('total_ms', 0.0)
        detailed[agent] = {
            'queries': queries,
            'total_ms': round(total_ms, 2),
            'avg_ms': round(total_ms / queries, 2) if queries > 0 else 0
        }

    return {
        'agents': detailed,
        'total_agents': len(_agent_stats),
        'cache': {
            'type': 'faiss-flat',
            'status': 'active'
        }
    }


@app.get("/recent")
def recent_ingestions(agent: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    """RT9400: Recent ingestions with optional agent filter."""
    from pathlib import Path
    import json

    ingest_dir = Path("F:/primewave-engine/haloscorn/scornspine/data/ingested")
    if not ingest_dir.exists():
        return {'success': True, 'recent': [], 'count': 0}

    limit = min(limit, 50)  # Cap at 50
    agent_filter = agent.lower() if agent else None

    # RT9400: Optimization - sort files by mtime first, only read top N
    files = list(ingest_dir.glob('*.json'))
    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

    ingestions = []
    checked = 0
    max_check = limit * 3  # Check 3x limit to allow for filtering

    for ingest_file in files:
        if len(ingestions) >= limit or checked >= max_check:
            break
        checked += 1

        try:
            data = json.loads(ingest_file.read_text())
            file_agent = data.get('agent', '').lower()

            if agent_filter and file_agent != agent_filter:
                continue

            messages = data.get('messages', [])
            preview = messages[-1].get('content', '')[:100] if messages else ''

            ingestions.append({
                'ingestion_id': ingest_file.stem,
                'agent': file_agent,
                'session_id': data.get('session_id', ''),
                'timestamp': data.get('timestamp', ''),
                'message_count': len(messages),
                'indexed': data.get('indexed', False),
                'preview': preview
            })
        except Exception:
            continue

    return {
        'success': True,
        'recent': ingestions,
        'count': len(ingestions),
        'total_available': len(files),
        'filter': agent_filter or 'all'
    }


@app.get("/cache/stats")
def cache_stats() -> Dict[str, Any]:
    """RT9300: Cache performance metrics."""
    spine = get_spine()
    return {
        'type': 'faiss-flat',
        'indexed_documents': len(spine.docs),
        'dimension': spine.dim,
        'status': 'active',
        'memory_mb': round(len(spine.docs) * spine.dim * 4 / 1024 / 1024, 2)
    }


class CapabilitySearchRequest(BaseModel):
    query: str
    top_k: int = 10


class DoubleHelixRequest(BaseModel):
    query: str
    agent: str = "halojinix"
    top_k: int = 5


@app.post("/double-helix")
async def double_helix_query(request: DoubleHelixRequest) -> Dict[str, Any]:
    """RT5010: Synthesis endpoint - fuses Mem0 and ScornSpine."""
    import httpx
    import asyncio

    start = time.time()
    query = request.query.strip()
    top_k = request.top_k

    # 1. Check Bloom Cache
    cache_key = f"dh:{query}:{request.agent}:{top_k}"
    cached = global_cache.get(cache_key)
    if cached and not query.startswith("!"):
        return {
            'success': True,
            'query': query,
            'results': cached,
            'latency_ms': 0.5,
            'cached': True
        }

    # Remove ! if present for searching
    clean_query = query[1:] if query.startswith("!") else query

    # 2. Truly Parallel Fetch (RT5010 IMPROVED)
    async with httpx.AsyncClient() as client:
        # Define tasks
        async def fetch_mem0():
            try:
                resp = await client.post(
                    "http://127.0.0.1:7790/memory/search",
                    json={"agent_id": request.agent, "query": clean_query, "limit": top_k},
                    timeout=10.0 # Bumped timeout
                )
                return resp.json().get("results", [])
            except Exception as e:
                print(f"[DoubleHelix] Mem0 failed: {e}")
                return []

        # Execute in parallel
        mem0_data, spine_results = await asyncio.gather(
            fetch_mem0(),
            hybrid_search(HybridRequest(query=clean_query, top_k=top_k))
        )

    # 3. Fusion
    fused = []
    # Process Mem0
    for r in mem0_data:
        # RT5010: Mem0 returns distances (smaller is better)
        # Convert to similarity score (higher is better)
        dist = r.get("score", 0)
        base_score = 1.0 / (1.0 + dist) if dist >= 0 else 0.5

        # RT5010: High weight for memories to ensure they stay relevant in hybrid fusion
        # Personal memories get the highest boost
        is_shared = r.get("metadata", {}).get("bucket") == "shared"
        weight = 8.0 if is_shared else 12.0 # Significant boost

        fused.append({
            "source": "Mem0",
            "text": r.get("memory", ""),
            "path": f"memory:{r.get('id', 'unknown')}",
            "score": base_score * weight, # Boost Mem0 results
            "metadata": r.get("metadata", {})
        })
    # Process Spine
    for r in spine_results.get("results", []):
        # RT8600: Use fused_score if available, else original score
        spine_score = r.get("fused_score", r.get("score", 0))

        fused.append({
            "source": "ScornSpine",
            "text": r.get("text", ""),
            "path": r.get("path", "unknown"),
            "score": spine_score,
            "metadata": r.get("metadata", {})
        })

    # Sort and re-rank (higher is better)
    fused.sort(key=lambda x: x['score'], reverse=True)
    results = fused[:top_k]

    elapsed = (time.time() - start) * 1000

    # Update cache
    global_cache.add(cache_key, results)

    return {
        'success': True,
        'query': query,
        'results': results,
        'latency_ms': round(elapsed, 2),
        'cached': False
    }


class VoteRequest(BaseModel):
    memory_id: str
    vote: str # "up" or "down"


@app.post("/memory/vote")
def vote_memory(request: VoteRequest) -> Dict[str, Any]:
    """RT5010: Phase 3 - Vote on memory relevance."""
    import sqlite3
    db_path = "F:/primewave-engine/data/memory_votes.db"

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Upsert
        cursor.execute("INSERT OR IGNORE INTO memory_votes (memory_id) VALUES (?)", (request.memory_id,))
        if request.vote == "up":
            cursor.execute("UPDATE memory_votes SET upvotes = upvotes + 1, last_voted_at = ? WHERE memory_id = ?",
                           (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), request.memory_id))
        else:
            cursor.execute("UPDATE memory_votes SET downvotes = downvotes + 1, last_voted_at = ? WHERE memory_id = ?",
                           (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), request.memory_id))

        conn.commit()
        conn.close()

        return {"success": True, "memory_id": request.memory_id, "vote": request.vote}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/memory/votes")
def get_votes() -> Dict[str, Any]:
    """Get all voting results."""
    import sqlite3
    db_path = "F:/primewave-engine/data/memory_votes.db"

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM memory_votes ORDER BY upvotes DESC")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return {"success": True, "votes": rows}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/capability-search")
def capability_search(request: CapabilitySearchRequest) -> Dict[str, Any]:
    """RT3820: Search tools, skills, scripts, APIs by keyword."""
    from pathlib import Path
    import json as json_lib

    if not request.query:
        raise HTTPException(status_code=400, detail="query is required")

    manifest_path = Path("F:/primewave-engine/data/capability-manifest.json")
    if not manifest_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Capability manifest not found. Run Generate-CapabilityManifest.ps1 first."
        )

    manifest = json_lib.loads(manifest_path.read_text())

    query_lower = request.query.lower()
    query_words = set(query_lower.split())
    results = []

    # Search tools
    for tool in manifest.get('tools', []):
        name = tool.get('name', tool.get('type', '')).lower()
        desc = tool.get('description', '').lower()
        text = f"{name} {desc}"
        score = sum(1 for word in query_words if word in text)
        if score > 0:
            results.append({
                'type': 'tool',
                'name': tool.get('name', tool.get('type', '')),
                'description': tool.get('description', ''),
                'score': score
            })

    # Search skills
    for skill in manifest.get('skills', []):
        name = skill.get('name', '').lower()
        trigger = skill.get('trigger', '').lower()
        text = f"{name} {trigger}"
        score = sum(1 for word in query_words if word in text)
        if score > 0:
            results.append({
                'type': 'skill',
                'name': skill.get('name', ''),
                'path': skill.get('path', ''),
                'trigger': skill.get('trigger', ''),
                'score': score
            })

    # Search scripts
    for script in manifest.get('scripts', []):
        name = script.get('name', '').lower()
        synopsis = script.get('synopsis', '').lower()
        text = f"{name} {synopsis}"
        score = sum(1 for word in query_words if word in text)
        if score > 0:
            results.append({
                'type': 'script',
                'name': script.get('name', ''),
                'path': script.get('path', ''),
                'synopsis': script.get('synopsis', ''),
                'score': score
            })

    # Sort by score descending and limit
    results.sort(key=lambda x: x['score'], reverse=True)
    results = results[:request.top_k]

    return {
        'success': True,
        'query': request.query,
        'results': results,
        'count': len(results)
    }


class VectorSearchRequest(BaseModel):
    vector: List[float]
    k: int = 5


@app.post("/search_vector")
def search_vector(request: VectorSearchRequest) -> Dict[str, Any]:
    """RT4400: Search by raw vector - COCONUT latent reasoning support."""
    import numpy as np

    if not request.vector or len(request.vector) == 0:
        raise HTTPException(status_code=400, detail="vector is required")

    start = time.time()
    spine = get_spine()

    # Convert to numpy array
    query_vector = np.array(request.vector, dtype=np.float32).reshape(1, -1)

    # Check dimension match
    if query_vector.shape[1] != spine.dim:
        raise HTTPException(
            status_code=400,
            detail=f"Vector dimension {query_vector.shape[1]} doesn't match index dimension {spine.dim}"
        )

    # Search in FAISS index
    k = min(request.k, 50)
    distances, indices = spine.index.search(query_vector, k)

    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx >= 0 and idx < len(spine.docs):
            doc = spine.docs[idx]
            results.append({
                'id': str(idx),
                'score': float(1.0 / (1.0 + dist)),  # Convert distance to score
                'distance': float(dist),
                'path': doc.get('path', 'unknown'),
                'text': doc.get('text', '')[:500],
                'rank': i + 1
            })

    elapsed = (time.time() - start) * 1000

    return {
        'success': True,
        'results': results,
        'count': len(results),
        'latency_ms': round(elapsed, 2),
        'dimension': spine.dim
    }


class EmbedRequest(BaseModel):
    text: str


@app.post("/embed")
def embed(request: EmbedRequest) -> Dict[str, Any]:
    """RT4400: Get raw embedding vector for text - COCONUT latent reasoning."""
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    start = time.time()
    spine = get_spine()

    # Get embedding from model
    vector = spine.embedder.encode([request.text.strip()])[0]

    elapsed = (time.time() - start) * 1000

    return {
        'success': True,
        'vector': vector.tolist(),
        'dimension': len(vector),
        'latency_ms': round(elapsed, 2)
    }


class LogRequest(BaseModel):
    agent: str = "unknown"
    role: str = "user"
    content: str


# Simple in-memory log for now (could persist to file)
_conversation_log: List[Dict[str, Any]] = []


@app.post("/log")
def log_conversation(request: LogRequest) -> Dict[str, Any]:
    """RT9300: Log conversation turn - Chronicle integration."""
    if not request.content:
        raise HTTPException(status_code=400, detail="content is required")

    entry = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'agent': request.agent,
        'role': request.role,
        'content': request.content,
        'id': len(_conversation_log)
    }

    _conversation_log.append(entry)

    # Keep only last 1000 entries in memory
    if len(_conversation_log) > 1000:
        _conversation_log.pop(0)

    return {
        'success': True,
        'entry': entry
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RT4400/RT31205: COCONUT - Continuous Thought in Latent Space
# Restored after accidental removal in RT30104 (532bbeb0)
# ═══════════════════════════════════════════════════════════════════════════════

class CoconutRequest(BaseModel):
    query: str
    top_k: int = 5
    max_steps: int = 3
    verbose: bool = False


@app.post("/coconut")
def coconut_reasoning(request: CoconutRequest) -> Dict[str, Any]:
    """RT4400: COCONUT latent space reasoning.

    Implements continuous thought iteration using TextCoconutReasoner:
    1. Initial query → search results
    2. Extract keywords from results
    3. Refine query and repeat
    4. Converge when results stabilize

    This provides latent-space-like reasoning without full vector operations.
    For true vector COCONUT, use /embed + /search_vector directly.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    start = time.time()
    spine = get_spine()

    # Create search function for TextCoconutReasoner
    def search_fn(query: str, k: int) -> List[Dict[str, Any]]:
        return spine.query(query.strip(), top_k=k)

    # Import and use TextCoconutReasoner
    try:
        from haloscorn.latent_space.coconut import TextCoconutReasoner

        reasoner = TextCoconutReasoner(
            search_fn=search_fn,
            max_steps=min(request.max_steps, 10),  # Cap at 10
            convergence_threshold=0.8
        )

        result = reasoner.reason(
            query=request.query.strip(),
            top_k=min(request.top_k, 20),
            verbose=request.verbose
        )

        elapsed = (time.time() - start) * 1000

        return {
            'success': True,
            'query': request.query,
            'final_query': result.get('final_query', request.query),
            'results': result.get('results', []),
            'steps': result.get('steps', []),
            'converged': result.get('converged', False),
            'total_steps': result.get('total_steps', 0),
            'latency_ms': round(elapsed, 2),
            'doc_count': len(spine.docs)
        }
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"COCONUT module not available: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"COCONUT reasoning failed: {e}"
        )


# -------------------------------------------------------------------------------
# RT31200: RERANK ENDPOINT - Cross-Encoder Reranking
# -------------------------------------------------------------------------------

class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: int = 10


@app.post("/rerank")
def rerank_documents(request: RerankRequest) -> Dict[str, Any]:
    """RT31200: Cross-encoder reranking endpoint.

    Uses ms-marco-MiniLM-L-6-v2 to rerank documents by relevance to query.
    More accurate than bi-encoder but slower - use for final result refinement.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    if not request.documents:
        raise HTTPException(status_code=400, detail="documents list required")

    start = time.time()
    reranker = get_reranker()

    if not reranker.is_available:
        raise HTTPException(status_code=503, detail="Cross-encoder not available")

    # Convert documents list to result format expected by reranker
    results = [{"text": doc, "index": i} for i, doc in enumerate(request.documents)]

    reranked = reranker.rerank(
        request.query.strip(),
        results,
        top_k=min(request.top_k, len(request.documents))
    )

    elapsed = (time.time() - start) * 1000

    return {
        'success': True,
        'query': request.query,
        'results': reranked,
        'count': len(reranked),
        'latency_ms': round(elapsed, 2),
        'model': reranker.model_name,
        'device': reranker.get_stats().get('device', 'unknown')
    }


# -------------------------------------------------------------------------------
# RT31200: LATENTGRID (UNIFIED GRAPH) ENDPOINTS
# -------------------------------------------------------------------------------

# RT31200: UnifiedGraph singleton (uses LOCAL Qdrant - Docker on port 6333)
_unified_graph = None

def get_unified_graph():
    """Get or create UnifiedGraph singleton using LOCAL Qdrant (Docker).

    RT31200: HALO brought Qdrant v1.16.3 online via Docker.
    Falls back to Cloud if QDRANT_URL is set.
    """
    global _unified_graph
    if _unified_graph is None:
        from qdrant_client import QdrantClient
        from haloscorn.latent_space.unified_graph import UnifiedGraph

        # Check for Cloud credentials first, then use local Docker
        qdrant_url = os.getenv('QDRANT_URL')
        qdrant_api_key = os.getenv('QDRANT_API_KEY')

        if qdrant_url and qdrant_api_key:
            # Cloud mode
            client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30.0)
        else:
            # Local Docker mode (HALO's infrastructure)
            local_host = os.getenv('QDRANT_LOCAL_HOST', 'localhost')
            local_port = int(os.getenv('QDRANT_LOCAL_PORT', '6333'))
            client = QdrantClient(host=local_host, port=local_port, timeout=30.0)

        _unified_graph = UnifiedGraph(client=client)

    return _unified_graph


@app.get("/graph/stats")
def graph_stats() -> Dict[str, Any]:
    """RT31200: Get LatentGrid (Unified Graph) statistics."""
    try:
        graph = get_unified_graph()
        stats = graph.get_stats()

        return {
            'success': True,
            'stats': stats
        }
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"UnifiedGraph not available: {e}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Qdrant not configured: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph stats failed: {e}")


class GraphQueryRequest(BaseModel):
    perspective: str = "jonah"  # halo, vera, jonah
    node_type: Optional[str] = None
    limit: int = 20


@app.post("/graph/query")
def graph_query(request: GraphQueryRequest) -> Dict[str, Any]:
    """RT31200: Query LatentGrid from agent perspective.

    Each agent sees the graph differently:
    - HALO: Code changes, dependencies, build impact
    - VERA: Active blockers, cross-agent, recent activity
    - JONAH: Open threads, root causes, low confidence items
    """
    if request.perspective not in ["halo", "vera", "jonah"]:
        raise HTTPException(status_code=400, detail="perspective must be halo, vera, or jonah")

    start = time.time()

    try:
        graph = get_unified_graph()
        nodes = graph.query_by_perspective(
            request.perspective,
            node_type=request.node_type,
            limit=min(request.limit, 100)
        )

        # Convert nodes to dict for JSON serialization
        results = []
        for node in nodes:
            results.append({
                'id': node.id,
                'type': node.type,
                'content': node.content,
                'created_by': node.created_by,
                'tags': node.tags,
                'perspective_data': node.get_perspective(request.perspective)
            })

        elapsed = (time.time() - start) * 1000

        return {
            'success': True,
            'perspective': request.perspective,
            'node_type': request.node_type,
            'results': results,
            'count': len(results),
            'latency_ms': round(elapsed, 2)
        }
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"UnifiedGraph not available: {e}")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"Qdrant not configured: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph query failed: {e}")


# -------------------------------------------------------------------------------
# RT31200: LATENT GRID - REAL-TIME AGENT SEMANTIC POSITIONING
# (20-round Triad Epic Forge consensus - JONAH 2026-01-07)
# -------------------------------------------------------------------------------

# LatentGrid singleton
_latent_grid = None


def get_latent_grid():
    """Get or create LatentGrid singleton."""
    global _latent_grid
    if _latent_grid is None:
        try:
            from haloscorn.latent_space.latent_grid import LatentGrid
            # Use direct embedder to avoid HTTP roundtrip (fixes deadlock)
            spine = get_spine()
            _latent_grid = LatentGrid(embedder=spine.embedder)
            print(f"[LatentGrid] Initialized with {_latent_grid.get_stats()['total_agents']} agents (direct embedder)")
        except Exception as e:
            print(f"[LatentGrid] Failed to initialize: {e}")
            raise
    return _latent_grid


class GridUpdateRequest(BaseModel):
    agent_id: str
    context: str
    source: str = "explicit"  # search, live-feed, signal, explicit


class GridProximityRequest(BaseModel):
    agent1: str
    agent2: str


class GridNearbyRequest(BaseModel):
    agent_id: str
    threshold: float = 0.5


@app.post("/grid/update")
def grid_update(request: GridUpdateRequest) -> Dict[str, Any]:
    """RT31200: Update agent position in latent semantic space.

    Called when agent:
    - Searches (auto-triggered from /search)
    - Posts to Signal
    - Updates live-feed context
    """
    if not request.agent_id or not request.context:
        raise HTTPException(status_code=400, detail="agent_id and context required")

    start = time.time()

    try:
        grid = get_latent_grid()
        state = grid.update_position(
            request.agent_id.lower(),
            request.context,
            request.source
        )
        elapsed = (time.time() - start) * 1000

        return {
            'success': True,
            'agent_id': state.agent_id,
            'focus_topic': state.focus_topic,
            'velocity_magnitude': float(np.linalg.norm(state.velocity)),
            'latency_ms': round(elapsed, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grid update failed: {e}")


@app.get("/grid/positions")
def grid_positions(include_stale: bool = True) -> Dict[str, Any]:
    """RT31200: Get all agent positions in latent space."""
    try:
        grid = get_latent_grid()
        positions = grid.get_all_positions(include_stale=include_stale)

        return {
            'success': True,
            'agents': positions,
            'stats': grid.get_stats()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grid positions failed: {e}")


@app.get("/grid/stats")
def grid_stats() -> Dict[str, Any]:
    """RT31200: Get LatentGrid statistics."""
    try:
        grid = get_latent_grid()
        return {
            'success': True,
            'stats': grid.get_stats()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grid stats failed: {e}")


@app.post("/grid/proximity")
def grid_proximity(request: GridProximityRequest) -> Dict[str, Any]:
    """RT31200: Get cosine distance between two agents.

    Returns distance in [0, 2]:
    - 0.0 = Identical focus (working on same thing)
    - 0.5 = Similar focus (nearby topics)
    - 1.0 = Orthogonal (unrelated)
    - 2.0 = Opposite (contradiction)
    """
    try:
        grid = get_latent_grid()
        distance = grid.get_proximity(request.agent1.lower(), request.agent2.lower())

        # Interpret distance
        if distance == float('inf'):
            interpretation = "unknown (one or both agents not tracked)"
        elif distance < 0.2:
            interpretation = "highly aligned - working on same topic"
        elif distance < 0.5:
            interpretation = "similar focus - related work"
        elif distance < 1.0:
            interpretation = "different focus - parallel tracks"
        else:
            interpretation = "divergent - unrelated or opposing"

        return {
            'success': True,
            'agent1': request.agent1,
            'agent2': request.agent2,
            'distance': distance,
            'interpretation': interpretation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grid proximity failed: {e}")


@app.post("/grid/nearby")
def grid_nearby(request: GridNearbyRequest) -> Dict[str, Any]:
    """RT31200: Find agents working on similar topics."""
    try:
        grid = get_latent_grid()
        nearby = grid.find_nearby_agents(request.agent_id.lower(), threshold=request.threshold)

        return {
            'success': True,
            'agent_id': request.agent_id,
            'threshold': request.threshold,
            'nearby_agents': [{'agent': a, 'distance': d} for a, d in nearby],
            'count': len(nearby)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grid nearby failed: {e}")


# -------------------------------------------------------------------------------
# RT31210: COHERENCE TELEMETRY - REAL-TIME AGENT ALIGNMENT MONITORING
# -------------------------------------------------------------------------------

_coherence_telemetry = None


def get_coherence_telemetry():
    """Get or create CoherenceTelemetry singleton."""
    global _coherence_telemetry
    if _coherence_telemetry is None:
        try:
            from haloscorn.telemetry.coherence import CoherenceTelemetry
            spine = get_spine()
            _coherence_telemetry = CoherenceTelemetry(embedder=spine.embedder)
            print("[CoherenceTelemetry] Initialized (direct embedder)")
        except Exception as e:
            print(f"[CoherenceTelemetry] Failed to initialize: {e}")
            raise
    return _coherence_telemetry


class CoherenceTrackRequest(BaseModel):
    agent_id: str
    context: str


@app.post("/coherence/track")
def coherence_track(request: CoherenceTrackRequest) -> Dict[str, Any]:
    """RT31210: Track agent context and get coherence score.

    Returns:
        coherence: 0-1 score (higher = more aligned team)
        drifting: True if team is diverging
        outlier: Which agent is furthest from the group
    """
    if not request.agent_id or not request.context:
        raise HTTPException(status_code=400, detail="agent_id and context required")

    start = time.time()

    try:
        telemetry = get_coherence_telemetry()
        score = telemetry.update(request.agent_id.lower(), request.context)
        summary = telemetry.summary()
        elapsed = (time.time() - start) * 1000

        return {
            'success': True,
            'coherence': round(summary['coherence'], 3),
            'drifting': bool(summary['drifting']),  # Convert numpy.bool_ to Python bool
            'outlier': summary['outlier'],
            'agents_tracked': summary['agents_tracked'],
            'latency_ms': round(elapsed, 2)
        }
    except Exception as e:
        import traceback
        print(f"[CoherenceTelemetry] ERROR tracking {request.agent_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Coherence tracking failed: {e}")


@app.get("/coherence")
def coherence_status() -> Dict[str, Any]:
    """RT31210: Get current team coherence status."""
    try:
        telemetry = get_coherence_telemetry()
        summary = telemetry.summary()

        return {
            'success': True,
            'coherence': round(summary['coherence'], 3),
            'coherence_pct': f"{summary['coherence']:.0%}",
            'drifting': bool(summary['drifting']),  # Convert numpy.bool_ to Python bool
            'outlier': summary['outlier'],
            'agents_tracked': summary['agents_tracked'],
            'measurements': int(summary['measurements'])  # Ensure int
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coherence status failed: {e}")


@app.get("/coherence/history")
def coherence_history(hours: int = 24) -> Dict[str, Any]:
    """RT31210: Get coherence history from database."""
    try:
        telemetry = get_coherence_telemetry()
        history = telemetry.get_history(hours=hours)

        return {
            'success': True,
            'hours': hours,
            'snapshots': [
                {'timestamp': h[0], 'coherence': h[1], 'drift': h[2], 'outlier': h[3]}
                for h in history
            ],
            'count': len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Coherence history failed: {e}")


# -------------------------------------------------------------------------------
# RT8600: PERFECTIFY - BM25 AND HYBRID SEARCH ENDPOINTS
# -------------------------------------------------------------------------------

class BM25Request(BaseModel):
    query: str
    top_k: int = 10


@app.post("/bm25")
def bm25_search(request: BM25Request) -> Dict[str, Any]:
    """RT8600: BM25 keyword search - exact term matching.

    BM25 excels at:
    - Technical terms (ADR-0050, RT263)
    - Proper nouns (HALO, VERA, JONAH)
    - Specific file paths
    - Version numbers
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    start = time.time()
    bm25 = get_bm25()

    if not bm25.is_ready:
        raise HTTPException(status_code=503, detail="BM25 index not ready")

    results = bm25.search(request.query.strip(), top_k=min(request.top_k, 50))
    elapsed = (time.time() - start) * 1000

    return {
        'success': True,
        'query': request.query,
        'results': results,
        'count': len(results),
        'latency_ms': round(elapsed, 2),
        'index_size': bm25.document_count
    }


class HybridRequest(BaseModel):
    query: str
    top_k: int = 10
    vector_weight: float = 0.7
    bm25_weight: float = 0.3


@app.post("/hybrid")
async def hybrid_search(request: HybridRequest) -> Dict[str, Any]:
    """RT8600: Hybrid search - combines vector + BM25 with Reciprocal Rank Fusion.

    Best of both worlds:
    - Vector: semantic similarity, meaning-based
    - BM25: exact keywords, technical terms
    - RRF: robust rank-based fusion (scale-independent)
    """
    import asyncio
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    start = time.time()
    spine = get_spine()
    bm25 = get_bm25()
    reranker = get_reranker()

    query = request.query.strip()
    top_k = min(request.top_k, 50)

    # RT5010: Check Bloom Filter RAM Cache (<1ms)
    cache_key = f"h:{query}:{top_k}:{request.vector_weight}:{request.bm25_weight}"
    cached_result = global_cache.get(cache_key)
    if cached_result and not query.startswith("!"):
        print(f"[RT5010] Cache HIT for hybrid: {query[:30]}...")
        return {
            'success': True,
            'query': query,
            'results': cached_result,
            'count': len(cached_result),
            'latency_ms': 0.5,
            'cached': True
        }

    # Get 2x candidates from each source for better fusion
    candidate_k = top_k * 2

    # Parallel candidates fetch using threads to avoid blocking the event loop
    # vector_task = spine.query(query, top_k=candidate_k)
    # bm25_task = bm25.search(query, top_k=candidate_k)

    vector_results, bm25_results = await asyncio.gather(
        asyncio.to_thread(spine.query, query, top_k=candidate_k),
        asyncio.to_thread(bm25.search, query, top_k=candidate_k) if bm25.is_ready else asyncio.to_thread(lambda: [])
    )

    # Convert to fusion format
    vector_for_fusion = [{"id": r.get("path", str(i)), **r} for i, r in enumerate(vector_results)]

    # Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion(
        vector_results=vector_for_fusion,
        bm25_results=bm25_results,
        vector_weight=request.vector_weight,
        bm25_weight=request.bm25_weight
    )[:top_k]

    # Optional reranking
    if reranker.is_available and len(fused) > 0:
        # Convert to reranker format
        for r in fused:
            if 'text' not in r and 'text_preview' in r:
                r['text'] = r['text_preview']
        # to_thread for CPU bound reranking
        fused = await asyncio.to_thread(reranker.rerank, query, fused, top_k=top_k)

    elapsed = (time.time() - start) * 1000

    # RT5010: Update cache
    global_cache.add(cache_key, fused)

    return {
        'success': True,
        'query': query,
        'results': fused,
        'count': len(fused),
        'latency_ms': round(elapsed, 2),
        'sources': {
            'vector_candidates': len(vector_results),
            'bm25_candidates': len(bm25_results),
            'weights': {'vector': request.vector_weight, 'bm25': request.bm25_weight}
        }
    }


# RT8800: Graceful shutdown handler
import atexit
import signal

_server_started = False  # RT9700: Only shutdown if server actually started

def _graceful_shutdown():
    """Save state before shutdown."""
    global _spine, _bm25, _server_started
    if not _server_started:
        return  # RT9700: Don't run shutdown if server never started
    print("[ScornSpine] Graceful shutdown initiated...")
    if _bm25:
        try:
            _bm25.save()
            print("[ScornSpine] BM25 index saved")
        except Exception as e:
            print(f"[ScornSpine] BM25 save failed: {e}")
    print("[ScornSpine] Shutdown complete")

atexit.register(_graceful_shutdown)

def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT."""
    _graceful_shutdown()
    sys.exit(0)

# Register signal handlers (Windows-safe)
try:
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
except (ValueError, OSError):
    pass  # May fail on Windows in certain contexts


if __name__ == "__main__":
    _server_started = True  # RT9700: Must be first to set module-level var

    print("=" * 50)
    print("ScornSpine Server")
    print(f"Port: {PORT}")
    print("=" * 50)

    # Pre-initialize spine and BM25
    get_spine()
    get_bm25()  # RT8600: Initialize BM25 index at startup

    # Run server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=PORT,
        log_level="info"
    )
