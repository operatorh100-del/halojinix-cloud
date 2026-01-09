"""
ScornSpine Metrics Collector
RT31210: Telemetry infrastructure for P50/P95/P99 latency tracking

Usage:
    from metrics import MetricsCollector, metrics
    
    # Record latency
    with metrics.timer("spine_search"):
        results = spine.query(...)
    
    # Or manually
    metrics.record("spine_search", latency_ms=45.2)
    
    # Get percentiles
    stats = metrics.get_percentiles("spine_search")
    # {"p50": 42, "p95": 95, "p99": 180, "count": 500}
    
    # Export to file
    metrics.export_metrics()
"""

import time
import json
import threading
from collections import defaultdict, deque
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import contextmanager


class MetricsCollector:
    """
    Thread-safe metrics collector with sliding window percentile tracking.
    RT31210: Part of EPIC FORGE INFRA dispatch.
    """
    
    def __init__(
        self, 
        window_size: int = 1000,
        metrics_dir: Path = Path("F:/primewave-engine/data/metrics")
    ):
        self.window_size = window_size
        self.metrics_dir = metrics_dir
        self.metrics_path = metrics_dir / "current.json"
        self.history_dir = metrics_dir / "history"
        
        # Thread-safe sliding windows for each operation
        self._windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._lock = threading.Lock()
        
        # Health tracking
        self._health: Dict[str, Dict[str, Any]] = {}
        self._error_counts: Dict[str, int] = defaultdict(int)
        self._request_count = 0
        self._start_time = time.time()
        
        # Ensure directories exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)
    
    def record(self, operation: str, latency_ms: float) -> None:
        """Record a latency measurement for an operation."""
        with self._lock:
            self._windows[operation].append(latency_ms)
            self._request_count += 1
    
    def record_error(self, operation: str) -> None:
        """Record an error for an operation."""
        with self._lock:
            self._error_counts[operation] += 1
    
    def update_health(self, service: str, status: str, **extra) -> None:
        """Update health status for a service."""
        with self._lock:
            self._health[service] = {
                "status": status,
                "last_check": datetime.utcnow().isoformat() + "Z",
                **extra
            }
    
    @contextmanager
    def timer(self, operation: str):
        """Context manager for timing operations."""
        start = time.perf_counter()
        try:
            yield
        except Exception as e:
            self.record_error(operation)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.record(operation, elapsed_ms)
    
    def get_percentiles(self, operation: str) -> Dict[str, Any]:
        """Get P50, P95, P99 for an operation."""
        with self._lock:
            data = sorted(self._windows[operation])
        
        if not data:
            return {"p50": 0, "p95": 0, "p99": 0, "count": 0, "avg": 0}
        
        n = len(data)
        return {
            "p50": round(data[n // 2], 2),
            "p95": round(data[int(n * 0.95)] if n >= 20 else data[-1], 2),
            "p99": round(data[int(n * 0.99)] if n >= 100 else data[-1], 2),
            "count": n,
            "avg": round(sum(data) / n, 2),
            "min": round(data[0], 2),
            "max": round(data[-1], 2)
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get complete metrics snapshot."""
        uptime = time.time() - self._start_time
        
        with self._lock:
            operations = list(self._windows.keys())
            total_errors = sum(self._error_counts.values())
            error_rate = (total_errors / self._request_count * 100) if self._request_count > 0 else 0
        
        latency = {op: self.get_percentiles(op) for op in operations}
        
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "uptime_seconds": round(uptime, 1),
            "latency": latency,
            "health": self._health.copy(),
            "throughput": {
                "total_requests": self._request_count,
                "requests_per_minute": round(self._request_count / (uptime / 60), 2) if uptime > 0 else 0
            },
            "errors": {
                "total": total_errors,
                "by_operation": dict(self._error_counts),
                "error_rate_pct": round(error_rate, 3)
            }
        }
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export metrics to file and return the data."""
        metrics = self.get_all_metrics()
        
        # Write current metrics
        self.metrics_path.write_text(json.dumps(metrics, indent=2))
        
        # Also save to history (daily snapshots)
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        history_file = self.history_dir / f"metrics-{date_str}.json"
        
        # Append to daily file (or create)
        if history_file.exists():
            try:
                existing = json.loads(history_file.read_text())
                if isinstance(existing, list):
                    existing.append(metrics)
                else:
                    existing = [existing, metrics]
            except json.JSONDecodeError:
                existing = [metrics]
        else:
            existing = [metrics]
        
        # Keep only last 100 entries per day
        if len(existing) > 100:
            existing = existing[-100:]
        
        history_file.write_text(json.dumps(existing, indent=2))
        
        return metrics
    
    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._windows.clear()
            self._error_counts.clear()
            self._health.clear()
            self._request_count = 0
            self._start_time = time.time()


# Global singleton instance
metrics = MetricsCollector()


# Convenience functions
def record(operation: str, latency_ms: float) -> None:
    """Record a latency measurement."""
    metrics.record(operation, latency_ms)


def timer(operation: str):
    """Context manager for timing operations."""
    return metrics.timer(operation)


def get_metrics() -> Dict[str, Any]:
    """Get all metrics."""
    return metrics.get_all_metrics()


def export_metrics() -> Dict[str, Any]:
    """Export metrics to file."""
    return metrics.export_metrics()
