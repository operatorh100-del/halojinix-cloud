import hashlib
import time
from typing import Dict, Any, Optional

class BloomCache:
    """
    RT5010: Ultra-fast RAM cache with Bloom Filter pre-check.
    Reduces redundant embedding searches for frequent queries.
    """
    def __init__(self, size: int = 1000, bit_size: int = 8192):
        self.size = size
        self.bit_size = bit_size
        self.filter = 0  # Simple bitset using int
        self.cache: Dict[str, Any] = {}
        self.insertion_order = []

    def _get_hashes(self, key: str):
        h1 = int(hashlib.md5(key.encode()).hexdigest(), 16) % self.bit_size
        h2 = int(hashlib.sha1(key.encode()).hexdigest(), 16) % self.bit_size
        return h1, h2

    def add(self, key: str, value: Any):
        h1, h2 = self._get_hashes(key)
        self.filter |= (1 << h1)
        self.filter |= (1 << h2)

        # LRU management
        if key in self.cache:
            self.insertion_order.remove(key)

        self.cache[key] = {
            "data": value,
            "timestamp": time.time()
        }
        self.insertion_order.append(key)

        if len(self.insertion_order) > self.size:
            oldest = self.insertion_order.pop(0)
            del self.cache[oldest]

    def get(self, key: str) -> Optional[Any]:
        # Fast Bloom Filter check
        h1, h2 = self._get_hashes(key)
        if not (self.filter & (1 << h1)) or not (self.filter & (1 << h2)):
            return None # Definitely not in cache

        # Potential hit, check dict
        entry = self.cache.get(key)
        if entry:
            # Check for TTL (e.g., 5 minutes)
            if time.time() - entry["timestamp"] < 300:
                return entry["data"]
            else:
                # Expired
                self.insertion_order.remove(key)
                del self.cache[key]

        return None

# Global instance for server-side use
global_cache = BloomCache()
