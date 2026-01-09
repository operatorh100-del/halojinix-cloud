"""
ScornSpine CRDTs - RT17000
Conflict-Free Replicated Data Types for cross-agent coherence.

Based on JONAH RT16900 research:
- LWWRegister: Last-Writer-Wins for atomic values
- ORSet: Observed-Remove Set for collections
- LWWMap: LWW semantics per key
- VectorClock: Conflict detection for concurrent modifications
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar
from dataclasses import dataclass, field, asdict
from enum import Enum


class ClockComparison(Enum):
    """Result of comparing two vector clocks."""
    A_BEFORE_B = "a_before_b"
    B_BEFORE_A = "b_before_a"
    CONCURRENT = "concurrent"  # CONFLICT!
    EQUAL = "equal"


@dataclass
class VectorClock:
    """
    Vector clock for conflict detection.
    Each agent maintains its own counter, clocks are compared to detect concurrency.
    """
    counters: Dict[str, int] = field(default_factory=dict)
    
    def increment(self, agent: str) -> "VectorClock":
        """Increment counter for agent."""
        self.counters[agent] = self.counters.get(agent, 0) + 1
        return self
    
    def merge(self, other: "VectorClock") -> "VectorClock":
        """Merge with another clock (take max of each counter)."""
        all_agents = set(self.counters.keys()) | set(other.counters.keys())
        for agent in all_agents:
            self.counters[agent] = max(
                self.counters.get(agent, 0),
                other.counters.get(agent, 0)
            )
        return self
    
    def copy(self) -> "VectorClock":
        """Create a deep copy."""
        return VectorClock(counters=self.counters.copy())
    
    def __lt__(self, other: "VectorClock") -> bool:
        """Check if self happened before other."""
        return self.compare(other) == ClockComparison.A_BEFORE_B
    
    def compare(self, other: "VectorClock") -> ClockComparison:
        """
        Compare two vector clocks.
        
        Returns:
            A_BEFORE_B: self happened before other
            B_BEFORE_A: other happened before self
            CONCURRENT: neither happened before the other (CONFLICT!)
            EQUAL: same version
        """
        all_agents = set(self.counters.keys()) | set(other.counters.keys())
        
        a_less = False
        b_less = False
        
        for agent in all_agents:
            a_val = self.counters.get(agent, 0)
            b_val = other.counters.get(agent, 0)
            
            if a_val < b_val:
                a_less = True
            if b_val < a_val:
                b_less = True
        
        if a_less and not b_less:
            return ClockComparison.A_BEFORE_B
        elif b_less and not a_less:
            return ClockComparison.B_BEFORE_A
        elif not a_less and not b_less:
            return ClockComparison.EQUAL
        else:
            return ClockComparison.CONCURRENT
    
    def to_dict(self) -> Dict[str, int]:
        """Serialize to dict."""
        return self.counters.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "VectorClock":
        """Deserialize from dict."""
        return cls(counters=data.copy())
    
    def __repr__(self) -> str:
        return f"VectorClock({self.counters})"


@dataclass
class LWWRegister:
    """
    Last-Writer-Wins Register.
    For atomic values where the latest update always wins.
    
    Use for: ADR status, agent states, single-value configs.
    """
    value: Any = None
    timestamp: float = 0.0
    writer: str = ""
    
    def set(self, value: Any, timestamp: Optional[float] = None, writer: str = "") -> bool:
        """
        Set value if timestamp is newer.
        
        Returns True if value was updated.
        """
        if timestamp is None:
            timestamp = datetime.utcnow().timestamp()
        
        if timestamp > self.timestamp:
            self.value = value
            self.timestamp = timestamp
            self.writer = writer
            return True
        return False
    
    def merge(self, other: "LWWRegister") -> "LWWRegister":
        """Merge with another register (latest wins)."""
        if other.timestamp > self.timestamp:
            self.value = other.value
            self.timestamp = other.timestamp
            self.writer = other.writer
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "value": self.value,
            "timestamp": self.timestamp,
            "writer": self.writer,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LWWRegister":
        """Deserialize from dict."""
        return cls(
            value=data.get("value"),
            timestamp=data.get("timestamp", 0.0),
            writer=data.get("writer", ""),
        )
    
    def __repr__(self) -> str:
        return f"LWWRegister(value={self.value!r}, ts={self.timestamp}, by={self.writer})"


@dataclass
class ORSet:
    """
    Observed-Remove Set.
    For collections where adds and removes can happen concurrently.
    
    Each element has a unique tag; removing an element removes all its tags.
    This allows concurrent add/remove to resolve correctly.
    
    Use for: Active skills list, agent subscriptions, valid ADRs.
    """
    elements: Dict[str, Set[str]] = field(default_factory=dict)  # element -> unique tags
    tombstones: Set[str] = field(default_factory=set)  # removed tags
    
    def add(self, element: str, unique_tag: Optional[str] = None) -> str:
        """
        Add element with unique tag.
        Returns the tag used.
        """
        if unique_tag is None:
            unique_tag = str(uuid.uuid4())
        
        if element not in self.elements:
            self.elements[element] = set()
        self.elements[element].add(unique_tag)
        
        return unique_tag
    
    def remove(self, element: str) -> bool:
        """
        Remove element by tombstoning all its tags.
        Returns True if element existed.
        """
        if element not in self.elements:
            return False
        
        # Tombstone all tags for this element
        self.tombstones.update(self.elements[element])
        del self.elements[element]
        return True
    
    def contains(self, element: str) -> bool:
        """Check if element is in set."""
        return element in self.elements and bool(self.elements[element])
    
    def value(self) -> Set[str]:
        """Get current set value."""
        return set(self.elements.keys())
    
    def merge(self, other: "ORSet") -> "ORSet":
        """Merge with another OR-Set."""
        # Add all elements from other
        for elem, tags in other.elements.items():
            if elem not in self.elements:
                self.elements[elem] = set()
            self.elements[elem].update(tags)
        
        # Merge tombstones
        self.tombstones.update(other.tombstones)
        
        # Remove tombstoned tags
        for elem in list(self.elements.keys()):
            self.elements[elem] -= self.tombstones
            if not self.elements[elem]:
                del self.elements[elem]
        
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "elements": {k: list(v) for k, v in self.elements.items()},
            "tombstones": list(self.tombstones),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ORSet":
        """Deserialize from dict."""
        return cls(
            elements={k: set(v) for k, v in data.get("elements", {}).items()},
            tombstones=set(data.get("tombstones", [])),
        )
    
    def __len__(self) -> int:
        return len(self.elements)
    
    def __repr__(self) -> str:
        return f"ORSet({self.value()})"


@dataclass
class LWWMap:
    """
    Last-Writer-Wins Map.
    Key-value store where each key has LWW semantics.
    
    Use for: Decisions registry, protocol versions, agent configs.
    """
    entries: Dict[str, Tuple[Any, float, str]] = field(default_factory=dict)  # key -> (value, timestamp, writer)
    
    def set(self, key: str, value: Any, timestamp: Optional[float] = None, writer: str = "") -> bool:
        """
        Set key if timestamp is newer.
        Returns True if value was updated.
        """
        if timestamp is None:
            timestamp = datetime.utcnow().timestamp()
        
        existing = self.entries.get(key)
        if existing is None or timestamp > existing[1]:
            self.entries[key] = (value, timestamp, writer)
            return True
        return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value for key."""
        entry = self.entries.get(key)
        return entry[0] if entry else default
    
    def get_with_metadata(self, key: str) -> Optional[Tuple[Any, float, str]]:
        """Get (value, timestamp, writer) for key."""
        return self.entries.get(key)
    
    def remove(self, key: str, timestamp: Optional[float] = None, writer: str = "") -> bool:
        """
        Remove key using tombstone (set to None).
        """
        if timestamp is None:
            timestamp = datetime.utcnow().timestamp()
        return self.set(key, None, timestamp, writer)
    
    def keys(self) -> List[str]:
        """Get all keys (excluding tombstoned)."""
        return [k for k, v in self.entries.items() if v[0] is not None]
    
    def items(self) -> List[Tuple[str, Any]]:
        """Get all (key, value) pairs (excluding tombstoned)."""
        return [(k, v[0]) for k, v in self.entries.items() if v[0] is not None]
    
    def merge(self, other: "LWWMap") -> "LWWMap":
        """Merge with another LWW-Map."""
        for key, (value, timestamp, writer) in other.entries.items():
            self.set(key, value, timestamp, writer)
        return self
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "entries": {
                k: {"value": v[0], "timestamp": v[1], "writer": v[2]}
                for k, v in self.entries.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LWWMap":
        """Deserialize from dict."""
        instance = cls()
        for key, entry in data.get("entries", {}).items():
            instance.entries[key] = (
                entry.get("value"),
                entry.get("timestamp", 0.0),
                entry.get("writer", ""),
            )
        return instance
    
    def __len__(self) -> int:
        return len(self.keys())
    
    def __repr__(self) -> str:
        return f"LWWMap({dict(self.items())})"


# Convenience functions for JSON serialization
def serialize_crdt(crdt) -> str:
    """Serialize any CRDT to JSON."""
    if isinstance(crdt, VectorClock):
        return json.dumps({"type": "VectorClock", "data": crdt.to_dict()})
    elif isinstance(crdt, LWWRegister):
        return json.dumps({"type": "LWWRegister", "data": crdt.to_dict()})
    elif isinstance(crdt, ORSet):
        return json.dumps({"type": "ORSet", "data": crdt.to_dict()})
    elif isinstance(crdt, LWWMap):
        return json.dumps({"type": "LWWMap", "data": crdt.to_dict()})
    else:
        raise TypeError(f"Unknown CRDT type: {type(crdt)}")


def deserialize_crdt(json_str: str):
    """Deserialize CRDT from JSON."""
    obj = json.loads(json_str)
    crdt_type = obj.get("type")
    data = obj.get("data", {})
    
    if crdt_type == "VectorClock":
        return VectorClock.from_dict(data)
    elif crdt_type == "LWWRegister":
        return LWWRegister.from_dict(data)
    elif crdt_type == "ORSet":
        return ORSet.from_dict(data)
    elif crdt_type == "LWWMap":
        return LWWMap.from_dict(data)
    else:
        raise ValueError(f"Unknown CRDT type: {crdt_type}")


if __name__ == "__main__":
    print("=== Vector Clock Demo ===")
    clock_halo = VectorClock()
    clock_halo.increment("halo").increment("halo").increment("halo")
    
    clock_jonah = VectorClock()
    clock_jonah.increment("jonah").increment("jonah")
    
    print(f"HALO's clock: {clock_halo}")
    print(f"JONAH's clock: {clock_jonah}")
    print(f"Comparison: {clock_halo.compare(clock_jonah)}")
    
    # Concurrent modification
    clock_vera = clock_halo.copy()
    clock_vera.increment("vera")
    clock_halo.increment("halo")
    
    print(f"\nAfter concurrent edits:")
    print(f"HALO's clock: {clock_halo}")
    print(f"VERA's clock: {clock_vera}")
    print(f"Comparison: {clock_halo.compare(clock_vera)}")  # CONCURRENT!
    
    print("\n=== LWW Register Demo ===")
    reg = LWWRegister()
    reg.set("initial", timestamp=1.0, writer="halo")
    print(f"After HALO: {reg}")
    
    reg.set("updated", timestamp=2.0, writer="jonah")
    print(f"After JONAH: {reg}")
    
    reg.set("stale", timestamp=1.5, writer="vera")  # Rejected (older)
    print(f"After VERA (stale): {reg}")
    
    print("\n=== OR-Set Demo ===")
    skills = ORSet()
    skills.add("S145")
    skills.add("S97")
    print(f"Active skills: {skills.value()}")
    
    skills.remove("S97")
    print(f"After removing S97: {skills.value()}")
    
    # Concurrent add/remove
    skills2 = ORSet()
    skills2.add("S97")  # Re-add on different replica
    
    skills.merge(skills2)
    print(f"After merge with concurrent add: {skills.value()}")
    
    print("\n=== LWW-Map Demo ===")
    decisions = LWWMap()
    decisions.set("model", "claude-opus", timestamp=1.0, writer="halo")
    decisions.set("deployment", "production", timestamp=1.0, writer="vera")
    
    print(f"Decisions: {dict(decisions.items())}")
    
    decisions.set("model", "gemini-2.5", timestamp=2.0, writer="jonah")
    print(f"After update: {dict(decisions.items())}")
