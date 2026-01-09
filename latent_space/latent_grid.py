"""
Latent Grid - Real-Time Agent Semantic Positioning
===================================================

RT31200: Implements the LatentGrid from HALOJINIX's Epic Forge dispatch.

This is NOT the UnifiedGraph (persistent knowledge graph).
This is ephemeral agent state tracking - where each agent is "thinking" in semantic space.

Architecture (20-round Triad consensus):
- 768-dim embeddings (via ScornSpine /embed endpoint)
- Velocity vectors with EMA smoothing (alpha=0.3)
- Cosine distance for proximity
- JSON persistence with atexit handler
- PCA projection for 2D visualization (fitted by HALO)

Usage:
    from haloscorn.latent_space.latent_grid import LatentGrid, get_latent_grid

    grid = get_latent_grid()
    grid.update_position("jonah", "Researching LatentGrid architecture")
    nearby = grid.find_nearby_agents("jonah", threshold=0.5)

Author: JONAH (Conductor) via 20-round Epic Forge
Date: 2026-01-07
Version: 1.0.0
"""

import atexit
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import requests

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AgentLatentState:
    """
    An agent's position in the latent semantic space.

    Attributes:
        agent_id: Agent identifier (halo, jonah, vera, halojinix)
        position: 768-dim embedding vector (current semantic focus)
        velocity: Rate of semantic drift (EMA smoothed)
        focus_topic: Human-readable description of current focus
        last_update: When this state was last updated
    """
    agent_id: str
    position: np.ndarray
    velocity: np.ndarray
    focus_topic: str
    last_update: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "agent_id": self.agent_id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "focus_topic": self.focus_topic,
            "last_update": self.last_update.isoformat(),
            "velocity_magnitude": float(np.linalg.norm(self.velocity))
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentLatentState":
        """Create from dictionary."""
        return cls(
            agent_id=data["agent_id"],
            position=np.array(data["position"]),
            velocity=np.array(data["velocity"]),
            focus_topic=data["focus_topic"],
            last_update=datetime.fromisoformat(data["last_update"])
        )


# =============================================================================
# Latent Grid Core
# =============================================================================

class LatentGrid:
    """
    Real-time semantic coordinate system for agent positions.

    Each agent has a "position" representing their current focus.
    Position updates when agent searches, posts, or updates live-feed.
    Velocity tracks rate of semantic drift over time.
    """

    # Configuration
    EMBEDDING_DIM = 768  # intfloat/multilingual-e5-base
    VELOCITY_ALPHA = 0.3  # EMA smoothing factor
    NEARBY_THRESHOLD = 0.5  # Cosine distance threshold for "nearby"
    STALE_SECONDS = 300  # 5 minutes before position decays
    PERSIST_INTERVAL = 60  # Save state every 60 seconds

    STATE_FILE = Path("data/latent-grid-state.json")
    PCA_FILE = Path("data/latent-grid-pca.pkl")

    def __init__(
        self,
        embedder=None,  # Direct embedder function (preferred)
        embed_endpoint: str = "http://127.0.0.1:7782/embed",  # Fallback HTTP
        auto_persist: bool = True
    ):
        """
        Initialize the Latent Grid.

        Args:
            embedder: Direct embedding function (avoids HTTP roundtrip)
            embed_endpoint: ScornSpine embedding endpoint (fallback if no embedder)
            auto_persist: Whether to auto-save state periodically
        """
        self.embedder = embedder  # Direct embedder (e.g., spine.embedder.encode)
        self.embed_endpoint = embed_endpoint
        self.agents: Dict[str, AgentLatentState] = {}
        self.pca = None  # Fitted by HALO for Panel viz
        self._lock = threading.Lock()
        self._last_persist = time.time()

        # Load existing state
        self._load_state()

        # Register shutdown handler
        if auto_persist:
            atexit.register(self._save_state)

        logger.info(f"LatentGrid initialized with {len(self.agents)} agents (embedder={'direct' if embedder else 'http'})")

    # =========================================================================
    # Embedding
    # =========================================================================

    def _get_embedding_direct(self, text: str) -> np.ndarray:
        """Get embedding using direct embedder (no HTTP roundtrip)."""
        try:
            # embedder.encode returns list of vectors, take first
            vector = self.embedder.encode([text.strip()])[0]
            return np.array(vector)
        except Exception as e:
            logger.error(f"Direct embedding failed: {e}")
            return np.zeros(self.EMBEDDING_DIM)

    @lru_cache(maxsize=100)
    def _get_embedding_http_cached(self, text: str) -> Tuple[float, ...]:
        """Get embedding via HTTP (fallback, cached)."""
        try:
            response = requests.post(
                self.embed_endpoint,
                json={"text": text},
                timeout=10
            )
            response.raise_for_status()
            # ScornSpine /embed returns {"success": True, "vector": [...]}
            data = response.json()
            embedding = data.get("vector", data.get("embedding", []))
            return tuple(embedding)
        except Exception as e:
            logger.error(f"HTTP embedding request failed: {e}")
            return tuple(np.zeros(self.EMBEDDING_DIM))

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding - uses direct embedder if available, else HTTP."""
        if self.embedder is not None:
            return self._get_embedding_direct(text[:500])
        else:
            cached = self._get_embedding_http_cached(text[:500])
            return np.array(cached)

    # =========================================================================
    # Position Updates
    # =========================================================================

    def update_position(
        self,
        agent_id: str,
        context: str,
        source: str = "explicit"
    ) -> AgentLatentState:
        """
        Update an agent's position in the latent space.

        Args:
            agent_id: Agent identifier (halo, jonah, vera, halojinix)
            context: Text context representing current focus
            source: Update source ("search", "live-feed", "signal", "explicit")

        Returns:
            Updated AgentLatentState
        """
        now = datetime.now(timezone.utc)
        new_position = self._get_embedding(context)

        with self._lock:
            if agent_id in self.agents:
                old_state = self.agents[agent_id]
                time_delta = (now - old_state.last_update).total_seconds()

                if time_delta > 0:
                    # Calculate velocity with EMA smoothing
                    raw_velocity = (new_position - old_state.position) / time_delta
                    velocity = (
                        self.VELOCITY_ALPHA * raw_velocity +
                        (1 - self.VELOCITY_ALPHA) * old_state.velocity
                    )
                else:
                    velocity = old_state.velocity
            else:
                velocity = np.zeros(self.EMBEDDING_DIM)

            # Extract focus topic
            if source == "search":
                focus = context[:100]
            else:
                # First sentence, cleaned
                focus = context.split('.')[0].strip()[:100]

            state = AgentLatentState(
                agent_id=agent_id,
                position=new_position,
                velocity=velocity,
                focus_topic=focus,
                last_update=now
            )
            self.agents[agent_id] = state

            # Auto-persist if interval elapsed
            if time.time() - self._last_persist > self.PERSIST_INTERVAL:
                self._save_state()
                self._last_persist = time.time()

        logger.debug(f"Updated position for {agent_id}: {focus[:50]}...")
        return state

    # =========================================================================
    # Proximity Calculations
    # =========================================================================

    def get_proximity(self, agent1: str, agent2: str) -> float:
        """
        Calculate cosine distance between two agents.

        Returns:
            Distance in range [0, 2]. 0 = identical, 2 = opposite.
            Returns inf if either agent not found.
        """
        with self._lock:
            if agent1 not in self.agents or agent2 not in self.agents:
                return float('inf')

            pos1 = self.agents[agent1].position
            pos2 = self.agents[agent2].position

        norm1 = np.linalg.norm(pos1)
        norm2 = np.linalg.norm(pos2)

        if norm1 == 0 or norm2 == 0:
            return float('inf')

        similarity = np.dot(pos1, pos2) / (norm1 * norm2)
        return float(1 - similarity)

    def find_nearby_agents(
        self,
        agent_id: str,
        threshold: float = None
    ) -> List[Tuple[str, float]]:
        """
        Find agents within cosine distance threshold.

        Args:
            agent_id: Reference agent
            threshold: Maximum cosine distance (default: NEARBY_THRESHOLD)

        Returns:
            List of (agent_id, distance) tuples, sorted by distance
        """
        if threshold is None:
            threshold = self.NEARBY_THRESHOLD

        # Copy positions under lock, then calculate distances without lock
        with self._lock:
            if agent_id not in self.agents:
                return []
            ref_pos = self.agents[agent_id].position.copy()
            other_agents = {k: v.position.copy() for k, v in self.agents.items() if k != agent_id}

        # Calculate distances without holding lock
        results = []
        ref_norm = np.linalg.norm(ref_pos)
        if ref_norm == 0:
            return []

        for other_id, other_pos in other_agents.items():
            other_norm = np.linalg.norm(other_pos)
            if other_norm == 0:
                continue
            similarity = np.dot(ref_pos, other_pos) / (ref_norm * other_norm)
            distance = float(1 - similarity)
            if distance < threshold:
                results.append((other_id, distance))

        return sorted(results, key=lambda x: x[1])

    # =========================================================================
    # Position Queries
    # =========================================================================

    def get_all_positions(self, include_stale: bool = True) -> Dict[str, Dict]:
        """
        Get all agent positions.

        Args:
            include_stale: Whether to include agents not updated recently

        Returns:
            Dict mapping agent_id to position data
        """
        now = datetime.now(timezone.utc)
        result = {}

        # Copy agent states under lock
        with self._lock:
            agent_snapshot = {k: v for k, v in self.agents.items()}

        # Process outside lock
        for agent_id, state in agent_snapshot.items():
            age_seconds = (now - state.last_update).total_seconds()

            if not include_stale and age_seconds > self.STALE_SECONDS:
                continue

            data = state.to_dict()
            data["age_seconds"] = age_seconds
            data["is_stale"] = age_seconds > self.STALE_SECONDS

            # Add 2D position if PCA available (no lock needed - pca is set once)
            if self.pca is not None:
                try:
                    projected = self.pca.transform([state.position])[0]
                    data["position_2d"] = (float(projected[0]), float(projected[1]))
                except Exception:
                    pass

            result[agent_id] = data

        return result

    def get_2d_position(self, agent_id: str) -> Optional[Tuple[float, float]]:
        """
        Get 2D projected position for visualization.

        Returns None if PCA not fitted or agent not found.
        """
        if self.pca is None:
            return None

        with self._lock:
            if agent_id not in self.agents:
                return None
            pos = self.agents[agent_id].position.copy()

        try:
            projected = self.pca.transform([pos])[0]
            return (float(projected[0]), float(projected[1]))
        except Exception:
            return None

    # =========================================================================
    # Persistence
    # =========================================================================

    def _load_state(self):
        """Load state from JSON file."""
        try:
            if self.STATE_FILE.exists():
                data = json.loads(self.STATE_FILE.read_text())
                for agent_id, state_data in data.get("agents", {}).items():
                    state_data["agent_id"] = agent_id
                    self.agents[agent_id] = AgentLatentState.from_dict(state_data)
                logger.info(f"Loaded {len(self.agents)} agent states from {self.STATE_FILE}")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
            self.agents = {}

    def _save_state(self):
        """Save state to JSON file."""
        try:
            data = {
                "version": "1.0.0",
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "agents": {}
            }

            with self._lock:
                for agent_id, state in self.agents.items():
                    data["agents"][agent_id] = {
                        "position": state.position.tolist(),
                        "velocity": state.velocity.tolist(),
                        "focus_topic": state.focus_topic,
                        "last_update": state.last_update.isoformat()
                    }

            self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.STATE_FILE.write_text(json.dumps(data, indent=2))
            logger.debug(f"Saved {len(self.agents)} agent states to {self.STATE_FILE}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def load_pca(self):
        """Load pre-fitted PCA from file (called by HALO after fitting)."""
        try:
            if self.PCA_FILE.exists():
                import pickle
                with open(self.PCA_FILE, "rb") as f:
                    self.pca = pickle.load(f)
                logger.info("Loaded PCA projection from file")
        except Exception as e:
            logger.warning(f"Failed to load PCA: {e}")

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get grid statistics."""
        now = datetime.now(timezone.utc)

        with self._lock:
            active_count = sum(
                1 for s in self.agents.values()
                if (now - s.last_update).total_seconds() < self.STALE_SECONDS
            )

            avg_velocity = 0.0
            if self.agents:
                avg_velocity = np.mean([
                    np.linalg.norm(s.velocity) for s in self.agents.values()
                ])

            return {
                "total_agents": len(self.agents),
                "active_agents": active_count,
                "stale_agents": len(self.agents) - active_count,
                "avg_velocity_magnitude": float(avg_velocity),
                "pca_fitted": self.pca is not None,
                "state_file": str(self.STATE_FILE),
                "embedding_dim": self.EMBEDDING_DIM
            }


# =============================================================================
# Singleton Access
# =============================================================================

_latent_grid: Optional[LatentGrid] = None
_grid_lock = threading.Lock()


def get_latent_grid() -> LatentGrid:
    """Get or create the LatentGrid singleton."""
    global _latent_grid

    with _grid_lock:
        if _latent_grid is None:
            _latent_grid = LatentGrid()
        return _latent_grid
