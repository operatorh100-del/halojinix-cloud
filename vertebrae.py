"""
Vertebrae - Document Categories for ScornSpine
==============================================

Defines the document categories that ScornSpine indexes, with priority
weights and collection patterns. Each vertebra is a segment of the spine,
and together they form the complete structure.

Categories are weighted by importance:
    1.0  = Critical (agent instructions, core rules)
    0.9  = High (decisions, handoffs, active code)
    0.8  = Standard (documentation, research)
    0.7  = Reference (configs, schemas)
    0.6  = Historical (session notes, archives)

UPDATED 2025-12-05 (The Great Combining):
- Added .ragignore support for exclusion patterns
- Removed low-quality ScornMiner extractions from knowledge-base/
- Consolidated to high-signal sources only
"""

from pathlib import Path
from typing import Dict, List, Any
import glob
import fnmatch
import logging

logger = logging.getLogger(__name__)

# Project root (relative to this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAGIGNORE_FILE = PROJECT_ROOT / ".ragignore"


# -------------------------------------------------------------------------------
# RAGIGNORE SUPPORT
# -------------------------------------------------------------------------------

def load_ragignore() -> List[str]:
    """Load exclusion patterns from .ragignore file."""
    patterns = []
    if RAGIGNORE_FILE.exists():
        with open(RAGIGNORE_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    # Skip negation patterns (we don't support !pattern yet)
                    if not line.startswith('!'):
                        patterns.append(line)
    return patterns


def should_exclude(path: Path, patterns: List[str]) -> bool:
    """Check if a path should be excluded based on ragignore patterns."""
    try:
        rel_path = path.relative_to(PROJECT_ROOT)
        rel_str = str(rel_path).replace('\\', '/')
    except ValueError:
        return False  # Path not relative to project, don't exclude

    for pattern in patterns:
        # Handle directory patterns (ending with /)
        if pattern.endswith('/'):
            dir_pattern = pattern.rstrip('/')
            if rel_str.startswith(dir_pattern + '/') or rel_str == dir_pattern:
                return True
        # Handle glob patterns
        elif fnmatch.fnmatch(rel_str, pattern):
            return True
        # Handle patterns that could match anywhere in path
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

# -------------------------------------------------------------------------------
# DOCUMENT CATEGORIES (VERTEBRAE)
# -------------------------------------------------------------------------------

DOCUMENT_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "critical": {
        "weight": 1.0,
        "description": "Core project rules, agent configuration, copilot instructions",
        "paths": [
            ".github/copilot-instructions.md",
            "agent-state.json",
            "Agent_Charter_knowledge_pack_1.txt",
        ],
        "patterns": [
            "docs/agents/*.md",
            "docs/agents/archive/*.md",
        ]
    },

    "agent_code": {
        "weight": 0.95,
        "description": "Agent implementations, bridges, and voice systems",
        "paths": [],
        "patterns": [
            "scripts/agent/**/*.ts",
            "scripts/voice/*.py",
            "haloscorn/**/*.py",
            "engine/agents/*.ts",
        ]
    },

    "haloscorn_config": {
        "weight": 0.9,
        "description": "HaloScorn system configs and prompts",
        "paths": [
            "haloscorn/config/haloscorn.json",
        ],
        "patterns": [
            "haloscorn/llm/*.py",
            "haloscorn/config/*.json",
        ]
    },

    "decisions": {
        "weight": 0.9,
        "description": "Architectural decisions and council votes",
        "paths": [],
        "patterns": [
            "docs/decisions/*.md",
        ]
    },

    "protocols": {
        "weight": 0.95,
        "description": "Agent protocols and operational procedures (RT311G)",
        "paths": [],
        "patterns": [
            "docs/protocols/*.md",
        ]
    },

    "handoffs": {
        "weight": 0.9,
        "description": "Inter-agent communication and context handoffs",
        "paths": [
            "agent-handoff/README.md",
            "agent-handoff/HANDOFF-TEMPLATE.md",
        ],
        "patterns": [
            "agent-handoff/messages/*.md",
            "agent-handoff/messages/**/*.md",
            "agent-handoff/messages/**/*.json",
            "agent-handoff/handoff/*.md",
            "agent-handoff/vera-insights/*.md",
            "agent-handoff/vera-insights/*.json",
            "agent-handoff/chat-clipboard/*.md",
            # REAL-TIME CHAT MIRRORS (added 2025-01-11 - ADR-0039 completion)
            "agent-handoff/chat/*.md",
            "agent-handoff/chat/**/*.md",
            "agent-handoff/chat/**/*.json",
        ]
    },

    "conversations": {
        "weight": 0.85,
        "description": "Archived conversation history",
        "paths": [],
        "patterns": [
            "logs/conversations/**/*.md",
            "logs/conversations/**/*.json",
        ]
    },

    "voice_transcripts": {
        "weight": 0.9,
        "description": "Voice transcription logs - accessibility context from The Human (RT286)",
        "paths": [
            "logs/voice-daemon-v7.log",
            "logs/voice-daemon.log",
        ],
        "patterns": [
            # Main voice daemon logs
            "logs/voice-daemon-v7.log",
            # Archived transcripts (when we create dedicated archive)
            "logs/voice-transcripts/**/*.log",
            "logs/voice-transcripts/**/*.md",
        ]
    },

    "research": {
        "weight": 0.85,
        "description": "Technical research, math, physics documentation",
        "paths": [],
        "patterns": [
            "research/*.md",
            "research/**/*.md",
            # HIGH-QUALITY knowledge-base folders ONLY (human-curated)
            "knowledge-base/implementations/**/*.md",
            "knowledge-base/strategy/*.md",
            "knowledge-base/halojinix-canon/*.md",
            "knowledge-base/research/*.md",
            "knowledge-base/real-time-learnings/*.md",
            # RT653: Voice system architecture docs (authoritative, RT134 two-process explained)
            "knowledge-base/voice/*.md",
            # RT462: Website knowledge - MUST READ for any HTML/CSS/JS work
            "knowledge-base/website/*.md",
            "knowledge-base/RT434-web-mastery-2025.md",
            # EXCLUDED: pending-review/, decisions/, insights/, blockers-resolved/, code-patterns/, questions/
            # These are low-quality ScornMiner auto-extracts that pollute the index
        ]
    },

    "external_docs": {
        "weight": 0.9,
        "description": "External API documentation - Claude/Anthropic, VS Code Extension APIs",
        "paths": [],
        "patterns": [
            "knowledge-base/external-docs/**/*.md",
        ]
    },

    "strategy": {
        "weight": 0.95,
        "description": "Strategy docs, Email Package, pitch materials - BUSINESS CRITICAL",
        "paths": [],
        "patterns": [
            "knowledge-base/strategy/*.md",
            "docs/pitch/*.md",
        ]
    },

    "real_time_learnings": {
        "weight": 0.9,
        "description": "Real-time problem-solutions, insights, gotchas, patterns",
        "paths": [
            "knowledge-base/real-time-learnings/README.md",
        ],
        "patterns": [
            "knowledge-base/real-time-learnings/*.md",
        ]
    },

    "implementations": {
        "weight": 1.0,
        "description": "Full implementation lifecycles - problems, attempts, roundtables, proven solutions",
        "paths": [
            "knowledge-base/implementations/README.md",
        ],
        "patterns": [
            "knowledge-base/implementations/**/*.md",
        ]
    },

    "engine_docs": {
        "weight": 0.8,
        "description": "Engine architecture and implementation docs",
        "paths": [
            "README.md",
            "CHANGELOG.md",
        ],
        "patterns": [
            "docs/*.md",
            "docs/**/*.md",
            "halojinix-intent-agent/docs/**/*.md",
        ]
    },

    "sessions": {
        "weight": 0.75,
        "description": "Session notes and work history",
        "paths": [],
        "patterns": [
            "docs/sessions/*.md",
            "logs/agents/*.md",
        ]
    },

    "lore": {
        "weight": 0.8,
        "description": "Halojinix lore, vision documents, manifestos",
        "paths": [
            "docs/HALOJINIX-VISION-DOCUMENT-2025.txt",
        ],
        "patterns": [
            "docs/lore/*.md",
            "inbox/processed/*.txt",
        ]
    },

    "agent_skills": {
        "weight": 0.95,
        "description": "Agent skill cards, STP doctrine, roundtable protocols, and operational capabilities",
        "paths": [
            "agent-skills/README.md",
            "agent-skills/shared/README.md",
            "agent-skills/shared/stp-doctrine.md",
        ],
        "patterns": [
            "agent-skills/**/*.md",
            "agent-skills/**/*.yaml",
            "agent-skills/**/skill-cards/*.md",
        ]
    },

    "configs": {
        "weight": 0.7,
        "description": "Configuration files and schemas",
        "paths": [
            "halonet.config.json",
        ],
        "patterns": [
            "schemas/*.json",
            "schemas/*.md",
        ]
    },

    "audit": {
        "weight": 0.7,
        "description": "Audit rules, reports, and standards",
        "paths": [],
        "patterns": [
            "audit/*.md",
            "audit/rules/*.ts",
            "audit/reports/*.md",
        ]
    },

    # RT1402: Codebooks - Static fallback knowledge layer
    "codebooks": {
        "weight": 0.95,
        "description": "Static codebooks for offline/fallback knowledge access - API references, architecture guides",
        "paths": [
            "docs/codebooks/README.md",
        ],
        "patterns": [
            "docs/codebooks/*.md",
            "docs/codebooks/**/*.md",
        ]
    },

    # RT1335: Chronicle System - timestamped accountability records
    "chronicle": {
        "weight": 0.95,
        "description": "Chronicle system: human directives, agent exchanges, accountability events (RT1335)",
        "paths": [
            "data/chronicle/SCHEMA.md",
        ],
        "patterns": [
            "data/chronicle/*.jsonl",
        ]
    },

    # RT4400: HALOJINIX Decision History - Latent Space Experiment 1
    "halojinix_decisions": {
        "weight": 0.95,
        "description": "HALOJINIX synthesis posts, coordination decisions, analysis - enables precedent-aware responses",
        "paths": [
            "knowledge-base/halojinix-decisions/README.md",
        ],
        "patterns": [
            "knowledge-base/halojinix-decisions/*.md",
        ]
    },
}


# -------------------------------------------------------------------------------
# DOCUMENT COLLECTION
# -------------------------------------------------------------------------------

def collect_all_documents() -> Dict[str, List[Path]]:
    """
    Collect all documents organized by category.
    Respects .ragignore exclusion patterns.

    Returns:
        Dict mapping category names to lists of file paths
    """
    collected = {}
    total = 0
    excluded_count = 0

    # Load exclusion patterns
    ragignore_patterns = load_ragignore()
    if ragignore_patterns:
        logger.info(f"Loaded {len(ragignore_patterns)} patterns from .ragignore")

    for category, config in DOCUMENT_CATEGORIES.items():
        files = []

        # Direct paths
        for path in config.get("paths", []):
            full_path = PROJECT_ROOT / path
            if full_path.exists():
                if not should_exclude(full_path, ragignore_patterns):
                    files.append(full_path)
                else:
                    excluded_count += 1

        # Glob patterns
        for pattern in config.get("patterns", []):
            matches = glob.glob(str(PROJECT_ROOT / pattern), recursive=True)
            for match in matches:
                path = Path(match)
                if path.is_file():
                    if not should_exclude(path, ragignore_patterns):
                        files.append(path)
                    else:
                        excluded_count += 1

        # Deduplicate
        files = list(set(files))
        collected[category] = files
        total += len(files)

    if excluded_count > 0:
        logger.info(f"Excluded {excluded_count} files via .ragignore")

    return collected


def get_category_weight(category: str) -> float:
    """Get the weight for a document category."""
    return DOCUMENT_CATEGORIES.get(category, {}).get("weight", 0.5)


def get_category_description(category: str) -> str:
    """Get the description for a document category."""
    return DOCUMENT_CATEGORIES.get(category, {}).get("description", "Unknown category")


# -------------------------------------------------------------------------------
# AGENT-SPECIFIC CATEGORY BOOSTS
# -------------------------------------------------------------------------------

# Per-agent boost multipliers for personalized search results
# Values > 1.0 = prioritize, < 1.0 = de-prioritize, 1.0 = neutral
# VERA-adjusted 2025-12-08 after 15-round stress test

AGENT_CATEGORY_BOOSTS: Dict[str, Dict[str, float]] = {
    "HALO": {
        "critical": 1.0,
        "agent_skills": 2.5,     # RT1152: Skills always prioritized
        "agent_code": 2.0,        # Primary domain - GPU/code
        "haloscorn_config": 1.0,
        "decisions": 1.0,
        "handoffs": 0.8,
        "conversations": 0.8,
        "voice_transcripts": 1.2, # RT286: Voice accessibility context
        "research": 0.7,          # JONAH's domain
        "strategy": 0.8,
        "real_time_learnings": 1.0,
        "implementations": 2.0,   # Full solution lifecycles
        "engine_docs": 1.8,       # Architecture docs
        "sessions": 0.8,
        "lore": 1.0,
        "configs": 1.0,
        "audit": 0.8,
        "external_docs": 1.5,     # VS Code Extension API especially
        "codebooks": 2.0,         # RT1402: Static fallback knowledge
        "chronicle": 1.0,         # RT1335: Accountability records
        "protocols": 1.2,         # RT8800: Infrastructure protocols
    },
    "JONAH": {
        "critical": 1.0,
        "agent_skills": 2.5,     # RT1152: Skills always prioritized
        "agent_code": 0.7,        # HALO's domain
        "haloscorn_config": 1.0,
        "decisions": 2.0,         # Primary domain - architecture
        "handoffs": 1.0,
        "conversations": 1.0,
        "voice_transcripts": 1.2, # RT286: Voice accessibility context
        "research": 2.0,          # Primary domain - research
        "strategy": 1.2,
        "real_time_learnings": 1.8,
        "implementations": 1.8,   # VERA-adjusted from 1.5
        "engine_docs": 1.0,
        "sessions": 1.2,
        "lore": 1.2,
        "configs": 1.0,
        "audit": 1.0,
        "external_docs": 2.0,     # Research domain - Claude API docs
        "codebooks": 2.5,         # RT1402: JONAH creates codebooks - highest priority
        "chronicle": 1.2,         # RT1335: Research references
        "protocols": 1.5,         # RT8800: Research protocols
    },
    "VERA": {
        "critical": 1.0,
        "agent_skills": 2.5,     # RT1152: Skills always prioritized
        "agent_code": 0.85,       # VERA-adjusted from 0.7 - needs visibility
        "haloscorn_config": 1.0,
        "decisions": 1.5,
        "handoffs": 2.0,          # Primary domain - coordination
        "conversations": 1.8,
        "voice_transcripts": 1.5, # RT286: Voice accessibility - VERA coordinates Human
        "research": 0.9,          # VERA-adjusted from 0.8
        "strategy": 2.0,          # Primary domain - planning
        "real_time_learnings": 1.5,
        "implementations": 1.2,   # VERA-adjusted from 0.8 - cross-cutting
        "engine_docs": 0.85,      # VERA-adjusted from 0.7 - needs visibility
        "sessions": 1.0,
        "lore": 1.0,
        "configs": 1.0,
        "audit": 1.2,
        "external_docs": 1.5,     # Coordination needs API knowledge
        "codebooks": 2.0,         # RT1402: Static fallback knowledge
        "chronicle": 1.5,         # RT1335: Coordination accountability
        "protocols": 2.0,         # RT8800: Coordination protocols - primary domain
    },
    "HALONET": {
        # Placeholder - neutral until HALONET activates (Level 1 agent)
        "critical": 1.0,
        "agent_skills": 2.5,     # RT1152: Skills always prioritized
        "agent_code": 1.0,
        "haloscorn_config": 1.5,  # Will manage configs
        "decisions": 1.0,
        "handoffs": 1.0,
        "conversations": 1.0,
        "voice_transcripts": 1.0, # RT286: Voice accessibility context
        "research": 1.0,
        "strategy": 1.0,
        "real_time_learnings": 1.0,
        "implementations": 1.2,
        "engine_docs": 1.0,
        "sessions": 1.0,
        "lore": 1.0,
        "configs": 1.5,           # Will manage configs
        "audit": 1.0,
        "external_docs": 1.0,     # Neutral
        "codebooks": 1.5,         # RT1402: Static fallback knowledge
        "chronicle": 1.0,         # RT1335: Accountability records
        "protocols": 1.0,         # RT8800: Added missing category
    },
    "HALOJINIX": {
        # RT8800: Primary orchestrator - synthesis and coordination
        "critical": 1.5,
        "agent_skills": 2.5,      # RT1152: Skills always prioritized
        "agent_code": 1.5,        # Cross-cutting awareness
        "haloscorn_config": 1.2,
        "decisions": 1.8,         # Synthesis requires decision context
        "handoffs": 1.8,          # Coordination domain
        "conversations": 1.5,
        "voice_transcripts": 1.8, # Primary voice interface
        "research": 1.2,
        "strategy": 1.8,          # Strategic oversight
        "real_time_learnings": 1.5,
        "implementations": 1.5,
        "engine_docs": 1.2,
        "sessions": 1.5,
        "lore": 1.2,
        "configs": 1.2,
        "audit": 1.2,
        "external_docs": 1.5,
        "codebooks": 2.0,         # RT1402: Synthesis needs full knowledge
        "chronicle": 1.5,         # RT1335: Accountability records
        "protocols": 1.8,         # RT8800: Coordination protocols
        "halojinix_decisions": 2.5, # RT8800: Own decisions highest
    },
}


def get_agent_boost(agent: str, category: str) -> float:
    """
    Get the boost multiplier for a category based on agent identity.

    Args:
        agent: Agent name (HALO, JONAH, VERA, HALONET)
        category: Document category from DOCUMENT_CATEGORIES

    Returns:
        Boost multiplier (default 1.0 if agent/category not found)
    """
    agent_upper = agent.upper() if agent else ""
    if agent_upper in AGENT_CATEGORY_BOOSTS:
        return AGENT_CATEGORY_BOOSTS[agent_upper].get(category, 1.0)
    return 1.0


# -------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------

def main():
    """CLI to inspect document categories."""
    print("-" * 60)
    print("  SCORNSPINE VERTEBRAE - Document Categories")
    print("-" * 60)
    print()

    collected = collect_all_documents()
    total = 0

    for category, files in sorted(collected.items(), key=lambda x: -DOCUMENT_CATEGORIES[x[0]]["weight"]):
        weight = DOCUMENT_CATEGORIES[category]["weight"]
        desc = DOCUMENT_CATEGORIES[category]["description"]
        count = len(files)
        total += count

        print(f"[{weight:.2f}] {category}: {count} files")
        print(f"       {desc}")
        if files and count <= 5:
            for f in files[:5]:
                print(f"         - {f.relative_to(PROJECT_ROOT)}")
        elif files:
            for f in files[:3]:
                print(f"         - {f.relative_to(PROJECT_ROOT)}")
            print(f"         ... and {count - 3} more")
        print()

    print("-" * 60)
    print(f"  Total: {total} documents across {len(collected)} categories")
    print("-" * 60)


if __name__ == "__main__":
    main()
