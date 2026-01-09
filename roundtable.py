"""
Roundtable - Structured Multi-Agent Discussion Capture
======================================================

Automatically captures roundtable discussions to the implementations
knowledge base. A roundtable is a structured decision-making event
with multiple agent perspectives, voting, and a final decision.

Architecture:
    - Extends Marrow conversation logging
    - Tags entries as part of a roundtable
    - Auto-files to knowledge-base/implementations/

Usage:
    from haloscorn.scornspine.roundtable import RoundtableCapture

    # Start a roundtable
    rt = RoundtableCapture(
        topic="agent-identity-disambiguation",
        implementation="agent-identity-disambiguation"
    )

    # Log agent contributions
    rt.add_round(1, "JONAH", "The failure mode is clear...")
    rt.add_round(1, "HALO", "Let me check what actually works...")

    # Record votes
    rt.add_vote("JONAH", True, "Approach C with enhancements")
    rt.add_vote("HALO", True, "Agreed")

    # Finalize
    rt.finalize(decision="Add rule #5 to IDENTITY RULES")
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
import threading
import logging

logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
IMPLEMENTATIONS_DIR = PROJECT_ROOT / "knowledge-base" / "implementations"


# -------------------------------------------------------------------------------
# DATA STRUCTURES
# -------------------------------------------------------------------------------

@dataclass
class RoundEntry:
    """A single round contribution from an agent."""
    round_number: int
    agent: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class VoteEntry:
    """A vote from an agent."""
    agent: str
    vote: bool  # True = approve, False = reject
    comment: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class RoundtableRecord:
    """Complete record of a roundtable discussion."""
    topic: str
    implementation: str
    started: str
    ended: Optional[str] = None
    participants: List[str] = field(default_factory=list)
    rounds: List[RoundEntry] = field(default_factory=list)
    votes: List[VoteEntry] = field(default_factory=list)
    decision: Optional[str] = None
    outcome: Optional[str] = None  # "UNANIMOUS", "MAJORITY", "NO_CONSENSUS"

    def vote_summary(self) -> Dict[str, int]:
        """Get vote counts."""
        approve = sum(1 for v in self.votes if v.vote)
        reject = sum(1 for v in self.votes if not v.vote)
        return {"approve": approve, "reject": reject, "total": len(self.votes)}

    def is_unanimous(self) -> bool:
        """Check if all votes agree."""
        if not self.votes:
            return False
        first_vote = self.votes[0].vote
        return all(v.vote == first_vote for v in self.votes)


# -------------------------------------------------------------------------------
# ROUNDTABLE CAPTURE
# -------------------------------------------------------------------------------

class RoundtableCapture:
    """
    Captures a roundtable discussion for the implementations knowledge base.

    Lifecycle:
    1. Create RoundtableCapture with topic and implementation name
    2. Call add_round() as agents contribute
    3. Call add_vote() when voting begins
    4. Call finalize() with the decision to save

    The roundtable is auto-filed to:
    knowledge-base/implementations/{implementation}/roundtables/YYYY-MM-DD-{topic}.md
    """

    def __init__(
        self,
        topic: str,
        implementation: Optional[str] = None,
        participants: Optional[List[str]] = None
    ):
        """
        Start a new roundtable capture.

        Args:
            topic: Short topic description (used in filename)
            implementation: Implementation folder name (if None, uses topic)
            participants: Initial list of participating agents
        """
        self.record = RoundtableRecord(
            topic=topic,
            implementation=implementation or topic,
            started=datetime.now().isoformat(),
            participants=participants or []
        )
        self._lock = threading.Lock()
        self._finalized = False

        logger.info(f"[Roundtable] Started: {topic}")

    def add_round(self, round_number: int, agent: str, content: str):
        """
        Add a round contribution from an agent.

        Args:
            round_number: Which round (1, 2, 3, ...)
            agent: Agent name (JONAH, HALO, VERA, etc.)
            content: The agent's contribution text
        """
        if self._finalized:
            raise RuntimeError("Cannot add to finalized roundtable")

        with self._lock:
            entry = RoundEntry(
                round_number=round_number,
                agent=agent.upper(),
                content=content
            )
            self.record.rounds.append(entry)

            # Track participants
            if agent.upper() not in self.record.participants:
                self.record.participants.append(agent.upper())

            logger.debug(f"[Roundtable] Round {round_number} - {agent}: {content[:50]}...")

    def add_vote(self, agent: str, approve: bool, comment: str = ""):
        """
        Record a vote from an agent.

        Args:
            agent: Agent name
            approve: True for approve, False for reject
            comment: Optional comment with the vote
        """
        if self._finalized:
            raise RuntimeError("Cannot add to finalized roundtable")

        with self._lock:
            vote = VoteEntry(
                agent=agent.upper(),
                vote=approve,
                comment=comment
            )
            self.record.votes.append(vote)

            logger.debug(f"[Roundtable] Vote - {agent}: {'[OK]' if approve else '[FAIL]'} {comment}")

    def finalize(self, decision: str) -> Path:
        """
        Finalize the roundtable and save to implementations folder.

        Args:
            decision: The final decision made

        Returns:
            Path to the saved markdown file
        """
        if self._finalized:
            raise RuntimeError("Roundtable already finalized")

        with self._lock:
            self.record.ended = datetime.now().isoformat()
            self.record.decision = decision

            # Determine outcome
            if not self.record.votes:
                self.record.outcome = "NO_VOTE"
            elif self.record.is_unanimous():
                self.record.outcome = "UNANIMOUS"
            else:
                summary = self.record.vote_summary()
                if summary["approve"] > summary["reject"]:
                    self.record.outcome = "MAJORITY_APPROVE"
                elif summary["reject"] > summary["approve"]:
                    self.record.outcome = "MAJORITY_REJECT"
                else:
                    self.record.outcome = "SPLIT"

            # Save to file
            filepath = self._save_to_file()
            self._finalized = True

            logger.info(f"[Roundtable] Finalized: {filepath}")
            return filepath

    def _save_to_file(self) -> Path:
        """Save the roundtable record to markdown file."""
        # Ensure implementation directory exists
        impl_dir = IMPLEMENTATIONS_DIR / self.record.implementation
        roundtables_dir = impl_dir / "roundtables"
        roundtables_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        date = datetime.now().strftime("%Y-%m-%d")
        safe_topic = self.record.topic.lower().replace(" ", "-")[:50]
        filename = f"{date}-{safe_topic}.md"
        filepath = roundtables_dir / filename

        # Handle duplicates
        counter = 1
        while filepath.exists():
            filename = f"{date}-{safe_topic}-{counter}.md"
            filepath = roundtables_dir / filename
            counter += 1

        # Build markdown
        md_lines = self._build_markdown()

        # Write file
        filepath.write_text("\n".join(md_lines), encoding='utf-8')

        return filepath

    def _build_markdown(self) -> List[str]:
        """Build markdown content for the roundtable record."""
        lines = [
            f"# Roundtable: {self.record.topic}",
            "",
            f"**Date:** {self.record.started.split('T')[0]}",
            f"**Participants:** {', '.join(self.record.participants)}",
            f"**Topic:** {self.record.topic}",
            f"**Outcome:** {self.record.outcome}",
            "",
            "---",
            "",
        ]

        # Group rounds
        rounds_by_number: Dict[int, List[RoundEntry]] = {}
        for entry in self.record.rounds:
            if entry.round_number not in rounds_by_number:
                rounds_by_number[entry.round_number] = []
            rounds_by_number[entry.round_number].append(entry)

        # Add rounds
        for round_num in sorted(rounds_by_number.keys()):
            lines.append(f"## Round {round_num}")
            lines.append("")

            for entry in rounds_by_number[round_num]:
                lines.append(f"**{entry.agent}:**")
                lines.append("")
                lines.append(entry.content)
                lines.append("")

        # Add voting section
        if self.record.votes:
            lines.append("## Voting")
            lines.append("")
            lines.append("| Agent | Vote | Comment |")
            lines.append("|-------|------|---------|")

            for vote in self.record.votes:
                vote_icon = "[OK]" if vote.vote else "[FAIL]"
                lines.append(f"| {vote.agent} | {vote_icon} | {vote.comment} |")

            lines.append("")

            summary = self.record.vote_summary()
            lines.append(f"**Result:** {self.record.outcome} ({summary['approve']}/{summary['total']})")
            lines.append("")

        # Add decision
        if self.record.decision:
            lines.append("## Decision")
            lines.append("")
            lines.append(self.record.decision)
            lines.append("")

        return lines

    def to_dict(self) -> dict:
        """Export as dictionary (for JSON serialization)."""
        return {
            "topic": self.record.topic,
            "implementation": self.record.implementation,
            "started": self.record.started,
            "ended": self.record.ended,
            "participants": self.record.participants,
            "rounds": [
                {
                    "round": r.round_number,
                    "agent": r.agent,
                    "content": r.content,
                    "timestamp": r.timestamp
                }
                for r in self.record.rounds
            ],
            "votes": [
                {
                    "agent": v.agent,
                    "vote": v.vote,
                    "comment": v.comment,
                    "timestamp": v.timestamp
                }
                for v in self.record.votes
            ],
            "decision": self.record.decision,
            "outcome": self.record.outcome
        }


# -------------------------------------------------------------------------------
# CONVENIENCE FUNCTIONS
# -------------------------------------------------------------------------------

def quick_roundtable(
    topic: str,
    implementation: str,
    rounds: List[Dict],
    votes: List[Dict],
    decision: str
) -> Path:
    """
    Quick helper to create and finalize a roundtable in one call.

    Args:
        topic: Roundtable topic
        implementation: Implementation folder name
        rounds: List of {"round": int, "agent": str, "content": str}
        votes: List of {"agent": str, "approve": bool, "comment": str}
        decision: Final decision text

    Returns:
        Path to saved file

    Example:
        quick_roundtable(
            topic="memory-optimization",
            implementation="memory-optimization",
            rounds=[
                {"round": 1, "agent": "JONAH", "content": "Analysis..."},
                {"round": 1, "agent": "HALO", "content": "Implementation..."},
            ],
            votes=[
                {"agent": "JONAH", "approve": True, "comment": "Agreed"},
                {"agent": "HALO", "approve": True, "comment": "Agreed"},
            ],
            decision="Use approach X"
        )
    """
    rt = RoundtableCapture(topic=topic, implementation=implementation)

    for r in rounds:
        rt.add_round(r["round"], r["agent"], r["content"])

    for v in votes:
        rt.add_vote(v["agent"], v["approve"], v.get("comment", ""))

    return rt.finalize(decision)


# -------------------------------------------------------------------------------
# API ENDPOINT HELPERS
# -------------------------------------------------------------------------------

# Active roundtables (for multi-request capture)
_active_roundtables: Dict[str, RoundtableCapture] = {}
_roundtables_lock = threading.Lock()


def start_roundtable(roundtable_id: str, topic: str, implementation: str) -> RoundtableCapture:
    """Start a new roundtable (for API use)."""
    with _roundtables_lock:
        if roundtable_id in _active_roundtables:
            raise ValueError(f"Roundtable {roundtable_id} already exists")

        rt = RoundtableCapture(topic=topic, implementation=implementation)
        _active_roundtables[roundtable_id] = rt
        return rt


def get_roundtable(roundtable_id: str) -> Optional[RoundtableCapture]:
    """Get an active roundtable by ID."""
    with _roundtables_lock:
        return _active_roundtables.get(roundtable_id)


def finalize_roundtable(roundtable_id: str, decision: str) -> Path:
    """Finalize and remove a roundtable."""
    with _roundtables_lock:
        if roundtable_id not in _active_roundtables:
            raise ValueError(f"Roundtable {roundtable_id} not found")

        rt = _active_roundtables.pop(roundtable_id)
        return rt.finalize(decision)


# -------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------

def main():
    """CLI for testing roundtable capture."""
    import argparse

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser(description="ScornSpine Roundtable - Discussion Capture")
    parser.add_argument("--test", action="store_true", help="Run test capture")
    parser.add_argument("--list", action="store_true", help="List recent roundtables")

    args = parser.parse_args()

    if args.test:
        print("?? Testing roundtable capture...")

        filepath = quick_roundtable(
            topic="test-roundtable",
            implementation="test-implementation",
            rounds=[
                {"round": 1, "agent": "JONAH", "content": "This is a test analysis from Jonah."},
                {"round": 1, "agent": "HALO", "content": "This is a test response from Halo."},
                {"round": 2, "agent": "JONAH", "content": "Follow-up analysis."},
                {"round": 2, "agent": "VERA", "content": "Strategic perspective from Vera."},
            ],
            votes=[
                {"agent": "JONAH", "approve": True, "comment": "Agreed with approach"},
                {"agent": "HALO", "approve": True, "comment": "Ready to implement"},
                {"agent": "VERA", "approve": True, "comment": "Aligned with strategy"},
            ],
            decision="Test decision: Proceed with approach X"
        )

        print(f"[OK] Test roundtable saved to: {filepath}")

    if args.list:
        print("?? Recent roundtables:")
        for impl_dir in IMPLEMENTATIONS_DIR.iterdir():
            if impl_dir.is_dir():
                rt_dir = impl_dir / "roundtables"
                if rt_dir.exists():
                    for rt_file in rt_dir.glob("*.md"):
                        print(f"  - {impl_dir.name}/{rt_file.name}")


if __name__ == "__main__":
    main()
