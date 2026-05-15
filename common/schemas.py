"""Shared types: the graph state, the LLM's structured analysis output,
and the structured audit-trail entry written to PostgreSQL."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, TypedDict

from pydantic import BaseModel, Field


Decision = Literal["auto_approve", "human_approval", "escalate"]
HumanChoice = Literal["approve", "reject", "edit"]


# Confidence thresholds
AUTO_APPROVE_THRESHOLD = 0.90   # >= 90% → no human, agent commits the comment directly
ESCALATE_THRESHOLD = 0.58       # < 58% → escalate: agent asks the reviewer specific questions
# 58–90% inclusive → human approval flow (reviewer clicks Approve / Reject / Edit)


def risk_level_for(confidence: float) -> str:
    """Map confidence → risk_level (used in AuditEntry).

    The thresholds invert: higher confidence ↔ lower risk.
    """
    if confidence >= AUTO_APPROVE_THRESHOLD:
        return "low"
    if confidence < ESCALATE_THRESHOLD:
        return "high"
    return "med"


class ReviewComment(BaseModel):
    """A single review comment the agent proposes."""

    file: str = Field(description="Path of the file the comment is about")
    line: int | None = Field(None, description="Line number, when known")
    severity: Literal["nit", "suggestion", "issue", "blocker"]
    body: str


class PRAnalysis(BaseModel):
    """LLM-structured output of the analyzer node."""

    summary: str = Field(description="One-paragraph description of what the PR does")
    risk_factors: list[str] = Field(default_factory=list)
    comments: list[ReviewComment] = Field(default_factory=list)
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Self-reported confidence that the review is complete and correct",
    )
    confidence_reasoning: str = Field(
        description="Why the model picked that confidence value"
    )
    escalation_questions: list[str] = Field(
        default_factory=list,
        description="Specific questions to ask a human reviewer if escalating",
    )


class AuditEntry(BaseModel):
    """One row of the PostgreSQL audit trail.

    Designed as a structured *decision log* — one entry per meaningful event
    in a review session (LLM analysis, HITL interrupt, reviewer response,
    final commit). The fields are first-class SQL columns so auditors can
    query directly (e.g. ``SELECT AVG(confidence) WHERE decision = 'approve'``).
    """

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the event was recorded (UTC).",
    )
    agent_id: str = Field(
        description="Identifier of the agent that produced the event "
                    "(e.g. 'pr-review-agent@v0.1')."
    )
    action: str = Field(
        description="What the agent did at this step — "
                    "'fetch_pr' | 'analyze' | 'route' | 'human_approval' | "
                    "'escalate' | 'synthesize' | 'commit'."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Current confidence at this step (mirrors PRAnalysis.confidence).",
    )
    risk_level: str = Field(
        description="Derived from confidence — 'low' / 'med' / 'high'. "
                    "Use risk_level_for(confidence) to compute.",
    )
    reviewer_id: str | None = Field(
        default=None,
        description="GitHub username of the human reviewer for HITL events. "
                    "None for fully automated steps.",
    )
    decision: str = Field(
        description="Outcome at this step — "
                    "'auto' | 'approve' | 'reject' | 'edit' | 'escalate' | 'pending'.",
    )
    reason: str | None = Field(
        default=None,
        description="Free-text explanation: confidence_reasoning for analyze, "
                    "human_feedback for HITL, etc.",
    )
    execution_time_ms: int = Field(
        ge=0,
        description="Wall-clock duration of the action in milliseconds.",
    )


class ReviewState(TypedDict, total=False):
    """LangGraph state — every node reads and updates this dict."""

    # Inputs
    pr_url: str
    thread_id: str

    # Populated by fetch_pr
    pr_title: str
    pr_author: str
    pr_diff: str
    pr_files: list[str]
    pr_head_sha: str

    # Populated by analyze
    analysis: PRAnalysis

    # Populated by route_by_confidence
    decision: Decision

    # Populated by HITL nodes
    human_choice: HumanChoice | None
    human_feedback: str | None
    escalation_answers: dict[str, str] | None

    # Populated by commit / final nodes
    posted_comment_body: str | None
    final_action: str | None
