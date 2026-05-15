"""Exercise 4 — Structured SQLite audit trail + durable checkpointer.

Zero setup — SQLite stores everything in a single file (`./hitl_audit.db`).
The audit_events schema is created automatically on first connection.

This completed version also implements the bonus challenges:
- time-travel helpers using ``aget_state_history``;
- confidence-calibration SQL over ``audit_events``;
- multi-reviewer escalation fan-out using LangGraph ``Send``;
- auto-edit when a human chooses ``edit``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import operator
import os
import time
import uuid
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from common.db import db_conn, db_path, write_audit_event
from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    AuditEntry,
    PRAnalysis,
    ReviewState,
    risk_level_for,
)


console = Console()
AGENT_ID = "pr-review-agent@v0.1"


class AuditReviewState(ReviewState, total=False):
    """Exercise-4 graph state with bonus fan-out aggregation fields."""

    reviewer_ids: list[str]
    target_reviewer_id: str
    fanout_answers: Annotated[list[dict[str, Any]], operator.add]
    posted_comment_url: str | None


REVIEW_SYSTEM_PROMPT = """You are a senior software engineer reviewing a GitHub pull request.
Return structured output only.

Focus on correctness, security, data migrations, maintainability, and tests.
Set confidence to how complete and correct you believe the review is.
When confidence is below 60%, populate escalation_questions with 2–4 specific,
context-rich questions for the reviewer. Each question must mention the file,
function, or diff section that made you uncertain and explain what answer would
change the review.
"""

SYNTHESIZE_SYSTEM_PROMPT = """You are refining a pull-request review after one or more reviewers answered
escalation questions. Return a complete PRAnalysis. Incorporate the answers,
raise confidence only when the answers genuinely reduce uncertainty, and keep
comments actionable and tied to changed files/lines when possible.
"""

EDIT_SYSTEM_PROMPT = """You are revising an automated PR review after a human selected edit.
Return a complete PRAnalysis. Keep valid findings, apply the human feedback,
remove unsupported claims, and make the final review ready to post to GitHub.
"""


def _elapsed_ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


def _reviewer_id() -> str | None:
    return os.environ.get("GITHUB_USER") or os.environ.get("USER") or None


def _configured_reviewers(state: AuditReviewState) -> list[str]:
    """Return reviewers from state, REVIEWER_IDS, or the current reviewer fallback."""
    state_reviewers = state.get("reviewer_ids") or []
    env_reviewers = [r.strip() for r in os.environ.get("REVIEWER_IDS", "").split(",") if r.strip()]
    reviewers = state_reviewers or env_reviewers or [_reviewer_id() or "reviewer"]
    deduped: list[str] = []
    for reviewer in reviewers:
        if reviewer and reviewer not in deduped:
            deduped.append(reviewer)
    return deduped


async def audit(state: AuditReviewState, entry: AuditEntry) -> None:
    """Write one structured AuditEntry row to the `audit_events` table."""
    await write_audit_event(
        thread_id=state["thread_id"],
        pr_url=state["pr_url"],
        entry=entry,
    )


async def node_fetch_pr(state: AuditReviewState) -> dict:
    console.print("[cyan]→ fetch_pr[/cyan]")
    t0 = time.monotonic()
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="fetch_pr",
        confidence=0.0,
        risk_level="med",
        decision="pending",
        reason=f"Fetched {len(pr.files_changed)} files, head={pr.head_sha[:7]}",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {
        "pr_title": pr.title,
        "pr_author": pr.author,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }


async def node_analyze(state: AuditReviewState) -> dict:
    console.print("[cyan]→ analyze[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        a: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Title: {state['pr_title']}\n"
                    f"Author: {state.get('pr_author', '?')}\n"
                    f"Files changed: {', '.join(state.get('pr_files', []))}\n\n"
                    f"Diff:\n{state['pr_diff']}"
                ),
            },
        ])
    console.print(f"  [green]✓[/green] confidence={a.confidence:.0%}, {len(a.comments)} comment(s)")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="analyze",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=a.confidence_reasoning,
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"analysis": a}


async def node_route(state: AuditReviewState) -> dict:
    console.print("[cyan]→ route[/cyan]")
    t0 = time.monotonic()
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="route",
        confidence=c,
        risk_level=risk_level_for(c),
        decision=decision,
        reason=f"Routed by confidence thresholds: auto>={AUTO_APPROVE_THRESHOLD:.0%}, escalate<{ESCALATE_THRESHOLD:.0%}",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"decision": decision}


async def node_prepare_human_approval(state: AuditReviewState) -> dict:
    """Audit the pending medium-risk HITL step before the interrupt node runs."""
    t0 = time.monotonic()
    a = state["analysis"]
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason="Waiting for reviewer approval, rejection, or edit instruction.",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {}


async def node_human_approval(state: AuditReviewState) -> dict:
    a = state["analysis"]
    t0 = time.monotonic()
    resp = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })
    choice = resp.get("choice")
    feedback = resp.get("feedback")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="human_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer_id(),
        decision=choice or "pending",
        reason=feedback or f"Reviewer selected {choice}.",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"human_choice": choice, "human_feedback": feedback}


def _render_comment_body(state: AuditReviewState) -> str:
    a = state["analysis"]
    lines = [f"### Automated review (confidence {a.confidence:.0%})", "", a.summary, ""]
    for c in a.comments:
        lines.append(f"- **[{c.severity}]** `{c.file}:{c.line or '?'}` — {c.body}")
    if state.get("human_feedback"):
        lines.append(f"\n_Reviewer note: {state['human_feedback']}_")
    if state.get("escalation_answers"):
        lines.append("\n_Reviewer answered escalation questions:_")
        for q, ans in state["escalation_answers"].items():
            lines.append(f"> **{q}** {ans}")
    return "\n".join(lines)


def _post(state: AuditReviewState) -> tuple[str, str, str | None]:
    body = _render_comment_body(state)
    try:
        post_review_comment(state["pr_url"], body)
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return "committed", body, state["pr_url"]
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed", body, None


async def node_commit(state: AuditReviewState) -> dict:
    console.print("[cyan]→ commit[/cyan]")
    t0 = time.monotonic()
    a = state["analysis"]

    if state.get("human_choice") in {"approve", "edit"}:
        action, body, comment_url = _post(state)
        if action == "committed" and state.get("human_choice") == "edit":
            final_action = "committed_after_edit"
        elif action == "committed" and state.get("escalation_answers"):
            final_action = "committed_after_escalation"
        else:
            final_action = action
    else:
        console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
        action, body, comment_url = "rejected", _render_comment_body(state), None
        final_action = "rejected"

    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="commit",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer_id() if state.get("human_choice") else None,
        decision=(state.get("human_choice") or "auto") if final_action != "rejected" else "reject",
        reason=f"final_action={final_action}",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"final_action": final_action, "posted_comment_body": body, "posted_comment_url": comment_url}


async def node_auto_approve(state: AuditReviewState) -> dict:
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    t0 = time.monotonic()
    a = state["analysis"]
    action, body, comment_url = _post(state)
    final_action = f"auto_{action}"
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="auto_approve",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="auto",
        reason=f"High-confidence review posted without human approval; final_action={final_action}",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"final_action": final_action, "posted_comment_body": body, "posted_comment_url": comment_url}


async def node_prepare_escalation(state: AuditReviewState) -> dict:
    """Audit and prepare reviewer fan-out before any reviewer interrupt fires."""
    t0 = time.monotonic()
    a = state["analysis"]
    reviewers = _configured_reviewers(state)
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="pending",
        reason=f"Waiting for escalation answers from {len(reviewers)} reviewer(s): {', '.join(reviewers)}",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"reviewer_ids": reviewers}


def route_to_reviewers(state: AuditReviewState) -> list[Send]:
    """Bonus: fan out the same escalation questions to each configured reviewer."""
    branch_base = {k: v for k, v in state.items() if k != "fanout_answers"}
    return [
        Send("ask_reviewer", {**branch_base, "target_reviewer_id": reviewer})
        for reviewer in _configured_reviewers(state)
    ]


async def node_ask_reviewer(state: AuditReviewState) -> dict:
    a = state["analysis"]
    t0 = time.monotonic()
    reviewer = state.get("target_reviewer_id") or "reviewer"
    questions = a.escalation_questions or [
        "What is the intent of this PR, and which behavior should be preserved?",
        "Are there migration, security, or compatibility constraints not visible in the diff?",
    ]

    answers = interrupt({
        "kind": "escalation",
        "reviewer_id": reviewer,
        "reviewer_thread_id": f"{state['thread_id']}::{reviewer}",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=reviewer,
        decision="escalate",
        reason=f"Reviewer answered {len(answers)} escalation question(s).",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"fanout_answers": [{"reviewer_id": reviewer, "answers": answers}]}


async def node_merge_escalation(state: AuditReviewState) -> dict:
    t0 = time.monotonic()
    fanout_answers = state.get("fanout_answers") or []
    merged: dict[str, str] = {}
    for item in fanout_answers:
        reviewer = item.get("reviewer_id", "reviewer")
        for question, answer in (item.get("answers") or {}).items():
            merged[f"{reviewer}: {question}"] = answer
    a = state["analysis"]
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="escalate",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        decision="escalate",
        reason=f"Collected escalation answers from {len(fanout_answers)} reviewer branch(es).",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"escalation_answers": merged}


async def node_synthesize(state: AuditReviewState) -> dict:
    console.print("[cyan]→ synthesize[/cyan]")
    t0 = time.monotonic()
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in (state.get("escalation_answers") or {}).items())
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": SYNTHESIZE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Title: {state['pr_title']}\n\n"
                    f"Original analysis:\n{state['analysis'].model_dump_json(indent=2)}\n\n"
                    f"Reviewer Q&A:\n{qa}\n\n"
                    f"Diff:\n{state['pr_diff']}"
                ),
            },
        ])
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="synthesize",
        confidence=refined.confidence,
        risk_level=risk_level_for(refined.confidence),
        decision="pending",
        reason=refined.confidence_reasoning,
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"analysis": refined}


async def node_refined_approval(state: AuditReviewState) -> dict:
    """Final reviewer confirm after escalation synthesis, required by the UI spec."""
    t0 = time.monotonic()
    a = state["analysis"]
    resp = interrupt({
        "kind": "refined_approval",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
        "escalation_answers": state.get("escalation_answers") or {},
    })
    choice = resp.get("choice")
    feedback = resp.get("feedback")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="refined_approval",
        confidence=a.confidence,
        risk_level=risk_level_for(a.confidence),
        reviewer_id=_reviewer_id(),
        decision=choice or "pending",
        reason=feedback or f"Reviewer selected {choice} after escalation synthesis.",
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"human_choice": choice, "human_feedback": feedback}


async def node_auto_edit(state: AuditReviewState) -> dict:
    console.print("[cyan]→ auto_edit[/cyan]")
    t0 = time.monotonic()
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]Rewriting review from human feedback...[/dim]"):
        revised: PRAnalysis = await llm.ainvoke([
            {"role": "system", "content": EDIT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Human feedback: {state.get('human_feedback') or '(no feedback provided)'}\n\n"
                    f"Current analysis:\n{state['analysis'].model_dump_json(indent=2)}\n\n"
                    f"Diff:\n{state['pr_diff']}"
                ),
            },
        ])
    console.print(f"  [green]✓[/green] revised confidence={revised.confidence:.0%}")
    await audit(state, AuditEntry(
        agent_id=AGENT_ID,
        action="auto_edit",
        confidence=revised.confidence,
        risk_level=risk_level_for(revised.confidence),
        reviewer_id=_reviewer_id(),
        decision="edit",
        reason=revised.confidence_reasoning,
        execution_time_ms=_elapsed_ms(t0),
    ))
    return {"analysis": revised}


def _route_after_human(state: AuditReviewState) -> str:
    return "auto_edit" if state.get("human_choice") == "edit" else "commit"


def build_graph(checkpointer):
    g = StateGraph(AuditReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr),
        ("analyze", node_analyze),
        ("route", node_route),
        ("auto_approve", node_auto_approve),
        ("prepare_human_approval", node_prepare_human_approval),
        ("human_approval", node_human_approval),
        ("auto_edit", node_auto_edit),
        ("commit", node_commit),
        ("prepare_escalation", node_prepare_escalation),
        ("ask_reviewer", node_ask_reviewer),
        ("merge_escalation", node_merge_escalation),
        ("synthesize", node_synthesize),
        ("refined_approval", node_refined_approval),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route",
        lambda s: s["decision"],
        {
            "auto_approve": "auto_approve",
            "human_approval": "prepare_human_approval",
            "escalate": "prepare_escalation",
        },
    )
    g.add_edge("auto_approve", END)
    g.add_edge("prepare_human_approval", "human_approval")
    g.add_conditional_edges("human_approval", _route_after_human, {"auto_edit": "auto_edit", "commit": "commit"})
    g.add_conditional_edges("prepare_escalation", route_to_reviewers)
    g.add_edge("ask_reviewer", "merge_escalation")
    g.add_edge("merge_escalation", "synthesize")
    g.add_edge("synthesize", "refined_approval")
    g.add_conditional_edges("refined_approval", _route_after_human, {"auto_edit": "auto_edit", "commit": "commit"})
    g.add_edge("auto_edit", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=checkpointer)


def handle_interrupt(payload: dict) -> dict:
    kind = payload["kind"]
    if kind in {"approval_request", "refined_approval"}:
        title = "Approve refined review?" if kind == "refined_approval" else "Approve?"
        console.print(Panel.fit(
            payload["summary"],
            title=f"{title} conf={payload['confidence']:.0%}",
            border_style="green",
        ))
        for c in payload.get("comments", []):
            console.print(f"  [{c['severity']}] {c['file']}:{c.get('line') or '?'} — {c['body']}")
        choice = ""
        while choice not in {"approve", "reject", "edit"}:
            choice = console.input("approve/reject/edit? ").strip().lower()
        feedback = console.input("Feedback: ").strip() if choice != "approve" else ""
        return {"choice": choice, "feedback": feedback}
    if kind == "escalation":
        reviewer = payload.get("reviewer_id", "reviewer")
        console.print(Panel.fit(
            payload["summary"],
            title=f"Escalation for {reviewer} · conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        if payload.get("risk_factors"):
            console.print("[red]Risks:[/red] " + ", ".join(payload["risk_factors"]))
        return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}
    raise ValueError(kind)


def _resume_value_from_interrupts(interrupts) -> Any:
    """Resume one or many pending interrupts.

    LangGraph requires an ``{interrupt_id: answer}`` map for simultaneous
    parallel interrupts. The fan-out escalation branch can produce exactly that.
    """
    if len(interrupts) == 1:
        return handle_interrupt(interrupts[0].value)
    resume: dict[str, Any] = {}
    for item in interrupts:
        interrupt_id = getattr(item, "id", None)
        if not interrupt_id:
            raise RuntimeError("A parallel interrupt did not expose an interrupt id.")
        resume[interrupt_id] = handle_interrupt(item.value)
    return resume


async def run(pr_url: str, thread_id: str | None, reviewers: list[str] | None = None):
    thread_id = thread_id or str(uuid.uuid4())
    console.rule("[bold]Exercise 4 — SQLite audit trail[/bold]")
    console.print(f"[dim]PR: {pr_url}[/dim]")
    console.print(f"[dim]thread_id = {thread_id}[/dim]")
    if reviewers:
        console.print(f"[dim]reviewers = {', '.join(reviewers)}[/dim]")
    console.print()

    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        initial_state: AuditReviewState = {"pr_url": pr_url, "thread_id": thread_id}
        if reviewers:
            initial_state["reviewer_ids"] = reviewers

        result = await app.ainvoke(initial_state, cfg)
        while "__interrupt__" in result:
            resume_value = _resume_value_from_interrupts(result["__interrupt__"])
            result = await app.ainvoke(Command(resume=resume_value), cfg)

        console.rule("Final")
        console.print(f"final_action = {result.get('final_action')}")
        console.print(f"\n[dim]Replay:[/dim] uv run python -m audit.replay --thread {thread_id}")
        console.print(f"[dim]History:[/dim] uv run python exercises/exercise_4_audit.py --history --thread {thread_id}")
        return result


async def show_state_history(thread_id: str) -> None:
    """Bonus: list checkpoints for time-travel."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        table = Table(title=f"Checkpoint history for {thread_id}")
        table.add_column("index")
        table.add_column("next")
        table.add_column("checkpoint_id")
        table.add_column("summary")
        idx = 0
        async for snapshot in app.aget_state_history(cfg):
            values = snapshot.values or {}
            checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id", "")
            summary = values.get("final_action") or values.get("decision") or values.get("pr_title") or ""
            table.add_row(str(idx), ",".join(snapshot.next or []), str(checkpoint_id), str(summary)[:80])
            idx += 1
        console.print(table)


async def resume_from_checkpoint(thread_id: str, checkpoint_index: int, answer_json: str) -> None:
    """Bonus: resume from an earlier checkpoint with a different reviewer answer."""
    answer = json.loads(answer_json)
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        snapshots = [snapshot async for snapshot in app.aget_state_history(cfg)]
        if checkpoint_index < 0 or checkpoint_index >= len(snapshots):
            raise IndexError(f"checkpoint index {checkpoint_index} is outside 0..{len(snapshots) - 1}")
        checkpoint_cfg = snapshots[checkpoint_index].config
        result = await app.ainvoke(Command(resume=answer), checkpoint_cfg)
        while "__interrupt__" in result:
            result = await app.ainvoke(Command(resume=_resume_value_from_interrupts(result["__interrupt__"])), checkpoint_cfg)
        console.print("[green]Time-travel branch finished[/green]")
        console.print(result.get("final_action"))


async def calibration_report() -> None:
    """Bonus: compare confidence against human approval outcomes."""
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT
                COUNT(DISTINCT thread_id) AS sessions,
                ROUND(AVG(confidence), 3) AS avg_confidence,
                SUM(CASE WHEN decision = 'approve' THEN 1 ELSE 0 END) AS approvals,
                SUM(CASE WHEN decision IN ('approve', 'reject', 'edit') THEN 1 ELSE 0 END) AS human_decisions,
                ROUND(AVG(CASE WHEN decision = 'approve' THEN confidence END), 3) AS avg_conf_when_approved,
                ROUND(AVG(CASE WHEN decision IN ('reject', 'edit') THEN confidence END), 3) AS avg_conf_when_not_approved
            FROM audit_events
            WHERE action IN ('human_approval', 'refined_approval')
            """
        ) as cur:
            row = await cur.fetchone()

        async with conn.execute(
            """
            SELECT risk_level,
                   COUNT(*) AS events,
                   ROUND(AVG(confidence), 3) AS avg_confidence,
                   SUM(CASE WHEN decision = 'approve' THEN 1 ELSE 0 END) AS approvals
              FROM audit_events
             WHERE action IN ('human_approval', 'refined_approval')
             GROUP BY risk_level
             ORDER BY risk_level
            """
        ) as cur:
            buckets = await cur.fetchall()

    table = Table(title="Confidence calibration")
    for col in row.keys():
        table.add_column(col)
    table.add_row(*(str(row[col]) for col in row.keys()))
    console.print(table)

    bucket_table = Table(title="Calibration by risk bucket")
    for col in ("risk_level", "events", "avg_confidence", "approvals"):
        bucket_table.add_column(col)
    for bucket in buckets:
        bucket_table.add_row(*(str(bucket[col]) for col in ("risk_level", "events", "avg_confidence", "approvals")))
    console.print(bucket_table)

    human_decisions = row["human_decisions"] or 0
    if human_decisions:
        approval_rate = (row["approvals"] or 0) / human_decisions
        avg_conf = row["avg_confidence"] or 0
        verdict = "over-confident" if avg_conf > approval_rate else "under-confident"
        console.print(
            f"[bold]Approval rate[/bold] = {approval_rate:.0%}; "
            f"[bold]average confidence[/bold] = {avg_conf:.0%}. "
            f"Model appears [bold]{verdict}[/bold] on the recorded HITL decisions."
        )
    else:
        console.print("[yellow]No human approval decisions recorded yet.[/yellow]")


def _parse_reviewers(value: str | None) -> list[str] | None:
    if not value:
        return None
    reviewers = [r.strip() for r in value.split(",") if r.strip()]
    return reviewers or None


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", help="GitHub pull request URL to review")
    p.add_argument("--thread", help="Resume or inspect an existing thread")
    p.add_argument("--reviewers", help="Comma-separated reviewer ids for Send fan-out escalation")
    p.add_argument("--history", action="store_true", help="List LangGraph checkpoints for a thread")
    p.add_argument("--time-travel", type=int, help="Resume from a checkpoint index listed by --history")
    p.add_argument("--answer-json", help="JSON resume value for --time-travel, e.g. '{\"choice\":\"reject\",\"feedback\":\"...\"}'")
    p.add_argument("--calibration", action="store_true", help="Show confidence calibration from audit_events")
    args = p.parse_args()

    if args.calibration:
        asyncio.run(calibration_report())
        return
    if args.history:
        if not args.thread:
            p.error("--history requires --thread")
        asyncio.run(show_state_history(args.thread))
        return
    if args.time_travel is not None:
        if not args.thread or not args.answer_json:
            p.error("--time-travel requires --thread and --answer-json")
        asyncio.run(resume_from_checkpoint(args.thread, args.time_travel, args.answer_json))
        return
    if not args.pr:
        p.error("--pr is required unless using --history, --time-travel, or --calibration")
    asyncio.run(run(args.pr, args.thread, _parse_reviewers(args.reviewers)))


if __name__ == "__main__":
    main()
