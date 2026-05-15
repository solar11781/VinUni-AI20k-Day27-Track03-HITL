"""Exercise 3 — Escalation branch with reviewer Q&A.

When confidence < 60%, the agent doesn't ask approve/reject — it asks specific
clarifying questions and then synthesizes a refined review from the answers.
"""

from __future__ import annotations

import argparse
import uuid

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.panel import Panel

from common.github import fetch_pr, post_review_comment
from common.llm import get_llm
from common.schemas import (
    AUTO_APPROVE_THRESHOLD,
    ESCALATE_THRESHOLD,
    PRAnalysis,
    ReviewState,
)


console = Console()


REVIEW_SYSTEM_PROMPT = """You are a senior software engineer reviewing a GitHub pull request.
Return structured output only.

Focus on correctness, security, data migrations, maintainability, and tests.
When confidence is below 60%, populate escalation_questions with 2–4 specific,
context-rich questions for the reviewer. Each question must mention the file,
function, or diff section that made you uncertain and explain what answer would
change the review.
"""

SYNTHESIZE_SYSTEM_PROMPT = """You are refining a pull-request review after a human reviewer answered
escalation questions. Return a complete PRAnalysis. Incorporate the answers,
raise confidence only when the answers genuinely reduce uncertainty, and keep
comments actionable and tied to changed files/lines when possible.
"""

EDIT_SYSTEM_PROMPT = """You are revising an automated PR review after a human selected edit.
Return a complete PRAnalysis. Keep valid findings, apply the human feedback,
remove unsupported claims, and make the final review ready to post to GitHub.
"""


def node_fetch_pr(state: ReviewState) -> dict:
    console.print("[cyan]→ fetch_pr[/cyan]")
    with console.status("[dim]Fetching PR from GitHub...[/dim]"):
        pr = fetch_pr(state["pr_url"])
    console.print(f"  [green]✓[/green] {len(pr.files_changed)} files, head {pr.head_sha[:7]}")
    return {
        "pr_title": pr.title,
        "pr_diff": pr.diff,
        "pr_files": pr.files_changed,
        "pr_head_sha": pr.head_sha,
    }


def node_analyze(state: ReviewState) -> dict:
    console.print("[cyan]→ analyze[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM reviewing the diff...[/dim]"):
        analysis = llm.invoke([
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Title: {state['pr_title']}\n"
                    f"Files changed: {', '.join(state.get('pr_files', []))}\n\n"
                    f"Diff:\n{state['pr_diff']}"
                ),
            },
        ])
    console.print(
        f"  [green]✓[/green] confidence={analysis.confidence:.0%}, "
        f"{len(analysis.escalation_questions)} question(s)"
    )
    return {"analysis": analysis}


def node_route(state: ReviewState) -> dict:
    console.print("[cyan]→ route[/cyan]")
    c = state["analysis"].confidence
    if c >= AUTO_APPROVE_THRESHOLD:
        decision = "auto_approve"
    elif c < ESCALATE_THRESHOLD:
        decision = "escalate"
    else:
        decision = "human_approval"
    console.print(f"  [green]✓[/green] decision=[bold]{decision}[/bold] (confidence={c:.0%})")
    return {"decision": decision}


def node_escalate(state: ReviewState) -> dict:
    """Ask the reviewer specific questions; return their answers in state."""
    a = state["analysis"]
    questions = a.escalation_questions or [
        "What is the intent of this PR, and which behavior should be preserved?",
        "Are there migration, security, or compatibility constraints not visible in the diff?",
    ]

    answers = interrupt({
        "kind": "escalation",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "risk_factors": a.risk_factors,
        "questions": questions,
    })
    return {"escalation_answers": answers}


def node_synthesize(state: ReviewState) -> dict:
    """Re-prompt LLM with the reviewer's answers and produce a refined review."""
    console.print("[cyan]→ synthesize[/cyan]")
    qa = "\n".join(
        f"Q: {question}\nA: {answer}"
        for question, answer in (state.get("escalation_answers") or {}).items()
    )
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]LLM refining review with reviewer answers...[/dim]"):
        refined = llm.invoke([
            {"role": "system", "content": SYNTHESIZE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Title: {state['pr_title']}\n\n"
                    f"Initial analysis:\n{state['analysis'].model_dump_json(indent=2)}\n\n"
                    f"Reviewer Q&A:\n{qa}\n\n"
                    f"Diff:\n{state['pr_diff']}"
                ),
            },
        ])
    console.print(f"  [green]✓[/green] refined confidence={refined.confidence:.0%}")
    return {"analysis": refined}


def node_human_approval(state: ReviewState) -> dict:
    a = state["analysis"]
    response = interrupt({
        "kind": "approval_request",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
    })
    return {"human_choice": response.get("choice"), "human_feedback": response.get("feedback")}


def node_refined_approval(state: ReviewState) -> dict:
    """Final confirmation after the low-confidence escalation has been refined."""
    a = state["analysis"]
    response = interrupt({
        "kind": "refined_approval",
        "pr_url": state["pr_url"],
        "confidence": a.confidence,
        "confidence_reasoning": a.confidence_reasoning,
        "summary": a.summary,
        "comments": [c.model_dump() for c in a.comments],
        "diff_preview": state["pr_diff"][:2000],
        "escalation_answers": state.get("escalation_answers") or {},
    })
    return {"human_choice": response.get("choice"), "human_feedback": response.get("feedback")}


def node_auto_edit(state: ReviewState) -> dict:
    console.print("[cyan]→ auto_edit[/cyan]")
    llm = get_llm().with_structured_output(PRAnalysis)
    with console.status("[dim]Rewriting review from human feedback...[/dim]"):
        revised = llm.invoke([
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
    return {"analysis": revised}


def _route_after_human(state: ReviewState) -> str:
    return "auto_edit" if state.get("human_choice") == "edit" else "commit"


def _render_comment_body(state: ReviewState) -> str:
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


def _post(state: ReviewState, label: str) -> str:
    try:
        post_review_comment(state["pr_url"], _render_comment_body(state))
        console.print(f"  [green]✓[/green] posted comment to {state['pr_url']}")
        return label
    except Exception as e:
        console.print(f"  [red]✗[/red] post failed: {e}")
        return "commit_failed"


def node_commit(state: ReviewState) -> dict:
    console.print("[cyan]→ commit[/cyan]")
    if state.get("human_choice") == "approve":
        label = "committed_after_escalation" if state.get("escalation_answers") else "committed"
        return {"final_action": _post(state, label)}
    if state.get("human_choice") == "edit":
        return {"final_action": _post(state, "committed_after_edit")}
    console.print(f"  [yellow]·[/yellow] skipping comment (choice={state.get('human_choice')})")
    return {"final_action": "rejected"}


def node_auto_approve(state: ReviewState) -> dict:
    console.print("[cyan]→ auto_approve[/cyan]  [dim]high confidence — posting directly[/dim]")
    return {"final_action": _post(state, "auto_approved")}


def build_graph():
    g = StateGraph(ReviewState)
    for name, fn in [
        ("fetch_pr", node_fetch_pr),
        ("analyze", node_analyze),
        ("route", node_route),
        ("auto_approve", node_auto_approve),
        ("human_approval", node_human_approval),
        ("commit", node_commit),
        ("escalate", node_escalate),
        ("synthesize", node_synthesize),
        ("refined_approval", node_refined_approval),
        ("auto_edit", node_auto_edit),
    ]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_pr")
    g.add_edge("fetch_pr", "analyze")
    g.add_edge("analyze", "route")
    g.add_conditional_edges(
        "route",
        lambda s: s["decision"],
        {"auto_approve": "auto_approve", "human_approval": "human_approval", "escalate": "escalate"},
    )
    g.add_edge("auto_approve", END)
    g.add_conditional_edges("human_approval", _route_after_human, {"auto_edit": "auto_edit", "commit": "commit"})
    g.add_edge("escalate", "synthesize")
    g.add_edge("synthesize", "refined_approval")
    g.add_conditional_edges("refined_approval", _route_after_human, {"auto_edit": "auto_edit", "commit": "commit"})
    g.add_edge("auto_edit", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=MemorySaver())


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
        console.print(Panel.fit(
            payload["summary"],
            title=f"Escalation conf={payload['confidence']:.0%}",
            border_style="yellow",
        ))
        return {q: console.input(f"Q: {q}\nA: ").strip() for q in payload["questions"]}
    raise ValueError(kind)


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--pr", required=True)
    args = p.parse_args()

    console.rule("[bold]Exercise 3 — escalation with reviewer Q&A[/bold]")
    console.print(f"[dim]PR: {args.pr}[/dim]\n")

    app = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}
    console.print(f"[dim]thread_id = {thread_id}[/dim]\n")

    result = app.invoke({"pr_url": args.pr, "thread_id": thread_id}, cfg)
    while "__interrupt__" in result:
        result = app.invoke(Command(resume=handle_interrupt(result["__interrupt__"][0].value)), cfg)

    console.rule("Final")
    console.print(f"final_action = {result.get('final_action')}")
    if "analysis" in result:
        console.print(f"final confidence = {result['analysis'].confidence:.0%}")


if __name__ == "__main__":
    main()
