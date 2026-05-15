"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

This app wraps the audited LangGraph from exercise 4 and includes the README
bonus paths: checkpoint history/time-travel, confidence calibration,
multi-reviewer escalation fan-out, and auto-edit on reviewer feedback.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Async helpers ─────────────────────────────────────────────────────────
def arun(coro):
    """Run an async helper from Streamlit's sync execution model."""
    return asyncio.run(coro)


async def recent_sessions(limit: int = 10) -> list[dict[str, Any]]:
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MAX(timestamp) AS last_event,
                   MAX(risk_level) AS worst_risk,
                   COUNT(*) AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT ?
            """,
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def calibration_summary() -> dict[str, Any]:
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT COUNT(DISTINCT thread_id) AS sessions,
                   ROUND(AVG(confidence), 3) AS avg_confidence,
                   SUM(CASE WHEN decision = 'approve' THEN 1 ELSE 0 END) AS approvals,
                   SUM(CASE WHEN decision IN ('approve', 'reject', 'edit') THEN 1 ELSE 0 END) AS human_decisions
              FROM audit_events
             WHERE action IN ('human_approval', 'refined_approval')
            """
        ) as cur:
            row = await cur.fetchone()
    data = dict(row) if row else {}
    human_decisions = data.get("human_decisions") or 0
    approvals = data.get("approvals") or 0
    data["approval_rate"] = approvals / human_decisions if human_decisions else None
    return data


async def state_history(thread_id: str) -> list[dict[str, Any]]:
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        graph = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        rows: list[dict[str, Any]] = []
        idx = 0
        async for snapshot in graph.aget_state_history(cfg):
            values = snapshot.values or {}
            rows.append({
                "index": idx,
                "next": ", ".join(snapshot.next or []),
                "checkpoint_id": snapshot.config.get("configurable", {}).get("checkpoint_id", ""),
                "summary": values.get("final_action") or values.get("decision") or values.get("pr_title") or "",
                "config": snapshot.config,
            })
            idx += 1
    return rows


def extract_interrupts(result: dict) -> list[dict[str, Any]]:
    return [
        {"id": getattr(item, "id", None), "payload": item.value}
        for item in result.get("__interrupt__", [])
    ]


async def run_graph(pr_url: str, thread_id: str, resume_value=None, reviewers: list[str] | None = None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        graph = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            initial_state: dict[str, Any] = {"pr_url": pr_url, "thread_id": thread_id}
            if reviewers:
                initial_state["reviewer_ids"] = reviewers
            return await graph.ainvoke(initial_state, cfg)
        return await graph.ainvoke(Command(resume=resume_value), cfg)


async def run_time_travel(thread_id: str, checkpoint_index: int, resume_value: Any):
    """Resume from a selected historical checkpoint with a new answer."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        graph = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        snapshots = [snapshot async for snapshot in graph.aget_state_history(cfg)]
        if checkpoint_index < 0 or checkpoint_index >= len(snapshots):
            raise IndexError("Checkpoint index is out of range")
        return await graph.ainvoke(Command(resume=resume_value), snapshots[checkpoint_index].config)


# ─── Session state ─────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "thread_id": None,
        "pr_url": "",
        "reviewers_text": "",
        "interrupts": [],
        "final": None,
        "history_rows": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")
st.caption("LangGraph interrupt() + SQLite audit trail + Streamlit approval UI")


# ─── Sidebar — recent sessions, calibration, time-travel ───────────────────
with st.sidebar:
    st.header("Recent sessions")
    try:
        sessions = arun(recent_sessions())
    except Exception as exc:
        st.caption(f"No audit table yet: {exc}")
        sessions = []

    if not sessions:
        st.caption("No sessions recorded yet.")
    for i, session in enumerate(sessions):
        label = f"{session['thread_id'][:8]} · {session['worst_risk']} · {session['events']} events"
        if st.button(label, key=f"session_{i}", use_container_width=True):
            st.session_state.thread_id = session["thread_id"]
            st.session_state.pr_url = session["pr_url"]
            st.session_state.interrupts = []
            st.session_state.final = None
            st.rerun()
        st.caption(session["pr_url"])

    with st.expander("Confidence calibration", expanded=False):
        cal = arun(calibration_summary())
        if cal.get("human_decisions"):
            st.metric("Human decisions", int(cal["human_decisions"]))
            st.metric("Average confidence", f"{float(cal['avg_confidence']):.0%}")
            st.metric("Approval rate", f"{float(cal['approval_rate']):.0%}")
            verdict = "over-confident" if float(cal["avg_confidence"] or 0) > float(cal["approval_rate"] or 0) else "under-confident"
            st.caption(f"Calibration read: model appears **{verdict}** on recorded HITL decisions.")
        else:
            st.caption("Run a few HITL sessions first.")

    with st.expander("Time-travel", expanded=False):
        if st.session_state.thread_id:
            if st.button("Load checkpoints", use_container_width=True):
                st.session_state.history_rows = arun(state_history(st.session_state.thread_id))
            if st.session_state.history_rows:
                selected = st.selectbox(
                    "Checkpoint",
                    st.session_state.history_rows,
                    format_func=lambda r: f"#{r['index']} next={r['next'] or '-'} · {str(r['summary'])[:32]}",
                )
                answer_text = st.text_area(
                    "Resume answer JSON",
                    value='{"choice":"reject","feedback":"Testing an alternate outcome"}',
                )
                if st.button("Resume from checkpoint", type="primary"):
                    try:
                        answer = json.loads(answer_text)
                        result = arun(run_time_travel(st.session_state.thread_id, selected["index"], answer))
                        if "__interrupt__" in result:
                            st.session_state.interrupts = extract_interrupts(result)
                            st.session_state.final = None
                        else:
                            st.session_state.interrupts = []
                            st.session_state.final = result
                        st.rerun()
                    except Exception as exc:
                        st.error(str(exc))
        else:
            st.caption("Start or select a session first.")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    reviewers_text = st.text_input(
        "Escalation reviewers (optional, comma-separated)",
        value=st.session_state.reviewers_text,
        help="When low confidence triggers escalation, the graph fans out to every reviewer with LangGraph Send.",
    )
    submitted = st.form_submit_button("Run review")


def parse_reviewers(value: str) -> list[str] | None:
    reviewers = [r.strip() for r in value.split(",") if r.strip()]
    return reviewers or None


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict, *, key_prefix: str = "approval") -> dict | None:
    """58–72% bucket or refined escalation: show review + 3 buttons."""
    kind = payload.get("kind")
    conf = payload["confidence"]
    title = "Refined review confirmation" if kind == "refined_approval" else "Approval requested"
    st.subheader(f"{title} — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    if payload.get("escalation_answers"):
        with st.expander("Escalation answers used for this refined review", expanded=True):
            for question, answer in payload["escalation_answers"].items():
                st.markdown(f"**{question}**")
                st.write(answer)

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_area("Feedback (needed for Edit; optional otherwise)", key=f"{key_prefix}_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary", key=f"{key_prefix}_approve"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject", key=f"{key_prefix}_reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit", key=f"{key_prefix}_edit"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict, *, key_prefix: str = "escalation") -> dict | None:
    """< 58% bucket: show risk factors + question form."""
    conf = payload["confidence"]
    reviewer = payload.get("reviewer_id", "reviewer")
    st.subheader(f"Strong escalation for {reviewer} — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("Risks: " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form(f"{key_prefix}_form"):
        answers: dict[str, str] = {}
        for i, question in enumerate(payload["questions"]):
            answers[question] = st.text_area(question, key=f"{key_prefix}_q_{i}")
        submitted_answers = st.form_submit_button("Submit answers", type="primary")
    if submitted_answers:
        return answers
    return None


def render_multi_interrupts(interrupts: list[dict[str, Any]]) -> Any | None:
    """Render one or many pending interrupts and return the proper resume payload."""
    if len(interrupts) == 1:
        payload = interrupts[0]["payload"]
        if payload["kind"] in {"approval_request", "refined_approval"}:
            return render_approval_card(payload, key_prefix=f"single_{payload['kind']}")
        return render_escalation_card(payload, key_prefix="single_escalation")

    st.subheader(f"{len(interrupts)} reviewer escalations are pending")
    st.caption("Submit all reviewer answers together so LangGraph can resume all parallel Send branches.")
    resume_map: dict[str, Any] = {}
    with st.form("multi_interrupt_form"):
        for idx, item in enumerate(interrupts):
            interrupt_id = item["id"]
            payload = item["payload"]
            reviewer = payload.get("reviewer_id", f"reviewer-{idx + 1}")
            st.markdown(f"### {reviewer}")
            st.caption(payload.get("confidence_reasoning", ""))
            if payload.get("risk_factors"):
                st.error("Risks: " + ", ".join(payload["risk_factors"]))
            st.markdown(payload.get("summary", ""))
            answers: dict[str, str] = {}
            for q_idx, question in enumerate(payload.get("questions", [])):
                answers[question] = st.text_area(question, key=f"multi_{idx}_{q_idx}")
            if not interrupt_id:
                st.warning("This interrupt has no id; update LangGraph if multi-resume fails.")
            else:
                resume_map[interrupt_id] = answers
        submitted_all = st.form_submit_button("Submit all reviewer answers", type="primary")
    return resume_map if submitted_all else None


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.reviewers_text = reviewers_text
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupts = []
    st.session_state.final = None
    st.session_state.history_rows = []

    with st.spinner("Fetching PR + asking the LLM..."):
        result = arun(run_graph(pr_url, st.session_state.thread_id, reviewers=parse_reviewers(reviewers_text)))

    if "__interrupt__" in result:
        st.session_state.interrupts = extract_interrupts(result)
    else:
        st.session_state.final = result


# Render current interrupt card(s), if any.
if st.session_state.interrupts:
    answer = render_multi_interrupts(st.session_state.interrupts)
    if answer is not None:
        with st.spinner("Resuming graph..."):
            result = arun(run_graph(
                st.session_state.pr_url,
                st.session_state.thread_id,
                resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupts = extract_interrupts(result)
        else:
            st.session_state.interrupts = []
            st.session_state.final = result
        st.rerun()


# Render final state, if reached.
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"✓ {action} — comment posted")
        link = final.get("posted_comment_url") or st.session_state.pr_url
        st.markdown(f"[View comment on GitHub]({link})")
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")

    with st.expander("Posted comment body / final review", expanded=False):
        st.markdown(final.get("posted_comment_body") or "No comment body available.")

    st.caption(
        f"thread_id = {st.session_state.thread_id} · replay: "
        f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )
