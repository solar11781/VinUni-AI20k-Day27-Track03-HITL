# Day27 — Track 3: HITL PR Review Agent

**Tên:** Bùi Trần Gia Bảo  
**Mã HV:** 2A202600009

A 2-hour lab that builds a human-in-the-loop pull-request review agent in **LangGraph**, end-to-end.

> 1. Agent reads a PR, analyzes code changes, proposes review comments.
> 2. Confidence ~72% → show diff + reasoning → user approves → agent commits the review.
> 3. Confidence ~58% → escalate: show context + specific questions for the reviewer.
> 4. Every interaction is written to a SQLite audit trail; full sessions can be replayed.

Students complete **4 exercises** (`exercises/`) that together build the full system. Each exercise has a runnable skeleton with `# TODO:` markers — your job is to fill them in.

## Background — what problem does this solve?

Code review is the bottleneck of most engineering teams: a senior engineer's time is finite, but PRs keep coming. PRs fall into roughly three buckets:

- **Mechanical** (typo fixes, dependency bumps, one-line refactors) — safe, but still need someone to click "Approve". Spending senior time here is pure waste.
- **Medium risk** (small features, schema additions) — usually fine, but a pair of human eyes catches edge cases. Reviewer time here pays off.
- **High risk** (auth, migrations, anything security-touching) — definitely need a human, often with _specific questions_ about intent and threat model.

If you apply the same human-driven review process to all three, you burn senior bandwidth on trivial PRs **and** you also fail to give risky PRs the careful attention they deserve.

### The HITL solution

This lab builds an agent that auto-triages every PR:

1. The LLM reads the diff and produces a structured review with a **self-reported confidence score**.
2. **Confidence > 72%** → the agent posts the review comment itself; no human in the loop.
3. **58–72%** → the agent pauses (`interrupt()`) and shows the reviewer the diff + LLM reasoning + a one-click **approve / reject / edit** panel. On approve, the comment posts.
4. **< 58%** → the agent **escalates strongly**. It shows the reviewer _specific questions the LLM is uncertain about_ ("Why MD5? Is `SYNC_URL` meant to be HTTPS in production?") and waits for answers. Then it re-asks the LLM with the answers in context and posts a **refined** review.

Every step writes a row to a structured audit table — every routing decision, every human interaction, every LLM call — so the team can replay any session for compliance or debugging.

### Flow diagram

```mermaid
flowchart TD
    Fetch[fetch_pr] --> Analyze[analyze]
    Analyze --> Route{confidence}
    Route -- "> 72%" --> Auto[auto_approve]
    Route -- "58–72%" --> Human["human_approval<br/>(interrupt)"]
    Route -- "< 58%" --> Escalate["escalate<br/>(interrupt)"]
    Escalate --> Synth[synthesize]
    Auto --> Commit[commit]
    Human --> Commit
    Synth --> Commit
```

Every node also writes one row to `audit_events`. The graph state is persisted by `AsyncSqliteSaver`, so an interrupted session can resume from exactly where it stopped — even after the process is killed.

## Lab assignment

**Goal:** Build a HITL agent with LangGraph `interrupt()` **+ a Streamlit approval UI**.

**Deliverables:**

- HITL agent
- Approval UI (Streamlit)
- Confidence-based routing
- PostgreSQL audit trail

**What this scaffolding already covers.** Exercises 1–4 deliver the HITL agent, confidence-based routing, and the structured audit trail. Two implementation notes:

- The lab uses **SQLite** (`./hitl_audit.db`) instead of PostgreSQL for zero-setup. The schema is row-oriented with first-class columns — the same `AuditEntry` queries transfer to Postgres in production by swapping the checkpointer and connection string.
- The approval UI is currently a terminal panel (Rich). The Streamlit UI is the fifth exercise — see [Exercise 5](#exercise-5--streamlit-approval-ui-apppy) below.

## Confidence routing

| Confidence | Branch           | Has HITL?                       | Demo PR           |
| ---------- | ---------------- | ------------------------------- | ----------------- |
| > 72%      | `auto_approve`   | No — agent commits directly     | (use any tiny PR) |
| 58–72%     | `human_approval` | Yes — approve / reject / edit   | PR-Demo #1        |
| < 58%      | `escalate`       | Yes — answer specific questions | PR-Demo #2        |

Thresholds live in `common/schemas.py` (`AUTO_APPROVE_THRESHOLD = 0.73`, `ESCALATE_THRESHOLD = 0.58`).

## Layout

```
.
├── README.md                # this file — the only documentation
├── pyproject.toml
├── .env.example
│
├── common/                  # shared utilities — provided, don't modify
│   ├── llm.py               # ChatOpenAI factory (OpenRouter)
│   ├── github.py            # httpx REST API client: fetch_pr / post_review_comment
│   ├── db.py                # aiosqlite connection + write_audit_event
│   └── schemas.py           # ReviewState (TypedDict) + PRAnalysis + AuditEntry (Pydantic)
│
├── exercises/               # YOUR WORK — skeleton code with TODOs
│   ├── exercise_1_confidence.py
│   ├── exercise_2_hitl.py
│   ├── exercise_3_escalation.py
│   └── exercise_4_audit.py
│
└── audit/                   # provided
    ├── schema.sql           # CREATE TABLE audit_events
    └── replay.py            # `python -m audit.replay --thread <id>` / --list
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- An [OpenRouter](https://openrouter.ai/keys) API key (free tier has $1 credit)
- A GitHub Personal Access Token — see next section

No Docker, no Postgres install, no `gh` CLI. The agent talks to GitHub directly via the REST API. Audit + checkpointer state both live in a single SQLite file (`./hitl_audit.db`) created automatically on first run.

## Get a GitHub Personal Access Token (PAT)

The agent calls the GitHub REST API directly to **read PR diffs** and **post review comments**. A PAT is required.

1. Open <https://github.com/settings/tokens/new> (classic — simplest).
2. Fill in:
   - **Note:** `Day27 HITL Lab`
   - **Expiration:** 30 days
   - **Scopes:** tick **`public_repo`** (enough for `VinUni-AI20k/PR-Demo` since it's a public repo).
3. Click **Generate token** → copy the `ghp_...` value immediately (shown only once).
4. Paste into `.env`:
   ```bash
   echo "GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" >> .env
   ```
5. Verify:
   ```bash
   curl -sH "Authorization: Bearer $(grep ^GITHUB_TOKEN .env | cut -d= -f2)" \
        https://api.github.com/repos/VinUni-AI20k/PR-Demo | grep '"full_name"'
   ```
   If you see `"full_name": "VinUni-AI20k/PR-Demo"` → OK. `401` → wrong scope or typo.

**Security:** `.env` is git-ignored. Each student uses **their own** PAT — comments show their username, so the instructor can see who approved what.

## Setup (once)

```bash
git clone <lab-repo-url> && cd Day27-Track3-HITL
uv sync                                     # install Python deps
cp .env.example .env && $EDITOR .env        # set OPENROUTER_API_KEY and GITHUB_TOKEN
```

Recommended optional variables:

```bash
# Used in audit rows for reviewer identity.
GITHUB_USER=your_github_username

# Optional custom SQLite path.
HITL_DB_PATH=hitl_audit.db

# Bonus multi-reviewer fan-out.
# Comma-separated reviewer labels/usernames.
# Used only by the bonus Send/fan-out implementation.
REVIEWER_IDS=reviewer1,reviewer2
```

The SQLite file `hitl_audit.db` is created on demand when exercise 4 runs.

## Demo PRs

Two PRs are kept at the public repo **<https://github.com/VinUni-AI20k/PR-Demo>** for testing:

| PR                               | URL                                              | Expected branch         |
| -------------------------------- | ------------------------------------------------ | ----------------------- |
| #1 — Add task priority field     | `https://github.com/VinUni-AI20k/PR-Demo/pull/1` | `human_approval` (~65%) |
| #2 — Add user login + cloud sync | `https://github.com/VinUni-AI20k/PR-Demo/pull/2` | `escalate` (<58%)       |

PR #1 is mechanical (~40 lines) with one open question (schema migration). PR #2 has multiple red flags (MD5 password hashing, plaintext token storage, SQL injection in `Storage.add`, hard-coded user_id, no tests for new code) — the LLM should drop into the escalate branch and ask the reviewer specific questions.

## The 5 exercises

Do them in order — each builds on the previous. Open the file, read all `# TODO:` comments top-to-bottom, fill them in, then run the command shown.

### Exercise 1 — Confidence routing (`exercise_1_confidence.py`)

Build a small LangGraph that fetches a PR, analyzes it, then routes to one of three terminal nodes by confidence (`auto_approve` / `human_approval` / `escalate`). No HITL yet — terminals just print which branch fired.

**You implement:** `node_analyze` (call LLM with `with_structured_output(PRAnalysis)`), `node_route` (return decision string), and the graph wiring (`add_node`, `add_edge`, `add_conditional_edges`).

```bash
uv run python exercises/exercise_1_confidence.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
uv run python exercises/exercise_1_confidence.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
```

Success = the two PRs print **different** branches.

### Exercise 2 — HITL with `interrupt()` (`exercise_2_hitl.py`)

Turn the `human_approval` node from a placeholder into a real pause-and-ask. Use `interrupt(payload)` inside the node, then resume the graph from `main()` with `Command(resume=<user_answer>)`.

**You implement:** `node_human_approval` (call `interrupt()` with diff + reasoning), graph compile with `MemorySaver` checkpointer, and the resume loop in `main()`:

```python
while "__interrupt__" in result:
    payload = result["__interrupt__"][0].value
    answer = prompt_human(payload)
    result = app.invoke(Command(resume=answer), cfg)
```

```bash
uv run python exercises/exercise_2_hitl.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
# → terminal pauses, shows a green panel, asks approve / reject / edit
```

Depending on the model/provider, the LLM may become overconfident and route
directly to `auto_approve` instead of `human_approval`.

If that happens, temporarily raise:

```python
AUTO_APPROVE_THRESHOLD = 0.90
```

in `common/schemas.py`, rerun the exercise, then restore it back to `0.73`
after testing.

### Exercise 3 — Escalation with reviewer Q&A (`exercise_3_escalation.py`)

When confidence is low, the agent shouldn't ask "approve/reject" — it should ask **specific questions** to gather context, then re-run the analysis with the answers.

**You implement:** prompt the LLM to populate `escalation_questions` when low-confidence, `node_escalate` (calls `interrupt(kind="escalation")` with the questions), and `node_synthesize` (re-prompts the LLM using the reviewer's answers to produce a refined review).

```bash
uv run python exercises/exercise_3_escalation.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
# → terminal shows a yellow panel; type an answer for each question
```

### Exercise 4 — Structured SQLite audit trail (`exercise_4_audit.py`)

Use `AsyncSqliteSaver` so the graph can resume after a crash, **and** at every meaningful node event, write one structured row to `audit_events` using the `AuditEntry` Pydantic model defined in `common/schemas.py`:

```python
class AuditEntry(BaseModel):
    timestamp: datetime          # auto: datetime.now(timezone.utc)
    agent_id: str                # e.g. "pr-review-agent@v0.1"
    action: str                  # "fetch_pr" | "analyze" | "route" | "human_approval" | ...
    confidence: float            # 0.0 – 1.0
    risk_level: str              # "low" | "med" | "high"  (use risk_level_for(confidence))
    reviewer_id: str | None      # GitHub username on HITL steps, None otherwise
    decision: str                # "auto" | "approve" | "reject" | "edit" | "escalate" | "pending"
    reason: str | None           # LLM reasoning OR human feedback
    execution_time_ms: int       # measure with time.monotonic() per node
```

Each field maps to a first-class SQL column, so auditors can query directly:

```sql
SELECT AVG(confidence) FROM audit_events WHERE decision = 'approve';
SELECT * FROM audit_events WHERE risk_level = 'high' AND decision = 'auto';   -- shouldn't happen!
```

**You implement:** the `audit(...)` helper (calls `write_audit_event(thread_id=..., pr_url=..., entry=AuditEntry(...))`) and construct an `AuditEntry` in every node. The `AsyncSqliteSaver` and resume loop are already wired for you in `run()`.

```bash
uv run python exercises/exercise_4_audit.py --pr https://github.com/VinUni-AI20k/PR-Demo/pull/1
# → after the run, replay the full session:
uv run python -m audit.replay --thread <thread_id-printed-above>
uv run python -m audit.replay --list   # see recent threads
```

#### Why two storage mechanisms?

| Mechanism                           | Purpose                               | Schema                          | Audience          |
| ----------------------------------- | ------------------------------------- | ------------------------------- | ----------------- |
| LangGraph **checkpointer** (SQLite) | Resume graph after crash, time-travel | Binary blob                     | LangGraph runtime |
| Table **`audit_events`**            | Structured decision log               | First-class columns (queryable) | Auditors, humans  |

Both live in the same `./hitl_audit.db` file. The assignment line "ghi vào audit trail" is fulfilled by `audit_events`. The checkpointer is a free bonus from using `AsyncSqliteSaver`.

### Exercise 5 — Streamlit approval UI (`app.py`)

The final assembly: wrap the LangGraph from exercises 1–4 into a **Streamlit** web UI so a reviewer can drive the whole flow from a browser instead of a terminal.

The same three confidence buckets apply, but the _human experience_ changes per bucket:

| Confidence | Branch           | What the reviewer sees in the UI                                                                                                                                                                  |
| ---------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **> 72%**  | `auto_approve`   | A success card: confidence + LLM summary + a "View comment on GitHub" link. Reviewer does **nothing** — agent already posted.                                                                     |
| **58–72%** | `human_approval` | The normal approval card: diff, LLM reasoning, list of comments + three buttons **Approve / Reject / Edit**. One click and the comment posts.                                                     |
| **< 58%**  | `escalate`       | The strong-escalation card: risk factors highlighted + a form with the LLM's specific questions. Reviewer fills answers, agent re-synthesizes, then shows the refined review for a final confirm. |

**You implement** (`app.py` at the repo root):

- A PR URL input form → trigger the graph.
- An `interrupt()` handler that renders the right card depending on `payload["kind"]` (`approval_request` vs `escalation`).
- A resume call: `app.ainvoke(Command(resume=user_choice_or_answers), cfg)`.
- `thread_id` persisted in `st.session_state` so reloading the page or jumping back to it keeps the session.
- A small "recent sessions" sidebar driven by `audit.replay.list_threads`.
- Add `streamlit` to `pyproject.toml` dependencies.

A skeleton with `# TODO:` markers is provided in `app.py`. Run with:

```bash
uv run streamlit run app.py
```

The skeleton wires up the Streamlit boilerplate (session state, page layout, top-level form). You fill in:

- The graph invocation
- The two interrupt renderers (`render_approval_card`, `render_escalation_card`)
- The resume logic

## How the assignment maps to files

| Assignment line                                 | Where you implement it                                       |
| ----------------------------------------------- | ------------------------------------------------------------ |
| Agent reads PR, analyzes, proposes comments     | `exercise_1_confidence.py` (analyze + routing)               |
| Confidence-based routing (72% vs 58% etc.)      | `common/schemas.py` thresholds + `exercise_1`                |
| 72% → diff + reasoning → user approves → commit | `exercise_2_hitl.py`                                         |
| 58% → escalate with specific questions          | `exercise_3_escalation.py`                                   |
| Audit trail + replay full session               | `exercise_4_audit.py`, `audit/schema.sql`, `audit/replay.py` |
| Streamlit approval UI (final assembly)          | `app.py`                                                     |

## Bonus challenges

1. **Time-travel.** Use `app.aget_state_history(config)` to list every checkpoint. Resume from an earlier checkpoint with a different reviewer answer and compare outcomes.
   List checkpoints for a thread:

   ```bash
   uv run python exercises/exercise_4_audit.py --history --thread <thread_id>
   ```

   Resume from a checkpoint with a different reviewer answer:

   ```bash
   uv run python exercises/exercise_4_audit.py \
     --time-travel <checkpoint_index> \
     --thread <thread_id> \
     --answer-json '{"choice":"reject","feedback":"Testing alternate outcome"}'
   ```

2. **Confidence calibration.** After N sessions, query `audit_events` to compute `AVG(confidence)` vs `count(human_choice = 'approve')`. Is the model over- or under-confident?
   Compute aggregate approval/confidence statistics from `audit_events`:

   ```bash
   uv run python exercises/exercise_4_audit.py --calibration
   ```

3. **Multi-reviewer fan-out.** Use the `Send` API to escalate the same questions to two reviewers (two threads) and wait for both.
   Escalate the same PR to multiple reviewers using LangGraph `Send`:

   ```bash
   uv run python exercises/exercise_4_audit.py \
     --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2 \
     --reviewers reviewer1,reviewer2
   ```

   Or via environment variable:

   ```bash
   REVIEWER_IDS=reviewer1,reviewer2 \
   uv run python exercises/exercise_4_audit.py \
     --pr https://github.com/VinUni-AI20k/PR-Demo/pull/2
   ```

4. **Auto-edit.** When the human chooses `edit`, call the LLM again to rewrite the review using `human_feedback`, then commit.
   In the HITL approval flow, choose `edit` and provide reviewer feedback.
   The graph calls the LLM again to rewrite the review using the feedback,
   then commits the revised review comment.

## FAQ / gotchas

**`interrupt()` raises `GraphInterrupt` instead of pausing.**
You forgot `checkpointer=...` in `.compile()`. Checkpointer is required for `interrupt()` to work.

**Graph restarts from scratch on resume.**
`thread_id` mismatch between the original `invoke` and the resuming `Command` call. Always print `thread_id` and use the same one when resuming.

**Duplicate review comments posted after resume.**
Side-effects placed _before_ `interrupt()` in the same node — the node re-runs from the top after resume. Move side-effects to a downstream node (e.g. `commit`).

**`post_review_comment` returns 403 / 404.**
Token lacks `public_repo` scope, or repo is private and you only have `public_repo`. Re-create the PAT with the right scope.

**LLM stays overconfident on PR #2 and never escalates.**
Temporarily raise `ESCALATE_THRESHOLD` in `common/schemas.py` to 0.70 to force the branch.

**`audit_events` schema is stale after a code change.**
The schema is created idempotently on first connection. To reset state completely, just delete the file:

```bash
rm hitl_audit.db
```

**Open the SQLite file to inspect manually.**

```bash
sqlite3 hitl_audit.db "SELECT action, confidence, decision, reviewer_id FROM audit_events ORDER BY id;"
```

Reset all checkpoint and audit state:

```bash
rm -f hitl_audit.db
```
