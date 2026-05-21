import datetime
import json
import os
import re
import sys
import threading
from pathlib import Path

import streamlit as st

# Ensure root path is accessible when running from ui/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.loader import Loader
from model.agent import SLMAgent
from model.inference import PeftAdapterInferenceEngine
from model.runtime import DEFAULT_BASE_MODEL, resolve_preferred_adapter
from rag.repo_map import RepoMapGenerator
from rag.retriever import HybridRetriever
from safety.hallucination import HallucinationDetector


ROOT_DIR = Path(__file__).resolve().parent.parent
FEEDBACK_FILE = ROOT_DIR / "human_feedback.jsonl"
TRAINING_DATA_FILE = ROOT_DIR / "preference_data_combined.jsonl"
HALLUCINATION_WARNING_PATTERN = re.compile(
    r"\s*⚠️ Hallucination Warning:\s*.*?\[[^\]]*\]\s*$",
    re.DOTALL,
)


st.set_page_config(page_title="Mobtrap", page_icon="code", layout="wide")


def initialize_session_state():
    """Initialize all session state keys used by the UI."""
    defaults = {
        "messages": [],
        "feedback_given": set(),
        "feedback_lock": threading.Lock(),
        "backend": None,
        "agent": None,
        "hallucination_detector": None,
        "retriever_mode": "unknown",
        "chunk_count": 0,
        "correction_pending": {},
        "target_repo": str(ROOT_DIR),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def load_backend(repo_path):
    """Load the backend agent, repository map, and chunks for a repository path."""
    loader = Loader(repo_path)
    chunks = loader.process_directory()
    repo_map_gen = RepoMapGenerator()
    repo_map_string = repo_map_gen.generate_map(chunks)

    retriever = HybridRetriever()
    retriever.ingest_chunks(chunks)

    adapter_path, adapter_label = resolve_preferred_adapter(ROOT_DIR)
    inference_engine = PeftAdapterInferenceEngine(
        base_model=DEFAULT_BASE_MODEL,
        adapter_path=adapter_path,
        device="auto",
    )
    agent = SLMAgent(
        repo_map_string=repo_map_string,
        hallucination_check=True,
        inference_engine=inference_engine,
        model=f"{DEFAULT_BASE_MODEL} + {adapter_label}",
    )
    agent.set_retriever(retriever)
    agent.set_chunks(chunks)

    return agent, repo_map_string, chunks


def strip_hallucination_warning(response):
    """Remove the agent's internal hallucination warning suffix from a response."""
    return HALLUCINATION_WARNING_PATTERN.sub("", response or "").rstrip()


def get_retriever_mode(agent):
    """Return the active retriever mode label from the attached retriever."""
    retriever = getattr(agent, "retriever", None)
    if retriever is None:
        return "unknown"
    if getattr(retriever, "qdrant_ready", False):
        return "hybrid"
    return "lexical"


def iso_timestamp():
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def get_previous_user_prompt(messages, assistant_index):
    """Return the nearest preceding user message content for an assistant message."""
    for index in range(assistant_index - 1, -1, -1):
        message = messages[index]
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def append_jsonl_row_locked(path, row):
    """Append a JSON row to a JSONL file while holding the shared feedback lock."""
    lock = st.session_state.feedback_lock
    serialized = json.dumps(row, ensure_ascii=False)
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def save_feedback_row(row):
    """Persist a single human feedback row and report any storage failures."""
    try:
        append_jsonl_row_locked(FEEDBACK_FILE, row)
        return True
    except FileNotFoundError:
        st.error("Failed to save feedback. Check file permissions.")
    except PermissionError:
        st.error("Failed to save feedback. Check file permissions.")
    except json.JSONDecodeError:
        st.error("Failed to save feedback. Check file permissions.")
    except OSError:
        st.error("Failed to save feedback. Check file permissions.")
    return False


@st.cache_data(ttl=10)
def read_feedback_stats():
    """Read feedback rows and compute dashboard statistics."""
    try:
        if not FEEDBACK_FILE.exists():
            return {
                "rows": [],
                "total": 0,
                "positive": 0,
                "negative": 0,
                "corrections": 0,
                "positive_ratio": 0.0,
                "risk_breakdown": {"low": 0, "medium": 0, "high": 0},
                "error": None,
            }

        rows = []
        with FEEDBACK_FILE.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                rows.append(json.loads(stripped))

        positive = sum(1 for row in rows if row.get("rating") == "positive")
        negative = sum(1 for row in rows if row.get("rating") == "negative")
        corrections = sum(1 for row in rows if row.get("correction_provided") is True)
        total = len(rows)
        risk_breakdown = {"low": 0, "medium": 0, "high": 0}
        for row in rows:
            risk = row.get("hallucination_risk")
            if risk in risk_breakdown:
                risk_breakdown[risk] += 1

        return {
            "rows": rows,
            "total": total,
            "positive": positive,
            "negative": negative,
            "corrections": corrections,
            "positive_ratio": (positive / total) if total else 0.0,
            "risk_breakdown": risk_breakdown,
            "error": None,
        }
    except FileNotFoundError:
        return {
            "rows": [],
            "total": 0,
            "positive": 0,
            "negative": 0,
            "corrections": 0,
            "positive_ratio": 0.0,
            "risk_breakdown": {"low": 0, "medium": 0, "high": 0},
            "error": None,
        }
    except PermissionError:
        return {
            "rows": [],
            "total": 0,
            "positive": 0,
            "negative": 0,
            "corrections": 0,
            "positive_ratio": 0.0,
            "risk_breakdown": {"low": 0, "medium": 0, "high": 0},
            "error": "Feedback file could not be read due to permissions.",
        }
    except json.JSONDecodeError:
        return {
            "rows": [],
            "total": 0,
            "positive": 0,
            "negative": 0,
            "corrections": 0,
            "positive_ratio": 0.0,
            "risk_breakdown": {"low": 0, "medium": 0, "high": 0},
            "error": "Feedback file contains invalid JSON.",
        }


def merge_human_feedback():
    """Merge valid human feedback rows into the main preference dataset atomically."""
    try:
        feedback_rows = []
        if FEEDBACK_FILE.exists():
            with FEEDBACK_FILE.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    feedback_rows.append(json.loads(stripped))

        valid_rows = []
        for row in feedback_rows:
            chosen = row.get("chosen", "")
            if chosen in ("", "__NEEDS_CORRECTION__"):
                continue
            valid_rows.append(
                {
                    "prompt": row.get("prompt", ""),
                    "chosen": chosen,
                    "rejected": row.get("rejected", ""),
                }
            )

        existing_rows = []
        existing_pairs = set()
        if TRAINING_DATA_FILE.exists():
            with TRAINING_DATA_FILE.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    parsed = json.loads(stripped)
                    existing_rows.append(parsed)
                    existing_pairs.add((parsed.get("prompt", ""), parsed.get("chosen", "")))

        new_rows = []
        seen_new_pairs = set()
        for row in valid_rows:
            pair = (row.get("prompt", ""), row.get("chosen", ""))
            if pair in existing_pairs or pair in seen_new_pairs:
                continue
            seen_new_pairs.add(pair)
            new_rows.append(row)

        if not new_rows:
            st.info("No new rows to merge.")
            return

        merged_rows = existing_rows + new_rows
        temp_path = TRAINING_DATA_FILE.with_suffix(TRAINING_DATA_FILE.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            for row in merged_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(TRAINING_DATA_FILE)
        st.success(f"Merged {len(new_rows)} new rows into preference_data_combined.jsonl")
    except FileNotFoundError:
        st.error("Failed to merge feedback into training data.")
    except PermissionError:
        st.error("Failed to merge feedback into training data.")
    except json.JSONDecodeError:
        st.error("Failed to merge feedback into training data.")
    except OSError:
        st.error("Failed to merge feedback into training data.")


def render_hallucination_details(scan):
    """Render the unverified entities and file references from a scan result."""
    unverified_entities = scan.get("unverified_entities", [])
    unverified_files = scan.get("unverified_files", [])
    if unverified_entities:
        st.markdown(
            "**Unverified entities:** "
            + ", ".join(f"`{entity}`" for entity in unverified_entities)
        )
    if unverified_files:
        st.markdown(
            "**Unverified files:** "
            + ", ".join(f"`{filepath}`" for filepath in unverified_files)
        )
    if not unverified_entities and not unverified_files:
        st.markdown("No specific unverified entities were identified.")


def render_feedback_controls(message_index, assistant_message):
    """Render positive and negative feedback controls for an assistant message."""
    if message_index in st.session_state.feedback_given:
        st.caption("✅ Feedback recorded")
        return

    scan = assistant_message.get("hallucination_scan") or {}
    prompt = get_previous_user_prompt(st.session_state.messages, message_index)
    response = assistant_message.get("content", "")

    buttons = st.columns(2)
    thumbs_up = buttons[0].button("👍 Helpful", key=f"thumbs_up_{message_index}")
    thumbs_down = buttons[1].button("👎 Not helpful", key=f"thumbs_down_{message_index}")

    if thumbs_up:
        row = {
            "prompt": prompt,
            "chosen": response,
            "rejected": "",
            "source": "human",
            "rating": "positive",
            "timestamp": iso_timestamp(),
            "hallucination_risk": scan.get("hallucination_risk", "low"),
        }
        if save_feedback_row(row):
            st.session_state.feedback_given.add(message_index)
            st.session_state.correction_pending.pop(message_index, None)
            read_feedback_stats.clear()
            st.rerun()

    if thumbs_down:
        st.session_state.correction_pending[message_index] = True
        st.rerun()

    if not st.session_state.correction_pending.get(message_index):
        return

    correction_key = f"correction_text_{message_index}"
    correction_text = st.text_area(
        "Optional: provide a better answer",
        key=correction_key,
    )
    correction_buttons = st.columns(2)
    submit_feedback = correction_buttons[0].button(
        "Submit feedback",
        key=f"submit_feedback_{message_index}",
    )
    skip_feedback = correction_buttons[1].button(
        "Skip",
        key=f"skip_feedback_{message_index}",
    )

    if submit_feedback:
        cleaned_correction = correction_text.strip()
        correction_provided = bool(cleaned_correction)
        row = {
            "prompt": prompt,
            "chosen": cleaned_correction if correction_provided else "__NEEDS_CORRECTION__",
            "rejected": response,
            "source": "human",
            "rating": "negative",
            "timestamp": iso_timestamp(),
            "correction_provided": correction_provided,
            "hallucination_risk": scan.get("hallucination_risk", "low"),
        }
        if save_feedback_row(row):
            st.session_state.feedback_given.add(message_index)
            st.session_state.correction_pending.pop(message_index, None)
            st.session_state.pop(correction_key, None)
            read_feedback_stats.clear()
            st.rerun()

    if skip_feedback:
        row = {
            "prompt": prompt,
            "chosen": "__NEEDS_CORRECTION__",
            "rejected": response,
            "source": "human",
            "rating": "negative",
            "timestamp": iso_timestamp(),
            "correction_provided": False,
            "hallucination_risk": scan.get("hallucination_risk", "low"),
        }
        if save_feedback_row(row):
            st.session_state.feedback_given.add(message_index)
            st.session_state.correction_pending.pop(message_index, None)
            st.session_state.pop(correction_key, None)
            read_feedback_stats.clear()
            st.rerun()


def render_message(message, index):
    """Render a chat message and any associated hallucination or feedback UI."""
    role = message.get("role", "assistant")
    with st.chat_message(role):
        if role == "user":
            st.markdown(message.get("content", ""))
            return

        content = message.get("content", "")
        scan = message.get("hallucination_scan") or {}
        guardrail = message.get("guardrail_result") or {}
        risk = scan.get("hallucination_risk", "low")

        if guardrail.get("reason") == "toxic_output":
            st.error("This response was blocked by safety guardrails due to toxic output risk.")
            st.markdown(content)
        elif risk == "high":
            st.warning(
                "⚠️ This response contains a high proportion of unverified code references "
                "and may be inaccurate. Please verify against the source files."
            )
            render_hallucination_details(scan)
            with st.expander("Show response anyway (may contain errors)"):
                st.markdown(content)
        else:
            st.markdown(content)
            if risk == "medium":
                with st.expander("ℹ️ Some entities could not be verified"):
                    render_hallucination_details(scan)
            if guardrail.get("unsupported_entities") or guardrail.get("unsupported_files"):
                with st.expander("Safety details"):
                    if guardrail.get("unsupported_entities"):
                        st.markdown("**Unsupported entities:** " + ", ".join(f"`{item}`" for item in guardrail["unsupported_entities"]))
                    if guardrail.get("unsupported_files"):
                        st.markdown("**Unsupported files:** " + ", ".join(f"`{item}`" for item in guardrail["unsupported_files"]))

        render_feedback_controls(index, message)


def render_status_bar():
    """Render the top-of-page status bar for backend readiness and indexing state."""
    agent_ready = st.session_state.agent is not None
    chunk_count = st.session_state.chunk_count
    retriever_mode = st.session_state.retriever_mode
    search_label = "Hybrid search active" if retriever_mode == "hybrid" else "Lexical fallback"
    status_label = "✅ Agent ready" if agent_ready else "⏳ Loading..."

    st.markdown(
        (
            "<div style='padding:0.75rem 1rem; border:1px solid #d0d7de; border-radius:0.75rem; "
            "background:#f6f8fa; margin-bottom:1rem;'>"
            f"{status_label} &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"📦 {chunk_count:,} chunks indexed &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"🔍 {search_label}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_feedback_dashboard():
    """Render the sidebar dashboard for human feedback metrics and merge actions."""
    with st.sidebar:
        st.header("📊 Feedback Dashboard")
        stats = read_feedback_stats()

        if stats.get("error"):
            st.error(stats["error"])

        if stats["total"] == 0:
            st.info("No feedback collected yet.")
        else:
            st.metric("Total feedback", stats["total"])
            st.metric("👍 Helpful", stats["positive"])
            st.metric("👎 Not helpful", stats["negative"])
            st.metric("Corrections provided", stats["corrections"])
            st.metric("Positive %", f"{stats['positive_ratio'] * 100:.1f}%")
            st.progress(stats["positive_ratio"])
            st.markdown(
                "Low risk: `{low}`  \nMedium risk: `{medium}`  \nHigh risk: `{high}`".format(
                    low=stats["risk_breakdown"]["low"],
                    medium=stats["risk_breakdown"]["medium"],
                    high=stats["risk_breakdown"]["high"],
                )
            )

        if st.button("🔁 Merge into training data", use_container_width=True):
            merge_human_feedback()
            read_feedback_stats.clear()


def render_sidebar(repo_map):
    """Render sidebar controls, dashboard content, and repository map details."""
    with st.sidebar:
        st.header("Settings")
        new_repo = st.text_input(
            "Local Repository Path",
            value=st.session_state.target_repo,
            help="The absolute path to the Python project you want to analyze.",
        )

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.feedback_given = set()
            st.session_state.correction_pending = {}
            st.rerun()

        if new_repo != st.session_state.target_repo:
            st.session_state.target_repo = new_repo
            st.cache_resource.clear()
            st.session_state.messages = []
            st.session_state.feedback_given = set()
            st.session_state.correction_pending = {}
            st.session_state.backend = None
            st.session_state.agent = None
            st.session_state.hallucination_detector = None
            st.session_state.retriever_mode = "unknown"
            st.session_state.chunk_count = 0
            st.rerun()

        if not Path(st.session_state.target_repo).is_dir():
            st.warning(f"Directory not found: {st.session_state.target_repo}")
            st.stop()

    render_feedback_dashboard()

    with st.sidebar:
        st.divider()
        st.header("Repository Map")
        with st.expander("View Full Map", expanded=False):
            st.code(repo_map, language="markdown")


def main():
    """Run the Streamlit application."""
    initialize_session_state()

    st.title("Mobtrap Code Assistant")
    st.caption("Code-aware chat with hallucination review and human feedback capture.")

    status_placeholder = st.empty()
    status_placeholder.markdown(
        (
            "<div style='padding:0.75rem 1rem; border:1px solid #d0d7de; border-radius:0.75rem; "
            "background:#f6f8fa; margin-bottom:1rem;'>"
            "⏳ Loading... &nbsp;&nbsp;|&nbsp;&nbsp; 📦 0 chunks indexed &nbsp;&nbsp;|&nbsp;&nbsp; "
            "🔍 Lexical fallback"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    try:
        with st.spinner(f"Analyzing {st.session_state.target_repo} and loading agent..."):
            agent, repo_map, chunks = load_backend(st.session_state.target_repo)
    except Exception as exc:
        st.session_state.backend = None
        st.session_state.agent = None
        st.session_state.hallucination_detector = None
        st.session_state.retriever_mode = "unknown"
        st.session_state.chunk_count = 0
        st.error(f"Failed to load backend: {exc}")
        st.stop()

    st.session_state.backend = (agent, repo_map, chunks)
    st.session_state.agent = agent
    st.session_state.hallucination_detector = HallucinationDetector(chunks)
    st.session_state.retriever_mode = get_retriever_mode(agent)
    st.session_state.chunk_count = len(chunks)

    status_placeholder.empty()
    render_status_bar()
    render_sidebar(repo_map)

    for index, message in enumerate(st.session_state.messages):
        render_message(message, index)

    prompt = st.chat_input("Ask a question about the codebase...")
    if not prompt:
        return

    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        with st.spinner("Thinking..."):
            raw_response = st.session_state.agent.ask(prompt)
        cleaned_response = strip_hallucination_warning(raw_response)
        guardrail = st.session_state.agent.last_guardrail_result
        scan = (
            {
                "hallucination_risk": guardrail.hallucination_risk,
                "hallucination_score": guardrail.hallucination_score,
                "verified_entities": guardrail.verified_entities,
                "unverified_entities": guardrail.unsupported_entities,
                "verified_files": guardrail.verified_files,
                "unverified_files": guardrail.unsupported_files,
                "unsupported_entities": guardrail.unsupported_entities,
                "unsupported_files": guardrail.unsupported_files,
            }
            if guardrail is not None
            else st.session_state.hallucination_detector.scan(cleaned_response)
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": cleaned_response,
                "hallucination_scan": scan,
                "guardrail_result": (
                    {
                        "allowed": guardrail.allowed,
                        "reason": guardrail.reason,
                        "toxicity_score": guardrail.toxicity_score,
                        "hallucination_score": guardrail.hallucination_score,
                        "unsupported_entities": guardrail.unsupported_entities,
                        "unsupported_files": guardrail.unsupported_files,
                    }
                    if guardrail is not None
                    else None
                ),
            }
        )
    except Exception as exc:
        error_message = (
            "**Connection Failed.** Please make sure your local Ollama daemon is running "
            "with `ollama run qwen2.5-coder:3b`.\n\n"
            f"*Error Detail: {exc}*"
        )
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": error_message,
                "hallucination_scan": {
                    "hallucination_risk": "low",
                    "hallucination_score": 0.0,
                    "verified_entities": [],
                    "unverified_entities": [],
                    "verified_files": [],
                    "unverified_files": [],
                },
            }
        )

    st.rerun()


main()
