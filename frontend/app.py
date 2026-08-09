import streamlit as st
import requests
import time
from pathlib import Path

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="ResearchLens",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 ResearchLens")
st.caption("AI-powered commercialisation assessment for research papers")

# ── Sidebar: upload ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Upload Paper")
    uploaded = st.file_uploader("Select a PDF", type=["pdf"])

    if uploaded and st.button("Analyse", type="primary"):
        with st.spinner("Uploading…"):
            resp = requests.post(
                f"{API_BASE}/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
            )
        if resp.ok:
            data = resp.json()
            st.session_state["active_paper_id"] = data["paper_id"]
            st.session_state["active_title"] = data["title"]
            st.success(f"Uploaded: {data['title']} ({data['pages']} pages)")
        else:
            st.error(f"Upload failed: {resp.text}")

# ── Main: status polling then results ────────────────────────────────────────
STAGE_LABELS = {
    "embedding": "Embedding text…",
    "scoring:patentability": "Running patentability agent…",
    "scoring:licensing": "Running licensing agent…",
    "scoring:spinout": "Running spin-out agent…",
    "scoring:risk": "Running risk agent…",
    "done": "Complete",
    "error": "Error",
}

AGENT_COLORS = {
    "patentability": "#4C9BE8",
    "licensing": "#56C271",
    "spinout": "#E8A84C",
    "risk": "#E86B6B",
}

AGENT_LABELS = {
    "patentability": "Patentability",
    "licensing": "Licensing",
    "spinout": "Spin-out",
    "risk": "Risk",
}


def score_bar(label: str, score: float, color: str, inverted: bool = False):
    """Render a labelled score bar."""
    display = 10 - score if inverted else score
    pct = int(display * 10)
    st.markdown(
        f"""
        <div style='margin-bottom:4px'>
            <span style='font-weight:600'>{label}</span>
            <span style='float:right;font-size:1.1em;font-weight:700'>{display:.1f}<span style='font-size:.75em;color:#888'>/10</span></span>
        </div>
        <div style='background:#eee;border-radius:6px;height:10px;margin-bottom:12px'>
            <div style='width:{pct}%;background:{color};height:10px;border-radius:6px'></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def verdict_badge(verdict: str, overall: float):
    if overall >= 7.5:
        bg, fg = "#d4edda", "#155724"
    elif overall >= 5.0:
        bg, fg = "#fff3cd", "#856404"
    else:
        bg, fg = "#f8d7da", "#721c24"
    st.markdown(
        f"<div style='background:{bg};color:{fg};padding:12px 18px;border-radius:8px;"
        f"font-weight:700;font-size:1.05em'>{verdict}</div>",
        unsafe_allow_html=True,
    )


def show_results(paper_id: int):
    resp = requests.get(f"{API_BASE}/results/{paper_id}")
    if not resp.ok:
        st.error("Could not load results.")
        return
    data = resp.json()
    scores = {s["agent"]: s for s in data["scores"]}
    overall = data.get("overall", 0.0)

    col_score, col_detail = st.columns([1, 2])

    with col_score:
        st.subheader("Overall Score")
        st.metric("", f"{overall:.1f} / 10")
        if scores:
            verdict_text = data.get("verdict") or (
                "Strong Commercialisation Potential" if overall >= 7.5
                else "Moderate Potential — Further Assessment Recommended" if overall >= 5.0
                else "Limited Commercialisation Potential"
            )
            verdict_badge(verdict_text, overall)

        st.markdown("---")
        st.subheader("Dimension Scores")
        for agent, color in AGENT_COLORS.items():
            if agent in scores:
                inverted = agent == "risk"
                score_bar(AGENT_LABELS[agent], scores[agent]["score"], color, inverted)

    with col_detail:
        st.subheader("Agent Analysis")
        tab_names = [AGENT_LABELS[a] for a in AGENT_COLORS if a in scores]
        agent_keys = [a for a in AGENT_COLORS if a in scores]
        if tab_names:
            tabs = st.tabs(tab_names)
            for tab, key in zip(tabs, agent_keys):
                s = scores[key]
                with tab:
                    display_score = (10 - s["score"]) if key == "risk" else s["score"]
                    st.metric("Score", f"{display_score:.1f} / 10")
                    st.markdown("**Rationale**")
                    st.write(s["rationale"])
                    if s.get("key_points"):
                        st.markdown("**Key Points**")
                        for pt in s["key_points"]:
                            st.markdown(f"- {pt}")

    # ── Q&A section ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Ask the Paper")
    st.caption("Ask any question and the AI will answer using the paper's text.")

    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = {}
    history = st.session_state["qa_history"].setdefault(paper_id, [])

    for qa in history:
        with st.chat_message("user"):
            st.write(qa["q"])
        with st.chat_message("assistant"):
            st.write(qa["a"])

    question = st.chat_input("Ask a question about this paper…", key=f"chat_{paper_id}")
    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                qa_resp = requests.post(
                    f"{API_BASE}/query/{paper_id}",
                    json={"question": question},
                )
            if qa_resp.ok:
                answer = qa_resp.json()["answer"]
                st.write(answer)
                history.append({"q": question, "a": answer})
            else:
                st.error(f"Could not get an answer: {qa_resp.text}")


# ── Active paper: poll until done ────────────────────────────────────────────
if "active_paper_id" in st.session_state:
    paper_id = st.session_state["active_paper_id"]
    title = st.session_state.get("active_title", f"Paper #{paper_id}")

    st.subheader(f"📄 {title}")

    status_resp = requests.get(f"{API_BASE}/status/{paper_id}")
    if status_resp.ok:
        status = status_resp.json()
        stage = status.get("stage", "unknown")

        if not status.get("done") and stage != "error":
            label = STAGE_LABELS.get(stage, stage)
            with st.status(label, expanded=True) as s_widget:
                for step_stage, step_label in STAGE_LABELS.items():
                    if step_stage == "done":
                        break
                    st.write(step_label)
                    if step_stage == stage:
                        break
            time.sleep(3)
            st.rerun()

        elif stage == "error":
            st.error(f"Processing failed: {status.get('error')}")

        else:
            show_results(paper_id)

# ── Paper library ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Paper Library")

hist_resp = requests.get(f"{API_BASE}/results")
if not hist_resp.ok:
    st.warning("Could not reach the API. Make sure the backend is running.")
else:
    rows = hist_resp.json()
    if not rows:
        st.info("No papers analysed yet. Upload a PDF to get started.")
    else:
        # ── Search & sort controls ────────────────────────────────────────────
        ctrl_left, ctrl_right = st.columns([3, 1])
        with ctrl_left:
            search = st.text_input("Search by title", placeholder="Type to filter…", label_visibility="collapsed")
        with ctrl_right:
            sort_by = st.selectbox("Sort by", ["Newest first", "Oldest first", "Score ↑", "Score ↓"], label_visibility="collapsed")

        # Filter
        if search:
            rows = [r for r in rows if search.lower() in (r.get("title") or r["filename"]).lower()]

        # Sort
        if sort_by == "Oldest first":
            rows = sorted(rows, key=lambda r: r.get("uploaded_at") or "")
        elif sort_by == "Score ↑":
            rows = sorted(rows, key=lambda r: r.get("overall") or 0.0)
        elif sort_by == "Score ↓":
            rows = sorted(rows, key=lambda r: r.get("overall") or 0.0, reverse=True)

        if not rows:
            st.info("No papers match your search.")
        else:
            st.caption(f"{len(rows)} paper{'s' if len(rows) != 1 else ''}")

        if "delete_confirm" not in st.session_state:
            st.session_state["delete_confirm"] = None

        for row in rows:
            paper_id  = row["id"]
            title     = row.get("title") or row["filename"]
            date      = (row.get("uploaded_at") or "")[:10]
            overall   = row.get("overall")
            verdict   = row.get("verdict", "")

            # Score badge colour
            if overall is None:
                badge_bg, badge_fg = "#e9ecef", "#495057"
                badge_txt = "Pending"
            elif overall >= 7.5:
                badge_bg, badge_fg = "#d4edda", "#155724"
                badge_txt = f"{overall:.1f}/10"
            elif overall >= 5.0:
                badge_bg, badge_fg = "#fff3cd", "#856404"
                badge_txt = f"{overall:.1f}/10"
            else:
                badge_bg, badge_fg = "#f8d7da", "#721c24"
                badge_txt = f"{overall:.1f}/10"

            row_left, row_mid, row_right = st.columns([5, 2, 2])

            with row_left:
                st.markdown(
                    f"<div style='padding:6px 0'>"
                    f"<span style='font-weight:600;font-size:1em'>{title}</span>"
                    f"<span style='color:#888;font-size:.85em;margin-left:10px'>{date}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with row_mid:
                st.markdown(
                    f"<div style='padding:8px 0'>"
                    f"<span style='background:{badge_bg};color:{badge_fg};padding:3px 10px;"
                    f"border-radius:12px;font-weight:700;font-size:.9em'>{badge_txt}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with row_right:
                btn_left, btn_right = st.columns(2)
                with btn_left:
                    if st.button("Load", key=f"load_{paper_id}", use_container_width=True):
                        st.session_state["active_paper_id"] = paper_id
                        st.session_state["active_title"] = title
                        st.session_state["delete_confirm"] = None
                        st.rerun()
                with btn_right:
                    if st.session_state["delete_confirm"] == paper_id:
                        if st.button("Confirm", key=f"confirm_{paper_id}", type="primary", use_container_width=True):
                            del_resp = requests.delete(f"{API_BASE}/papers/{paper_id}")
                            if del_resp.ok:
                                if st.session_state.get("active_paper_id") == paper_id:
                                    st.session_state.pop("active_paper_id", None)
                                    st.session_state.pop("active_title", None)
                                st.session_state["delete_confirm"] = None
                                st.rerun()
                            else:
                                st.error("Delete failed.")
                    else:
                        if st.button("Delete", key=f"delete_{paper_id}", use_container_width=True):
                            st.session_state["delete_confirm"] = paper_id
                            st.rerun()

            st.divider()
