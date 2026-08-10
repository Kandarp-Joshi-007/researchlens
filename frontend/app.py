import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

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
    st.header("Upload Papers")
    uploaded_files = st.file_uploader(
        "Select one or more PDFs", type=["pdf"], accept_multiple_files=True
    )
    deep = st.checkbox(
        "Deep analysis",
        help="Score each dimension three times and report the spread. "
             "Roughly three times slower.",
    )

    if uploaded_files and st.button("Analyse", type="primary"):
        queued, duplicates, failed = [], [], []
        progress = st.progress(0.0)
        for index, item in enumerate(uploaded_files, start=1):
            try:
                resp = requests.post(
                    f"{API_BASE}/upload",
                    files={"file": (item.name, item.getvalue(), "application/pdf")},
                    params={"deep": str(deep).lower()},
                    timeout=300,
                )
            except requests.RequestException as exc:
                failed.append(f"{item.name}: {exc}")
                continue
            finally:
                progress.progress(index / len(uploaded_files))

            if resp.ok:
                data = resp.json()
                (duplicates if data.get("duplicate") else queued).append(data)
                st.session_state["active_paper_id"] = data["paper_id"]
                st.session_state["active_title"] = data["title"]
            else:
                detail = resp.json().get("detail", resp.text) if resp.content else resp.text
                failed.append(f"{item.name}: {detail}")

        progress.empty()
        if queued:
            st.success(f"Queued {len(queued)} paper(s) for analysis.")
        if duplicates:
            st.info(f"{len(duplicates)} already analysed — showing saved results.")
        for message in failed:
            st.error(message)

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

# Categorical hues in fixed order, one per dimension — never cycled or
# reassigned by rank. Validated for colourblind separation against the chart
# surface; the previous green/amber pair was indistinguishable under
# protanopia (ΔE 4.1). Numeric labels sit beside every bar, which is the
# required relief for the sub-3:1 contrast of the lighter hues.
AGENT_COLORS = {
    "patentability": "#2a78d6",  # blue
    "licensing": "#eb6834",      # orange
    "spinout": "#1baf7a",        # aqua
    "risk": "#eda100",           # yellow
}

# Reserved status colours for verdict bands — never used for a dimension.
VERDICT_STYLES = {
    "strong": ("#e6f4e6", "#0b5d0b", "#0ca30c"),
    "moderate": ("#fdf3dd", "#6b4c00", "#fab219"),
    "limited": ("#fae7e7", "#8c1f1f", "#d03b3b"),
}


def verdict_band(overall):
    if overall is None:
        return None
    if overall >= 7.5:
        return "strong"
    return "moderate" if overall >= 5.0 else "limited"

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

    report = requests.get(f"{API_BASE}/papers/{paper_id}/report")
    if report.ok:
        payload = report.json()
        st.download_button(
            "Download commercialisation brief",
            data=payload["markdown"],
            file_name=payload["filename"],
            mime="text/markdown",
        )

    col_score, col_detail = st.columns([1, 2])

    with col_score:
        st.subheader("Overall Score")
        st.metric("Overall", f"{overall:.1f} / 10", label_visibility="collapsed")
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

                    if (s.get("samples") or 1) > 1:
                        low, high = s.get("score_min"), s.get("score_max")
                        if key == "risk":
                            low, high = 10 - high, 10 - low
                        spread = (high or 0) - (low or 0)
                        agreement = ("high" if spread <= 1 else
                                     "moderate" if spread <= 3 else "low")
                        st.caption(
                            f"{s['samples']} samples · range {low:.1f}–{high:.1f} "
                            f"· {agreement} agreement"
                        )
                    st.markdown("**Rationale**")
                    st.write(s["rationale"])
                    if s.get("key_points"):
                        st.markdown("**Key Points**")
                        for pt in s["key_points"]:
                            st.markdown(f"- {pt}")

                    evidence = s.get("evidence") or []
                    if evidence:
                        with st.expander(f"Evidence from the paper ({len(evidence)})"):
                            for quote in evidence:
                                st.markdown(
                                    f"<div style='border-left:3px solid {AGENT_COLORS[key]};"
                                    f"padding:6px 0 6px 12px;margin-bottom:10px;color:inherit;"
                                    f"font-style:italic;opacity:.85'>“{quote}”</div>",
                                    unsafe_allow_html=True,
                                )
                    else:
                        st.caption("No evidence quotes recorded for this score.")

    # ── Prior art ─────────────────────────────────────────────────────────────
    works = data.get("prior_art") or []
    if works:
        st.markdown("---")
        st.subheader("Prior Art")
        st.caption("Similar published work from OpenAlex, used to ground the "
                   "patentability assessment.")
        for work in works:
            year = work.get("year") or "n.d."
            venue = work.get("venue") or "—"
            doi = work.get("doi")
            title = f"[{work['title']}]({doi})" if doi else work["title"]
            st.markdown(
                f"- {title}  \n"
                f"  <span style='color:#888;font-size:.85em'>{work.get('authors') or 'Unknown'} · "
                f"{year} · {venue} · cited {work.get('citations', 0)}×</span>",
                unsafe_allow_html=True,
            )

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

# ── Portfolio ─────────────────────────────────────────────────────────────────
def render_portfolio(rows):
    """Headline numbers, distribution and comparison across all scored papers."""
    scored = [r for r in rows if r.get("overall") is not None]

    st.markdown("---")
    st.subheader("Portfolio")

    if not scored:
        st.info("No scored papers yet.")
        return

    values = [r["overall"] for r in scored]
    bands = {"strong": 0, "moderate": 0, "limited": 0}
    for value in values:
        bands[verdict_band(value)] += 1

    # Headline numbers: a stat tile reads better than a chart for a few papers.
    tiles = st.columns(4)
    tiles[0].metric("Papers scored", len(scored))
    tiles[1].metric("Mean score", f"{sum(values) / len(values):.1f}")
    tiles[2].metric("Strong", bands["strong"])
    tiles[3].metric("Limited", bands["limited"])

    # A distribution only earns its space once there are enough papers to
    # have a shape; below that the leaderboard already shows every value.
    if len(scored) >= 8:
        st.caption("Score distribution")
        buckets = {}
        for value in values:
            key = min(int(value), 9)
            buckets[key] = buckets.get(key, 0) + 1
        table = pd.DataFrame(
            {"papers": [buckets.get(b, 0) for b in range(11)]},
            index=[f"{b}–{b + 1}" for b in range(11)],
        )
        st.bar_chart(table, height=200, color="#2a78d6")

    # ── Comparison ────────────────────────────────────────────────────────────
    if len(scored) >= 2:
        st.markdown("**Compare two papers**")
        labels = {f"{r.get('title') or r['filename']} ({r['overall']}/10)": r["id"]
                  for r in scored}
        names = list(labels)
        left, right = st.columns(2)
        first = left.selectbox("Paper A", names, index=0, key="cmp_a")
        second = right.selectbox("Paper B", names, index=min(1, len(names) - 1),
                                 key="cmp_b")

        if labels[first] != labels[second]:
            detail_a = requests.get(f"{API_BASE}/results/{labels[first]}")
            detail_b = requests.get(f"{API_BASE}/results/{labels[second]}")
            if detail_a.ok and detail_b.ok:
                render_comparison(detail_a.json(), detail_b.json())
        else:
            st.caption("Pick two different papers to compare.")


def render_comparison(a, b):
    """Dimension-by-dimension comparison, direct-labelled rather than colour-coded."""
    scores_a = {s["agent"]: s["score"] for s in a["scores"]}
    scores_b = {s["agent"]: s["score"] for s in b["scores"]}

    header = st.columns([2, 1, 1])
    header[0].markdown("**Dimension**")
    header[1].markdown("**Paper A**")
    header[2].markdown("**Paper B**")

    for agent, label in AGENT_LABELS.items():
        raw_a, raw_b = scores_a.get(agent), scores_b.get(agent)
        if raw_a is None or raw_b is None:
            continue
        shown_a = 10 - raw_a if agent == "risk" else raw_a
        shown_b = 10 - raw_b if agent == "risk" else raw_b
        row = st.columns([2, 1, 1])
        row[0].markdown(
            f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;"
            f"background:{AGENT_COLORS[agent]};margin-right:8px'></span>{label}",
            unsafe_allow_html=True,
        )
        row[1].markdown(f"**{shown_a:.1f}**")
        delta = shown_b - shown_a
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
        row[2].markdown(f"**{shown_b:.1f}**  <span style='color:#898781;font-size:.85em'>"
                        f"{arrow} {abs(delta):.1f}</span>", unsafe_allow_html=True)

    totals = st.columns([2, 1, 1])
    totals[0].markdown("**Overall**")
    totals[1].markdown(f"**{a.get('overall')}**")
    totals[2].markdown(f"**{b.get('overall')}**")


# ── Paper library ─────────────────────────────────────────────────────────────
hist_resp = requests.get(f"{API_BASE}/results")
if not hist_resp.ok:
    st.warning("Could not reach the API. Make sure the backend is running.")
else:
    rows = hist_resp.json()
    render_portfolio(rows)

    st.markdown("---")
    st.subheader("Paper Library")

    if not rows:
        st.info("No papers analysed yet. Upload a PDF to get started.")
    else:
        table = pd.DataFrame([{
            "id": r["id"],
            "title": r.get("title") or r["filename"],
            "uploaded": (r.get("uploaded_at") or "")[:10],
            "overall": r.get("overall"),
            "verdict": r.get("verdict"),
        } for r in rows])
        st.download_button(
            "Download library as CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name="researchlens-portfolio.csv",
            mime="text/csv",
        )
        with st.expander("Table view"):
            st.dataframe(table, use_container_width=True, hide_index=True)
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

            # Score badge colour, from the reserved verdict palette
            band = verdict_band(overall)
            if band is None:
                badge_bg, badge_fg = "#eeeeec", "#52514e"
                badge_txt = "Pending"
            else:
                badge_bg, badge_fg, _ = VERDICT_STYLES[band]
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
