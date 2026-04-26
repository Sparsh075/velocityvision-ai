from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import requests
import streamlit as st

API_BASE = os.getenv("VELOCITYVISION_API", "http://127.0.0.1:8000")

st.set_page_config(page_title="VelocityVision", page_icon="VV", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --bg0: #050816;
            --bg1: #0b132b;
            --card: rgba(255, 255, 255, 0.08);
            --card2: rgba(15, 23, 42, 0.70);
            --stroke: rgba(148, 163, 184, 0.25);
            --text: #e8eefc;
            --muted: #93a8d6;
            --cy: #22d3ee;
            --or: #fb923c;
            --li: #94a3b8;
        }
        .stApp {
            background:
                radial-gradient(1200px 600px at -10% -20%, #162448 0%, transparent 70%),
                radial-gradient(1200px 600px at 110% -20%, #073042 0%, transparent 70%),
                linear-gradient(130deg, var(--bg0), var(--bg1));
            color: var(--text);
        }
        .glass {
            background: var(--card);
            border: 1px solid var(--stroke);
            border-radius: 18px;
            backdrop-filter: blur(12px);
            box-shadow: 0 18px 40px rgba(3, 7, 18, 0.45);
            padding: 18px;
        }
        .hero {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 2px;
            letter-spacing: 0.2px;
        }
        .sub {
            color: var(--muted);
            margin-bottom: 16px;
        }
        .stat {
            background: var(--card2);
            border: 1px solid var(--stroke);
            border-radius: 14px;
            padding: 14px;
            animation: floatUp 0.6s ease;
        }
        .stat h4 {
            margin: 0;
            color: var(--muted);
            font-size: 0.92rem;
            font-weight: 500;
        }
        .stat h2 {
            margin: 7px 0 0;
            color: var(--text);
            font-size: 1.45rem;
        }
        @keyframes floatUp {
            0% { transform: translateY(6px); opacity: 0.2; }
            100% { transform: translateY(0); opacity: 1; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def api_get(path: str, timeout: int = 45) -> requests.Response:
    response = requests.get(f"{API_BASE}{path}", timeout=timeout)
    return response


def post_video(uploaded_file, detect_stride: int) -> str:
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "video/mp4")}
    response = requests.post(
        f"{API_BASE}/api/process",
        params={"detect_stride": detect_stride},
        files=files,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["job_id"]


def fetch_status(job_id: str) -> Dict[str, Any]:
    response = api_get(f"/api/jobs/{job_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_artifact(job_id: str, artifact: str, timeout: int = 90) -> Optional[bytes]:
    response = api_get(f"/api/jobs/{job_id}/{artifact}", timeout=timeout)
    if response.status_code != 200:
        return None
    return response.content


if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "status_cache" not in st.session_state:
    st.session_state.status_cache = {}

with st.sidebar:
    st.markdown("### VelocityVision")
    st.caption("AI Sports Intelligence Platform")
    page = st.radio("Navigate", ["Upload", "Live Analysis", "Insights Dashboard"], label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"API: {API_BASE}")

st.markdown('<div class="hero">VelocityVision Analyst Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Team intelligence, homography-aware speed, possession analytics, and advanced pass networks.</div>', unsafe_allow_html=True)

if page == "Upload":
    c1, c2 = st.columns([1.2, 1], gap="large")

    with c1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Upload Match Clip")
        uploaded_file = st.file_uploader("Short football video (15-30 sec)", type=["mp4", "mov", "avi", "mkv"])
        detect_stride = st.slider("CPU Frame Sampling", min_value=1, max_value=4, value=1)

        submit = st.button("Start Analysis", type="primary", use_container_width=True, disabled=uploaded_file is None)
        if submit and uploaded_file is not None:
            try:
                with st.spinner("Submitting clip to backend..."):
                    job_id = post_video(uploaded_file, detect_stride)
                st.session_state.job_id = job_id
                st.success(f"Analysis job created: {job_id}")
            except Exception as exc:
                st.error(f"Submission failed: {exc}")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.subheader("Pipeline Features")
        st.write("- YOLOv8 detection and SORT tracking")
        st.write("- K-Means jersey clustering (Team A / Team B)")
        st.write("- Homography-based speed estimation")
        st.write("- Possession timeline and team percentages")
        st.write("- Heatmap and passing network exports")
        st.write("- Player ranking CSV")
        st.markdown('</div>', unsafe_allow_html=True)

if page == "Live Analysis":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Live Analysis Status")

    job_id = st.session_state.job_id
    if not job_id:
        st.info("Start a job from Upload page to view live progress.")
    else:
        try:
            status = fetch_status(job_id)
            st.session_state.status_cache = status
            st.caption(f"Job ID: {job_id}")

            p = float(status.get("progress", 0.0))
            st.progress(p)
            st.write(status.get("message", ""))

            cols = st.columns(4)
            metrics = status.get("metrics") or {}
            possession = metrics.get("possession_pct", {})
            cols[0].markdown(f'<div class="stat"><h4>Avg Speed</h4><h2>{metrics.get("avg_speed_kmh", 0):.2f} km/h</h2></div>', unsafe_allow_html=True)
            cols[1].markdown(f'<div class="stat"><h4>Max Speed</h4><h2>{metrics.get("max_speed_kmh", 0):.2f} km/h</h2></div>', unsafe_allow_html=True)
            cols[2].markdown(f'<div class="stat"><h4>Total Passes</h4><h2>{metrics.get("pass_count", 0)}</h2></div>', unsafe_allow_html=True)
            cols[3].markdown(f'<div class="stat"><h4>Possession A/B</h4><h2>{possession.get("Team A", 0):.1f}% / {possession.get("Team B", 0):.1f}%</h2></div>', unsafe_allow_html=True)

            if status.get("status") == "failed":
                st.error(status.get("error", "Processing failed."))

            if status.get("status") == "completed":
                video_bytes = fetch_artifact(job_id, "download", timeout=140)
                if video_bytes:
                    st.video(video_bytes)
                    st.download_button(
                        "Download Annotated Video",
                        data=video_bytes,
                        file_name=f"velocityvision_{job_id}.mp4",
                        mime="video/mp4",
                        use_container_width=True,
                    )
            else:
                time.sleep(2)
                st.rerun()

        except Exception as exc:
            st.error(f"Status refresh failed: {exc}")

    st.markdown('</div>', unsafe_allow_html=True)

if page == "Insights Dashboard":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("Insights Dashboard")

    job_id = st.session_state.job_id
    if not job_id:
        st.info("Run an analysis first to unlock insights.")
    else:
        status = st.session_state.status_cache
        if not status:
            try:
                status = fetch_status(job_id)
            except Exception:
                status = {}

        if status.get("status") != "completed":
            st.warning("Analysis is not completed yet.")
        else:
            metrics = status.get("metrics") or {}
            team_stats = metrics.get("team_speed_stats", {})

            t1, t2 = st.columns(2)
            ta = team_stats.get("Team A", {})
            tb = team_stats.get("Team B", {})
            t1.markdown(f'<div class="stat"><h4>Team A Avg / Max</h4><h2>{ta.get("avg_speed_kmh",0):.2f} / {ta.get("max_speed_kmh",0):.2f} km/h</h2></div>', unsafe_allow_html=True)
            t2.markdown(f'<div class="stat"><h4>Team B Avg / Max</h4><h2>{tb.get("avg_speed_kmh",0):.2f} / {tb.get("max_speed_kmh",0):.2f} km/h</h2></div>', unsafe_allow_html=True)

            a1, a2, a3 = st.columns(3)
            with a1:
                heatmap = fetch_artifact(job_id, "heatmap")
                if heatmap:
                    st.image(heatmap, caption="Movement Heatmap")
            with a2:
                pass_graph = fetch_artifact(job_id, "pass-network")
                if pass_graph:
                    st.image(pass_graph, caption="Advanced Passing Network")
            with a3:
                possession = fetch_artifact(job_id, "possession-timeline")
                if possession:
                    st.image(possession, caption="Possession Timeline")

            rankings = metrics.get("player_rankings", [])
            if rankings:
                st.markdown("#### Player Rankings")
                st.dataframe(rankings, use_container_width=True)

            csv_data = fetch_artifact(job_id, "player-csv")
            if csv_data:
                st.download_button(
                    "Export Player Analytics CSV",
                    data=csv_data,
                    file_name=f"velocityvision_players_{job_id}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    st.markdown('</div>', unsafe_allow_html=True)
