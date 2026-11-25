"""AeroPredict AI | Obsidian Flight Deck (Milestone 7).

Preserves all previous tabs, adds the official test-set evaluation review.
No NASA test labels are used for model tuning, training, or selection.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_fd001
from src.preprocessing import audit_dataset, build_report, training_life_stage
from src.feature_engineering import add_causal_features, training_rul_labels, feature_columns
from src.engine_monitoring import MONITORED_SENSORS, health_band


# -------------------------------------------------------------
# Page configuration and visual system
# -------------------------------------------------------------
st.set_page_config(
    page_title="AeroPredict AI | Obsidian Flight Deck",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

INK = "#101018"
PANEL = "#1C1E2B"
PANEL_ALT = "#171925"
OUTLINE = "#343344"
TEXT = "#F5F1ED"
MUTED = "#ABB0C3"
ORANGE = "#FF8658"
LILAC = "#B3A3FF"
SILVER = "#D0D4E2"
GRID = "#313342"

st.markdown(
    '<style>\n/* AeroPredict / Obsidian 2.0 — visual hierarchy & navigation overhaul */\n:root { color-scheme: dark; }\nhtml, body, [data-testid="stAppViewContainer"] {\n  font-family: Inter, \'Segoe UI Variable\', \'Segoe UI\', system-ui, sans-serif;\n}\n.stApp {\n  color:#F5F1ED;\n  background:\n    radial-gradient(ellipse 75% 42% at 87% -25%,rgba(179,163,255,.095),transparent 80%),\n    #101018;\n}\n[data-testid="stHeader"] {background:rgba(16,16,24,.95);}\n[data-testid="stMainBlockContainer"], .block-container {\n  max-width:1450px;\n  padding:2.2rem 3.2rem 4.6rem;\n}\n/* Sidebar replaces the nine squeezed, horizontally scrolling tabs. */\n[data-testid="stSidebar"] { background:#161620; border-right:1px solid #30303E; }\n[data-testid="stSidebar"] > div { background:#161620; }\n[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding:1.4rem .6rem 1.6rem; }\n.ap-side-brand {padding:17px 10px 8px;display:flex;align-items:center;gap:12px;}\n.ap-side-mark {background:#FF8658;color:#1A1520;width:39px;height:39px;border-radius:12px;\n  display:grid;place-items:center;font-size:23px;font-weight:900;box-shadow:0 5px 22px #ff865826;}\n.ap-side-logo {font-weight:820;color:#F7F3F1;letter-spacing:-.035em;font-size:1.13rem;}\n.ap-side-logo em {font-style:normal;color:#FF8658;}\n.ap-side-caption {font-size:.64rem;letter-spacing:.13em;color:#9FA0B5;font-weight:780;margin:2px 0 0;}\n.ap-nav-label {margin:22px 10px 6px;font-size:.63rem;color:#8B8DA6;letter-spacing:.17em;font-weight:840;}\n[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] {gap:3px;display:flex;flex-direction:column;}\n[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label {\n  padding:11px 12px;border-radius:12px;margin:0;color:#C4C4D4;background:transparent;\n  border:1px solid transparent;cursor:pointer;transition:background .16s,border-color .16s;\n}\n[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:hover {\n  background:#272533;border-color:#454154;\n}\n[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {\n  background:linear-gradient(90deg,#352A39,#282433);border-color:#6F4F64;\n}\n[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label p {\n  color:#D5D3E2;font-size:.84rem;font-weight:680;\n}\n[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {color:#FFB697;}\n[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] {flex:1;}\n[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label div[role="radio"] {\n  height:11px;width:11px;border-color:#716D82;flex-shrink:0;\n}\n.ap-side-foot {margin:19px 10px 0;padding:16px;border:1px solid #353243;\n border-radius:15px;background:#201E2C;font-size:.76rem;color:#B5B2C6;line-height:1.7;}\n.ap-side-foot b {color:#FFC5AB;}\n/* Typography, page header and airy editorial layout. */\nh1,h2,h3,h4 {font-weight:760;letter-spacing:-.035em;color:#FBF8F5 !important;}\np,li,.stCaption {color:#B5B3C5;}\na {color:#FFB193 !important;}\n.ap-workspace-head {display:flex;align-items:center;justify-content:space-between;gap:20px;\n  margin:6px 0 22px;padding-bottom:24px;border-bottom:1px solid #302F3F;}\n.ap-page-eyebrow {font-size:.69rem;font-weight:800;letter-spacing:.15em;color:#FF9E7B;margin-bottom:11px;}\n.ap-page-title {font-size:clamp(1.9rem,3.6vw,3rem);font-weight:800;line-height:1.08;\n  letter-spacing:-.058em;color:#FBF8F5;margin-bottom:9px;}\n.ap-page-subtitle {color:#ACADBF;font-size:.91rem;line-height:1.65;max-width:790px;}\n.ap-page-badge {border:1px solid #4B455B;border-radius:999px;padding:9px 14px;\n  font-size:.7rem;color:#E0DAF0;font-weight:750;white-space:nowrap;background:#242233;}\n.ap-page-badge i{display:inline-block;height:7px;width:7px;border-radius:50%;background:#FF8658;\n margin-right:7px;box-shadow:0 0 10px #FF865880;}\n.ap-overview-hero {position:relative;overflow:hidden;min-height:245px;\n  border:1px solid #4E3A4F;border-radius:24px;padding:38px 42px;margin-bottom:22px;\n  background:linear-gradient(108deg,#25202F 0%,#221D31 55%,#34253D 100%);}\n.ap-overview-hero::after {content:\'\';position:absolute;right:-80px;top:-145px;width:480px;height:480px;\n  border:1px solid #ffffff24;border-radius:50%;box-shadow:0 0 0 54px #ffffff09,0 0 0 114px #ffffff07;\n  pointer-events:none;}\n.ap-overview-kicker {font-size:.7rem;letter-spacing:.2em;font-weight:830;color:#FFB38F;}\n.ap-overview-title {position:relative;z-index:1;max-width:760px;margin:13px 0 13px;font-size:clamp(2rem,4vw,3.75rem);\n  font-weight:810;letter-spacing:-.063em;line-height:1.05;color:#FFF9F4;}\n.ap-overview-title span {color:#FF986F;}\n.ap-overview-desc {position:relative;z-index:1;max-width:690px;line-height:1.68;color:#CEC9DC;font-size:.96rem;}\n/* Sleek metrics: high contrast, clean numbers, visually quiet edges. */\n.ap-metric {background:linear-gradient(135deg,#20202C,#1A1B26);border:1px solid #333343;\n  border-radius:17px;min-height:105px;padding:19px 20px;margin:2px 0 17px;}\n.ap-metric-label {color:#A6A4BA;font-size:.68rem;letter-spacing:.035em;font-weight:680;margin-bottom:13px;\n  text-transform:none;}\n.ap-metric-value {font-size:clamp(1.45rem,2.2vw,2.15rem);font-weight:785;\n  letter-spacing:-.055em;color:#FAF6F2;line-height:1.05;}\n.ap-metric-footer {font-size:.65rem;letter-spacing:.04em;color:#BCADFF;margin-top:10px;}\n.ap-section-head {display:flex;align-items:center;gap:11px;margin:34px 0 13px;}\n.ap-section-head:before {content:\'\';height:24px;width:3px;background:#FF8658;border-radius:4px;flex:none;}\n.ap-section-no {display:none;}\n.ap-section-title {font-size:1.28rem;font-weight:750;color:#F4F0F3;letter-spacing:-.038em;}\n.ap-hint {font-size:.83rem;color:#A9A9BE;line-height:1.6;margin-bottom:16px;}\n.ap-journey-card {min-height:140px;border:1px solid #383445;border-radius:18px;\n  background:linear-gradient(130deg,#211E2C,#1A1B27);padding:20px 20px 10px;margin:7px 0 4px;}\n.ap-journey-icon {font-size:1.2rem;color:#FF9D75;margin-bottom:13px;}\n.ap-journey-title {font-weight:760;font-size:1rem;color:#F5F1ED;letter-spacing:-.025em;margin-bottom:7px;}\n.ap-journey-desc {font-size:.77rem;line-height:1.52;color:#A8A9C0;}\n.ap-storyline {border:1px solid #3D3750;border-radius:16px;background:#232032;padding:18px 23px;\n  margin:15px 0;color:#D9D5E5;line-height:1.65;font-size:.85rem;}\n.ap-storyline strong {color:#FFB395;}\n/* Streamlit controls and tables stay practical and keyboard-accessible. */\ndiv[data-testid="stWidgetLabel"] p, label {color:#DBD8E8 !important;font-weight:680;}\ndiv[data-testid="stSelectbox"] div[data-baseweb="select"]>div,\ndiv[data-testid="stMultiSelect"] div[data-baseweb="select"]>div {\n  background:#252534 !important;color:#F5F1ED !important;\n  border:1px solid #54516B !important;border-radius:12px !important;min-height:48px!important;\n}\ndiv[data-baseweb="select"] span,div[data-baseweb="select"] input,\ndiv[data-testid="stSelectbox"] * {color:#F5F1ED!important;}\n[data-baseweb="popover"],[role="listbox"] {background:#252535!important;color:#F5F1ED!important;}\n[role="option"],[role="option"] * {color:#F5F1ED!important;}\n[role="option"]:hover {background:#40364D!important;}\ndiv[data-testid="stTextInput"] input,div[data-testid="stNumberInput"] input {background:#252534!important;\n  border:1px solid #54516B!important;color:#F5F1ED!important;border-radius:10px!important;}\ndiv[data-testid="stSlider"] [role="slider"] {background:#FF8658!important;}\n[data-testid="stDataFrame"] {border:1px solid #373647;border-radius:14px;overflow:hidden;}\ndiv[data-testid="stExpander"] {border:1px solid #414053;border-radius:12px;}\n/* Data Laboratory uses the only remaining tabs; make them look like an app\'s sub-navigation. */\ndiv[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:7px;background:#1B1B29;padding:5px;border-radius:14px;\n  border:1px solid #313042;overflow-x:auto;}\ndiv[data-testid="stTabs"] button[data-baseweb="tab"] {border-radius:10px;padding:8px 15px;color:#B9B4CF!important;\n  font-weight:700!important;background:transparent;}\ndiv[data-testid="stTabs"] button[data-baseweb="tab"] p {color:inherit!important;}\ndiv[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {\n  color:#FFAE8B!important;background:#3A2B38;}\ndiv[data-testid="stTabs"] [data-baseweb="tab-highlight"],div[data-testid="stTabs"] [data-baseweb="tab-border"] {display:none;}\n.stButton button {border-radius:11px!important;border:1px solid #544B5E!important;\n  background:#272333!important;color:#F6F1F8!important;font-weight:700!important;\n  min-height:41px;}\n.stButton button:hover {border-color:#FF8658!important;color:#FFAE8B!important;background:#31283A!important;}\ndiv[data-testid="stDownloadButton"] button {background:#FF8658!important;color:#1F1820!important;\n  border:0!important;border-radius:12px!important;font-weight:780!important;padding:.54rem 1rem;}\ndiv[data-testid="stRadio"] label p,div[data-testid="stCheckbox"] label p,\ndiv[data-testid="stToggle"] label p {color:#E6E0EE!important;}\n/* Old milestone banners become concise, editorial insights instead of large blocks. */\n.ap-spotlight,.ap-m3-banner,.ap-model-callout,.ap-m6-banner {\n  background:#231F2D;border:1px solid #4D4053;border-left:3px solid #FF8658;\n  border-radius:13px;padding:15px 19px;margin:10px 0 20px;\n  color:#D7D0E1;line-height:1.65;font-size:.84rem;}\n.ap-spotlight b,.ap-m3-banner b,.ap-model-callout b,.ap-m6-banner b,.ap-quality-note b {color:#FFB397;}\n.ap-m6-kicker,.ap-status-badge {color:#BEB2E4;font-size:.66rem;letter-spacing:.10em;font-weight:760;\n background:none;border:0;padding:0;}\n.ap-m6-title {font-weight:770;font-size:1.15rem;color:#F5F0F4;letter-spacing:-.02em;margin:6px 0;}\n.ap-m6-description {color:#C5BFD3;font-size:.83rem;line-height:1.7;max-width:1030px;}\n.ap-quality-note,.ap-note {background:#231F2D;border-left:3px solid #B3A3FF;padding:12px 16px;\n  border-radius:0 12px 12px 0;color:#CEC9DF;line-height:1.65;font-size:.8rem;margin:14px 0;}\n.ap-m6-status {border-radius:13px;padding:13px 17px;border:1px solid #655377;background:#252339;\n  margin:8px 0 16px;color:#E5DFF1;font-weight:730;}\n.ap-m6-status b {color:#FFAC8B;}\n.ap-m6-status.urgent {border-color:#A35F61;background:#38252A;}\n.ap-m6-status.watch {border-color:#B97955;background:#3A2929;}\n.ap-m6-status.routine {border-color:#7163A2;background:#27243A;}\n.ap-side {background:#21212E;border:1px solid #383648;border-radius:17px;padding:22px;}\n.ap-side-title {color:#FFA886;font-size:.7rem;font-weight:800;letter-spacing:.13em;margin-bottom:16px;}\n.ap-small-label {color:#ACABBE;font-size:.73rem;margin-bottom:7px;}\n.ap-side-number {font-size:2rem;color:#FAF6F3;font-weight:800;margin-bottom:12px;letter-spacing:-.04em;}\n.ap-rule {height:1px;background:#393747;margin:15px 0;}\n.ap-mini {font-size:.76rem;color:#ABA8BC;line-height:1.7;}\n.ap-model-meta {color:#B0ABC5;font-size:.81rem;line-height:1.7;}\n.ap-footer {color:#8C8CA2;border-top:1px solid #303041;padding-top:22px;margin-top:30px;\n  font-size:.72rem;line-height:1.7;}\n@media(max-width:1000px) {\n [data-testid="stMainBlockContainer"],.block-container {padding:1.5rem 1.2rem 3rem;}\n .ap-workspace-head {flex-wrap:wrap;}\n .ap-overview-hero {padding:28px;}\n}\n@media(max-width:640px){\n .ap-overview-title{font-size:2.35rem;}\n .ap-overview-hero:after {opacity:.25;}\n .ap-page-badge{font-size:.63rem;}\n}\n\n/* Visitor-first Overview refinements: keep Obsidian 2.0 visual language. */\n.ap-overview-hero { margin-bottom:12px; padding-bottom:40px; }\n.ap-overview-hero .ap-overview-title { max-width:820px; }\n.ap-overview-actions { margin:0 0 29px; }\ndiv[data-testid="stButton"] button[kind="primary"] {\n  background:linear-gradient(105deg,#FF8658,#FFAD84)!important;\n  color:#1F1722!important;border:1px solid #FFAD84!important;\n  font-weight:810!important;border-radius:13px!important;min-height:52px;\n  box-shadow:0 8px 26px #FF865822;\n}\ndiv[data-testid="stButton"] button[kind="primary"] p {color:#1F1722!important;}\ndiv[data-testid="stButton"] button[kind="secondary"] {\n  border:1px solid #6A607C!important;background:#242130!important;\n  border-radius:13px!important;min-height:52px;\n}\n.ap-onboard-context {margin:10px 0 19px;padding:18px 0 15px;\n  display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:27px;\n  border-top:1px solid #353241;border-bottom:1px solid #353241;}\n.ap-onboard-context-item {padding:4px 6px 4px 0;}\n.ap-onboard-overline {font-size:.68rem;letter-spacing:.145em;color:#FFAF8D;font-weight:820;margin-bottom:9px;}\n.ap-onboard-heading {font-size:1.17rem;line-height:1.3;letter-spacing:-.025em;\n  font-weight:760;color:#F6F0F6;margin-bottom:7px;}\n.ap-onboard-copy {font-size:.83rem;line-height:1.72;color:#BEBBCE;}\n.ap-onboard-steps {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));\n  gap:0;border:1px solid #383344;border-radius:18px;overflow:hidden;background:#1D1C28;\n  margin:9px 0 18px;}\n.ap-onboard-step {padding:20px 19px;min-height:133px;position:relative;}\n.ap-onboard-step + .ap-onboard-step {border-left:1px solid #393546;}\n.ap-onboard-step-num {font-size:.71rem;letter-spacing:.08em;font-weight:800;\n  color:#FFAD8A;margin-bottom:13px;}\n.ap-onboard-step-head {font-size:.93rem;font-weight:760;color:#FAF5F8;margin-bottom:6px;}\n.ap-onboard-step-copy {font-size:.77rem;line-height:1.54;color:#BDB9CC;}\n.ap-onboard-note {font-size:.79rem;color:#ACA9BF;line-height:1.65;margin:8px 0 22px;}\n.ap-onboard-note b {color:#E4DFF1;}\n@media(max-width:850px) {\n  .ap-onboard-context{grid-template-columns:1fr;gap:12px;}\n  .ap-onboard-steps{grid-template-columns:1fr;}\n  .ap-onboard-step + .ap-onboard-step{border-left:0;border-top:1px solid #393546;}\n  .ap-overview-hero{padding:26px 23px;}\n}\n</style>',
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, footer: str) -> None:
    """Compact, accessible metric component used throughout the existing views."""
    st.markdown(
        f'<div class="ap-metric"><div class="ap-metric-label">{escape(label)}</div>'
        f'<div class="ap-metric-value">{escape(value)}</div>'
        f'<div class="ap-metric-footer">{escape(footer)}</div></div>',
        unsafe_allow_html=True,
    )


def section_title(number: str, heading: str) -> None:
    """Section markers are intentionally dropped in the modern layout."""
    st.markdown(
        f'<div class="ap-section-head"><span class="ap-section-title">{escape(heading)}</span></div>',
        unsafe_allow_html=True,
    )


def tune_plot(fig: go.Figure, *, height: int = 370) -> go.Figure:
    """Apply cohesive dark Plotly styling with readable labels."""
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=24, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Segoe UI, Arial, sans-serif", size=13),
        hoverlabel=dict(bgcolor=PANEL, font_color=TEXT, bordercolor=OUTLINE),
        xaxis=dict(gridcolor=GRID, linecolor=OUTLINE, zeroline=False,
                   title_font=dict(color=MUTED), tickfont=dict(color=SILVER)),
        yaxis=dict(gridcolor=GRID, linecolor=OUTLINE, zeroline=False,
                   title_font=dict(color=MUTED), tickfont=dict(color=SILVER)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=SILVER)),
        hovermode="x unified",
    )
    return fig


# -------------------------------------------------------------
# Dataset: use the project's existing validated data loader
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_fd001()


try:
    train, test, test_rul = get_data()
except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
    st.error("NASA FD001 data could not be loaded. Existing project files are safe.")
    st.code(
        r".\.venv\Scripts\python.exe download_dataset.py" + "\n"
        + r".\.venv\Scripts\python.exe check_dataset.py",
        language="powershell",
    )
    st.caption(f"Details: {exc}")
    st.stop()

life_by_engine = train.groupby("engine_id")["cycle"].max().sort_index()
engine_ids = [int(v) for v in life_by_engine.index.tolist()]
sensors = [f"sensor_{i}" for i in range(1, 22)]

NAV_ITEMS = [
    "Overview", "Engine Focus", "Fleet Patterns", "Data Explorer",
    "Data Laboratory", "RUL Studio", "Model Lab", "Optimization Hub",
    "Health Monitor", "Final Evaluation",
]
NAV_ICONS = {
    "Overview": "◫", "Engine Focus": "◉", "Fleet Patterns": "▥",
    "Data Explorer": "≡", "Data Laboratory": "✦", "RUL Studio": "◇",
    "Model Lab": "✧", "Optimization Hub": "✳", "Health Monitor": "◈",
    "Final Evaluation": "✺",
}
VIEW_DESCRIPTIONS = {
    "Overview": ("Your fleet, at a glance.", "A focused starting point for simulated engine intelligence."),
    "Engine Focus": ("Engine telemetry", "Inspect one engine's operating history and compare sensor behavior."),
    "Fleet Patterns": ("Fleet patterns", "Understand engine histories across the NASA FD001 training fleet."),
    "Data Explorer": ("Data explorer", "Browse original sensor observations and export selected records."),
    "Data Laboratory": ("Data laboratory", "Investigate data quality, drift, distributions and correlations."),
    "RUL Studio": ("RUL studio", "Explore known training labels and engineered sensor-history features."),
    "Model Lab": ("Model lab", "Compare Random Forest and XGBoost on engine-held-out validation data."),
    "Optimization Hub": ("Optimization hub", "Explore grouped tuning, engine errors and SHAP model explanations."),
    "Health Monitor": ("Health monitor", "Replay historical validation telemetry and examine illustrative RUL estimates."),
    "Final Evaluation": ("Final evaluation", "Review the official FD001 test snapshot: one estimate per test engine."),
}

if "ap_next_route" in st.session_state:
    st.session_state["ap_route"] = st.session_state.pop("ap_next_route")

with st.sidebar:
    st.markdown(
        '<div class="ap-side-brand"><div class="ap-side-mark">✦</div>'
        '<div><div class="ap-side-logo">AeroPredict <em>AI</em></div>'
        '<div class="ap-side-caption">OBSIDIAN FLIGHT DECK</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="ap-nav-label">EXPLORE WORKSPACES</div>', unsafe_allow_html=True)
    selected_view = st.radio(
        "Open a workspace", NAV_ITEMS, key="ap_route", label_visibility="collapsed",
        format_func=lambda value: f"{NAV_ICONS[value]}    {value}",
    )
    st.markdown(
        '<div class="ap-side-foot"><b>NASA FD001</b> · simulated historical data<br>'
        'Model predictions are educational and are not approved for aircraft maintenance.</div>',
        unsafe_allow_html=True,
    )

# Keep engine context out of the main workspace unless it is being used.
engine_id = engine_ids[0]
chosen_sensor = "sensor_2"
smooth = True
if selected_view in ("Engine Focus", "RUL Studio"):
    with st.sidebar:
        st.markdown('<div class="ap-nav-label">ENGINE CONTEXT</div>', unsafe_allow_html=True)
        engine_id = st.selectbox(
            "Training engine", engine_ids,
            format_func=lambda eng: f"ENG-{eng:03d}", key="ap_engine_select",
        )
        chosen_sensor = st.selectbox(
            "Sensor channel", sensors, index=1,
            format_func=lambda s: f"Sensor {s.split('_')[1].zfill(2)}",
            key="ap_sensor_select",
        )
        if selected_view == "Engine Focus":
            smooth = st.toggle("Smooth sensor signal", value=True, key="ap_smooth")

engine_df = train.loc[train["engine_id"] == engine_id].sort_values("cycle").copy()
life_cycles = int(engine_df["cycle"].max())
observation_cycle = life_cycles
if selected_view in ("Engine Focus", "RUL Studio"):
    with st.sidebar:
        observation_cycle = st.slider(
            "Inspect recorded cycle", min_value=int(engine_df["cycle"].min()),
            max_value=life_cycles, value=life_cycles, step=1,
            help="Choose a point in a complete, simulated training-engine history.",
        )

page_title, page_desc = VIEW_DESCRIPTIONS[selected_view]
st.markdown(
    '<div class="ap-workspace-head"><div>'
    '<div class="ap-page-eyebrow">AEROPREDICT / ' + escape(selected_view.upper()) + '</div>'
    '<div class="ap-page-title">' + escape(page_title) + '</div>'
    '<div class="ap-page-subtitle">' + escape(page_desc) + '</div></div>'
    '<div class="ap-page-badge"><i></i> C-MAPSS · FD001</div></div>',
    unsafe_allow_html=True,
)

if selected_view == "Overview":
    st.markdown('''
    <div class="ap-overview-hero">
      <div class="ap-overview-kicker">NASA C-MAPSS / PREDICTIVE MAINTENANCE DEMO</div>
      <div class="ap-overview-title">What if an engine could
        <span>tell us how much life it has left?</span></div>
      <div class="ap-overview-desc">AeroPredict AI explores that question using simulated jet-engine
        sensor histories. Select an engine, replay its readings, and see how machine learning
        estimates the remaining number of operating cycles.</div>
    </div>
    ''', unsafe_allow_html=True)

    # A true Streamlit button rather than an inert HTML link; uses the existing routed sidebar.
    primary, secondary, spacer = st.columns([1.5, 1.3, 2.2], gap="small")
    with primary:
        if st.button("Explore Engine Demo  →", type="primary", key="ap_start_demo", use_container_width=True):
            st.session_state["ap_next_route"] = "Health Monitor"
            st.rerun()
    with secondary:
        if st.button("View model results", type="secondary", key="ap_start_results", use_container_width=True):
            st.session_state["ap_next_route"] = "Final Evaluation"
            st.rerun()

    st.markdown('''
    <div class="ap-onboard-context">
      <div class="ap-onboard-context-item">
        <div class="ap-onboard-overline">THE PROBLEM</div>
        <div class="ap-onboard-heading">Unexpected machine failures are expensive.</div>
        <div class="ap-onboard-copy">Equipment degrades as it runs. Understanding that
          deterioration could help maintenance teams plan earlier instead of reacting
          after a breakdown.</div>
      </div>
      <div class="ap-onboard-context-item">
        <div class="ap-onboard-overline">WHAT WE BUILT</div>
        <div class="ap-onboard-heading">A way to explore predictive maintenance.</div>
        <div class="ap-onboard-copy">Our Random Forest and XGBoost models study
          recorded sensor patterns and estimate remaining useful life (RUL).
          Interactive charts and SHAP show the data behind those estimates.</div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    section_title("", "Your experience starts here")
    st.markdown('''
    <div class="ap-onboard-steps">
      <div class="ap-onboard-step">
        <div class="ap-onboard-step-num">01 / CHOOSE</div>
        <div class="ap-onboard-step-head">Pick an engine</div>
        <div class="ap-onboard-step-copy">Browse a preloaded NASA engine history. No upload needed.</div>
      </div>
      <div class="ap-onboard-step">
        <div class="ap-onboard-step-num">02 / REPLAY</div>
        <div class="ap-onboard-step-head">Move through its cycles</div>
        <div class="ap-onboard-step-copy">See how recorded sensor readings and model estimates change.</div>
      </div>
      <div class="ap-onboard-step">
        <div class="ap-onboard-step-num">03 / UNDERSTAND</div>
        <div class="ap-onboard-step-head">Explore the prediction</div>
        <div class="ap-onboard-step-copy">Compare models and inspect their evaluation and explanations.</div>
      </div>
    </div>
    <div class="ap-onboard-note"><b>About the demo:</b> NASA C-MAPSS uses simulated historical engine data, not live aircraft telemetry.
      Predictions are educational examples, not operational maintenance guidance. A model trained with
      a 125-cycle capped target cannot distinguish longer lifetimes accurately.</div>
    ''', unsafe_allow_html=True)

    section_title("", "Inside the dataset")
    headline = [
        ("Training engines", f"{len(engine_ids):,}", "COMPLETE HISTORIES"),
        ("Observations", f"{len(train):,}", "TRAINING RECORDS"),
        ("Sensor channels", "21", "PER CYCLE"),
        ("Test engines", f"{test['engine_id'].nunique():,}", "OFFICIAL FD001"),
    ]
    columns = st.columns(4, gap="medium")
    for column, (label, value, footer) in zip(columns, headline):
        with column:
            metric_card(label, value, footer)

    section_title("", "Evaluated on unseen engines")
    metrics_file = Path(__file__).resolve().parent / "outputs" / "m7_official_test_metrics.csv"
    if metrics_file.is_file():
        try:
            df_scores = pd.read_csv(metrics_file)
            selected = df_scores.loc[
                (df_scores["model"] == "XGBoost") &
                (df_scores["evaluation_target"] == "true_uncapped_rul")
            ]
            if selected.empty:
                raise ValueError("Missing uncapped XGBoost benchmark")
            score = selected.iloc[0]
            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Test MAE", f"{float(score['mae']):.2f}", "CYCLES · UNCAPPED")
            with c2:
                metric_card("Test RMSE", f"{float(score['rmse']):.2f}", "CYCLES · UNCAPPED")
            with c3:
                metric_card("Test R²", f"{float(score['r2']):.3f}", "100 TEST ENGINES")
            st.markdown(
                '<div class="ap-storyline"><strong>Reading these results:</strong> This is the official '
                'held-out FD001 test benchmark for our selected XGBoost model. '
                'Accuracy here does not imply real-aircraft maintenance reliability, '
                'and RUL above 125 cycles remains a modeling limitation.</div>',
                unsafe_allow_html=True,
            )
        except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
            st.info(f"Benchmark report could not be displayed: {exc}")
    else:
        st.info("Run evaluate_official_test.py to display your verified FD001 benchmark.")

    st.caption("Educational research demonstration · Simulated NASA C-MAPSS FD001 · No CSV upload required")

if selected_view == "Engine Focus":
    section_title("02 /", "Sensor timeline")
    graph_col, profile_col = st.columns([2.15, 1], gap="large")
    with graph_col:
        series = engine_df[chosen_sensor]
        main_chart = go.Figure()
        main_chart.add_trace(go.Scatter(
            x=engine_df["cycle"], y=series,
            name="Original reading", mode="lines",
            line=dict(color=LILAC, width=2),
            hovertemplate="Cycle %{x}<br>Reading %{y:.4f}<extra></extra>",
        ))
        if smooth:
            rolling = series.rolling(7, min_periods=1).mean()
            main_chart.add_trace(go.Scatter(
                x=engine_df["cycle"], y=rolling,
                name="7-cycle average", mode="lines",
                line=dict(color=ORANGE, width=3),
                hovertemplate="Cycle %{x}<br>Average %{y:.4f}<extra></extra>",
            ))
        main_chart.add_vline(
            x=observation_cycle,
            line_dash="dot",
            line_color="#FFBE9F",
            line_width=1.6,
        )
        main_chart.update_xaxes(title="Operating cycle")
        main_chart.update_yaxes(title=f"Sensor {chosen_sensor.split('_')[1]} reading")
        st.plotly_chart(tune_plot(main_chart, height=385), use_container_width=True)
        st.caption(
            "Historical simulated readings only. Sensor channels do not all share a physical unit. "
            "The vertical dotted line marks the cycle you selected."
        )

    with profile_col:
        selected_row = engine_df.loc[engine_df["cycle"] == observation_cycle].iloc[0]
        reading = float(selected_row[chosen_sensor])
        cycles_after = life_cycles - observation_cycle
        progress = observation_cycle / life_cycles * 100
        st.markdown(
            '<div class="ap-side">'
            '<div class="ap-side-title">◈ &nbsp; Historical engine profile</div>'
            '<div class="ap-small-label">Selected engine</div>'
            f'<div class="ap-side-number">ENG-{engine_id:03d}</div>'
            '<div class="ap-rule"></div>'
            '<div class="ap-small-label">Observed lifetime in training data</div>'
            f'<div class="ap-side-number">{life_cycles:,} <span style="font-size:.8rem;color:#B9B5C7">cycles</span></div>'
            '<div class="ap-small-label">Reading at inspected cycle</div>'
            f'<div class="ap-side-number">{reading:,.3f}</div>'
            '<div class="ap-rule"></div>'
            f'<div class="ap-mini">Cycle {observation_cycle:,} of {life_cycles:,} '
            f'({progress:.1f}% of the recorded history).<br>'
            f'{cycles_after:,} recorded cycles remain after this point.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="ap-note">This is the <b>known training history</b>, '
            'not a model prediction of engine failure.</div>',
            unsafe_allow_html=True,
        )

    section_title("03 /", "Compare sensor behavior")
    compare_cols = st.multiselect(
        "Choose 2–5 sensors to compare",
        options=sensors,
        default=["sensor_2", "sensor_11"],
        max_selections=5,
        format_func=lambda s: f"Sensor {s.split('_')[1].zfill(2)}",
        help="Readings are standardized individually so different sensor scales can be compared.",
    )
    if compare_cols:
        normalized = engine_df[["cycle"] + compare_cols].copy()
        constant_sensors = []
        for column in compare_cols:
            mean = float(normalized[column].mean())
            std = float(normalized[column].std(ddof=0))
            if np.isclose(std, 0):
                normalized[column] = 0.0
                constant_sensors.append(column)
            else:
                normalized[column] = (normalized[column] - mean) / std

        compare_fig = go.Figure()
        palette = [ORANGE, LILAC, "#F7C98C", "#B7B5CB", "#E3A4BD"]
        for index, sensor in enumerate(compare_cols):
            compare_fig.add_trace(go.Scatter(
                x=normalized["cycle"], y=normalized[sensor],
                mode="lines", name=sensor.replace("_", " ").title(),
                line=dict(color=palette[index], width=2.25),
                hovertemplate="Cycle %{x}<br>Z-score %{y:.2f}<extra></extra>",
            ))
        compare_fig.update_xaxes(title="Operating cycle")
        compare_fig.update_yaxes(title="Within-engine z-score")
        st.plotly_chart(tune_plot(compare_fig, height=310), use_container_width=True)
        st.caption("Standardized for comparison: (reading − engine mean) / engine standard deviation.")
        if constant_sensors:
            st.info("Constant-reading channels are shown as a flat zero line: "
                    + ", ".join(constant_sensors))
    else:
        st.info("Select at least one sensor to display a comparison.")

if selected_view == "Fleet Patterns":
    section_title("04 /", "Fleet history overview")
    st.markdown(
        '<div class="ap-hint">These plots summarize complete training histories. '
        'They do not estimate future failures.</div>',
        unsafe_allow_html=True,
    )
    hist_col, scatter_col = st.columns(2, gap="large")
    with hist_col:
        durations = life_by_engine.values
        histogram = go.Figure()
        histogram.add_trace(go.Histogram(
            x=durations, nbinsx=20,
            marker=dict(color=ORANGE, line=dict(color=INK, width=0.7)),
            hovertemplate="Engine lifetime: %{x} cycles<br>Count: %{y}<extra></extra>",
        ))
        histogram.update_xaxes(title="Recorded lifetime (cycles)")
        histogram.update_yaxes(title="Number of engines")
        st.markdown("**Training engine lifetime distribution**")
        st.plotly_chart(tune_plot(histogram, height=350), use_container_width=True)
    with scatter_col:
        scatter = go.Figure()
        scatter.add_trace(go.Scatter(
            x=life_by_engine.index, y=life_by_engine.values, mode="markers",
            marker=dict(color=LILAC, size=9, opacity=.85, line=dict(color=INK, width=.7)),
            hovertemplate="Engine %{x}<br>Lifetime %{y} cycles<extra></extra>",
        ))
        scatter.update_xaxes(title="Engine ID")
        scatter.update_yaxes(title="Recorded lifetime (cycles)")
        st.markdown("**Cycles recorded for each training engine**")
        st.plotly_chart(tune_plot(scatter, height=350), use_container_width=True)

    fleet_a, fleet_b, fleet_c = st.columns(3)
    with fleet_a:
        metric_card("Shortest history", f"{life_by_engine.min():,}", "RECORDED CYCLES")
    with fleet_b:
        metric_card("Median history", f"{life_by_engine.median():,.0f}", "RECORDED CYCLES")
    with fleet_c:
        metric_card("Longest history", f"{life_by_engine.max():,}", "RECORDED CYCLES")

if selected_view == "Data Explorer":
    section_title("05 /", "Dataset browser")
    st.markdown(
        '<div class="ap-hint">Inspect the raw NASA readings without modifying source values.</div>',
        unsafe_allow_html=True,
    )
    source = st.radio("Choose dataset", ["Training histories", "Test histories"], horizontal=True)
    active_data = train if source == "Training histories" else test
    source_engine_ids = sorted(int(i) for i in active_data["engine_id"].unique())
    browse_id = st.selectbox(
        "Select engine to browse", source_engine_ids,
        index=source_engine_ids.index(engine_id) if engine_id in source_engine_ids else 0,
        format_func=lambda engine: f"ENG-{engine:03d}",
        key="table_engine_id",
    )
    filtered = active_data[active_data["engine_id"] == browse_id].sort_values("cycle")
    st.dataframe(filtered, use_container_width=True, hide_index=True, height=370)
    st.download_button(
        "Download selected engine CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name=f"fd001_{'train' if source == 'Training histories' else 'test'}_engine_{browse_id:03d}.csv",
        mime="text/csv",
    )
    with st.expander("How is test RUL provided?"):
        st.write(
            "Training trajectories are complete up to simulated failure. Test trajectories "
            "stop earlier. NASA supplies a separate remaining-useful-life label for each "
            "test engine at its final observed cycle. Those labels will be used for model "
            "evaluation in a later milestone."
        )
        st.write(f"Test RUL labels loaded: **{len(test_rul):,}**")


# -------------------------------------------------------------
# Milestone 2: Data Laboratory (exploration of TRAIN data only)
# -------------------------------------------------------------
if selected_view == "Data Laboratory":
    section_title("06 /", "Data laboratory")
    st.markdown(
        '<div class="ap-hint">Audit sensor records, investigate distributions, and examine '
        'relationships without using test-set RUL labels.</div>',
        unsafe_allow_html=True,
    )
    train_quality = audit_dataset(train, "FD001 train")
    test_quality = audit_dataset(test, "FD001 test")
    quality_report = build_report(train, test, test_rul)

    st.markdown(
        '<div class="ap-spotlight"><b>Milestone 02 is about trustworthy data.</b> '
        'The source files remain unchanged. We check completeness, detect duplicate '
        'engine-cycle records, flag constant channels, and identify patterns in '
        'complete training histories. Predictions and learned feature scaling come later.</div>',
        unsafe_allow_html=True,
    )
    quality_tab, profile_tab, relation_tab, phase_tab = st.tabs(
        ["01  Quality scan", "02  Sensor profile", "03  Relationships", "04  Life-stage signals"]
    )

    with quality_tab:
        st.markdown("### Dataset integrity / at a glance")
        qa, qb, qc, qd = st.columns(4, gap="medium")
        with qa:
            metric_card("Training quality", train_quality["status"], "SOURCE VALIDATION")
        with qb:
            metric_card("Missing values", f"{train_quality['missing_cells'] + test_quality['missing_cells']:,}", "TRAIN + TEST")
        with qc:
            metric_card("Duplicate engine-cycle keys", f"{train_quality['duplicate_engine_cycles'] + test_quality['duplicate_engine_cycles']:,}", "TRAIN + TEST")
        with qd:
            metric_card("Constant training features", str(len(train_quality["constant_features"])), "FLAGGED FOR MODELING")

        if train_quality["status"] == "PASS" and test_quality["status"] == "PASS":
            st.success("Integrity checks passed on both FD001 training and test histories.")
        else:
            st.warning("Data quality checks flagged issues. Inspect the audit tables before preprocessing.")
        summary = pd.DataFrame([
            {"Dataset": name, "Engines": info["engines"], "Records": info["rows"],
             "Missing cells": info["missing_cells"], "Duplicate rows": info["duplicate_rows"],
             "Duplicate engine-cycle keys": info["duplicate_engine_cycles"],
             "Engine histories with gaps": info["engines_with_cycle_gaps"],
             "Status": info["status"]}
            for name, info in (("Train", train_quality), ("Test", test_quality))
        ])
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.markdown("**Channels with a single value across all training observations**")
        if train_quality["constant_features"]:
            st.markdown(", ".join(f"`{c}`" for c in train_quality["constant_features"]))
        else:
            st.caption("No completely constant operating-setting or sensor columns were found.")
        st.caption("Constant channels are retained in the cleaned CSVs but should be excluded from later modeling.")
        st.download_button(
            "Download quality report (JSON)",
            data=json.dumps(quality_report, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="fd001_quality_report.json", mime="application/json",
            key="quality_download",
        )
        st.markdown(
            '<div class="ap-quality-note"><b>Prepare your reusable CSVs:</b> '
            'Run <code>run_preprocessing.py</code> from the VS Code terminal. '
            'It validates and sorts the raw histories into <code>data/processed/</code>; '
            'it does not remove readings, create target labels, or fit scalers.</div>',
            unsafe_allow_html=True,
        )
        st.code(r".\.venv\Scripts\python.exe run_preprocessing.py", language="powershell")

    with profile_tab:
        st.markdown("### Sensor distribution / early versus late")
        st.markdown(
            '<div class="ap-hint">Comparing <b>known training</b> life stages can reveal '
            'sensor changes. Life stage is calculated from future failure cycles for EDA '
            'only, never as a prediction input.</div>', unsafe_allow_html=True,
        )
        profile_sensor = st.selectbox(
            "Inspect sensor distribution", sensors,
            index=sensors.index("sensor_11"), key="lab_profile_sensor",
            format_func=lambda v: "Sensor " + v.split("_")[-1].zfill(2),
        )
        stage_data = training_life_stage(train)
        early = stage_data.loc[stage_data["life_stage"] == "Early (0–25%)", profile_sensor]
        late = stage_data.loc[stage_data["life_stage"] == "Late (75–100%)", profile_sensor]
        dist_col, stats_col = st.columns([1.7, 1], gap="large")
        with dist_col:
            dist_fig = go.Figure()
            for values, label, color in (
                (early, "Early · first 25%", LILAC),
                (late, "Late · last 25%", ORANGE),
            ):
                dist_fig.add_trace(go.Histogram(
                    x=values, name=label, nbinsx=42, opacity=.67,
                    marker=dict(color=color), histnorm="probability density",
                    hovertemplate="Reading %{x:.4f}<br>Density %{y:.4f}<extra></extra>",
                ))
            dist_fig.update_layout(barmode="overlay")
            dist_fig.update_xaxes(title=f"{profile_sensor} · reading")
            dist_fig.update_yaxes(title="Density")
            st.plotly_chart(tune_plot(dist_fig, height=365), use_container_width=True)
        with stats_col:
            st.markdown("**Sensor reading summary**")
            sd = train[profile_sensor]
            stats = pd.DataFrame([
                ("Min", sd.min()), ("Median", sd.median()),
                ("Mean", sd.mean()), ("Max", sd.max()),
                ("Std. deviation", sd.std()), ("Unique readings", sd.nunique()),
            ], columns=["Statistic", "Value"])
            st.dataframe(stats, hide_index=True, use_container_width=True)
            if sd.nunique() <= 1:
                st.warning("This sensor is constant across training records; it cannot explain differences.")
        st.caption("Histogram values are descriptive summaries from simulated training engines; no diagnosis or failure model is implied.")

    with relation_tab:
        st.markdown("### Relationships between sensor channels")
        defaults = ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15", "sensor_21"]
        relations = st.multiselect(
            "Select 2–12 channels for Pearson correlation", sensors,
            default=defaults, max_selections=12,
            key="lab_correlations",
            format_func=lambda v: "Sensor " + v.split("_")[-1].zfill(2),
        )
        if len(relations) >= 2:
            nonconstant = [col for col in relations if train[col].nunique() > 1]
            excluded = [col for col in relations if col not in nonconstant]
            if len(nonconstant) >= 2:
                corr = train[nonconstant].corr(method="pearson")
                heat = go.Figure(data=go.Heatmap(
                    z=corr.values, x=nonconstant, y=nonconstant,
                    zmin=-1, zmax=1, zmid=0,
                    colorscale=[[0, "#B3A3FF"], [.5, "#292836"], [1, "#FF8658"]],
                    colorbar=dict(title="r", tickfont=dict(color=SILVER)),
                    hovertemplate="%{y} × %{x}<br>r = %{z:.3f}<extra></extra>",
                ))
                heat.update_yaxes(autorange="reversed")
                st.plotly_chart(tune_plot(heat, height=480), use_container_width=True)
                st.caption("Pearson correlation across training rows. Correlation alone is not causation, degradation, or model importance.")
            else:
                st.info("Select at least two sensors that vary across engine observations.")
            if excluded:
                st.info("Constant channels omitted from the correlation heatmap: " + ", ".join(excluded))
        else:
            st.info("Choose at least two sensors to plot a correlation heatmap.")

    with phase_tab:
        st.markdown("### Life-stage signals / exploratory drift")
        st.markdown(
            '<div class="ap-hint">How much do mean sensor readings differ between '
            'the first and last quarters of <b>training</b> engine histories?</div>',
            unsafe_allow_html=True,
        )
        # Use the same EDA-only life-stage table as the histogram above.
        life_view = training_life_stage(train)
        beginning = life_view.loc[life_view["life_stage"] == "Early (0–25%)", sensors].mean()
        ending = life_view.loc[life_view["life_stage"] == "Late (75–100%)", sensors].mean()
        sd = train[sensors].std(ddof=0)
        valid = sd[sd.gt(1e-12)].index
        shift = ((ending[valid] - beginning[valid]) / sd[valid]).sort_values()
        if not shift.empty:
            drift_fig = go.Figure(go.Bar(
                x=shift.values,
                y=[c.replace("sensor_", "S") for c in shift.index],
                orientation="h",
                marker=dict(color=[ORANGE if v >= 0 else LILAC for v in shift.values]),
                hovertemplate="Mean change / overall SD: %{x:.3f}<extra></extra>",
            ))
            drift_fig.add_vline(x=0, line_color="rgba(245,241,237,.5)")
            drift_fig.update_xaxes(title="Late − early mean / training standard deviation")
            drift_fig.update_yaxes(title="Sensor channel")
            st.plotly_chart(tune_plot(drift_fig, height=610), use_container_width=True)
        st.markdown(
            '<div class="ap-quality-note"><b>Interpretation:</b> Large absolute differences '
            'show changes across known engine histories, not proven causes or predictive '
            'feature importance. Future life percentages are EDA-only and must not '
            'be included as ML features.</div>', unsafe_allow_html=True,
        )

# -------------------------------------------------------------
# Milestone 3 — RUL Studio (known TRAIN labels, no ML predictions)
# -------------------------------------------------------------
if selected_view == "RUL Studio":
    section_title("07 /", "RUL & feature engineering studio")
    st.markdown(
        '<div class="ap-m3-banner"><span class="ap-status-badge">MILESTONE 03</span><br>'
        '<b>From sensor readings to model-ready signals.</b> This view shows '
        'the <b>known remaining life</b> of complete <b>training</b> histories. '
        'The rolling means, standard deviations, and five-cycle deltas use only '
        'readings available up to the selected cycle. <b>No model predictions are shown yet.</b></div>',
        unsafe_allow_html=True,
    )

    engine_features = add_causal_features(engine_df)
    full_labels = training_rul_labels(engine_df)
    selected_target = full_labels.loc[full_labels["cycle"] == observation_cycle].iloc[0]
    at_cycle = engine_features.loc[engine_features["cycle"] == observation_cycle].iloc[0]
    true_rul = int(selected_target["rul_true"])
    capped_rul = int(selected_target["rul_capped_125"])
    mean_5 = float(at_cycle[f"{chosen_sensor}_mean_5"])
    mean_20 = float(at_cycle[f"{chosen_sensor}_mean_20"])

    r1, r2, r3, r4 = st.columns(4, gap="medium")
    with r1:
        metric_card("Known remaining life", f"{true_rul:,}", "TRAINING GROUND TRUTH · CYCLES")
    with r2:
        metric_card("Optional capped target", f"{capped_rul:,}", "MAX 125 CYCLES")
    with r3:
        metric_card("5-cycle sensor mean", f"{mean_5:,.2f}", chosen_sensor.upper())
    with r4:
        metric_card("20-cycle sensor mean", f"{mean_20:,.2f}", chosen_sensor.upper())

    st.markdown(
        '<div class="ap-hint">Uses ENG-'
        f'{engine_id:03d}, cycle {observation_cycle} and {escape(chosen_sensor)} '
        'from the engine selector above. These targets are shown for learning only.</div>',
        unsafe_allow_html=True,
    )
    plot_a, plot_b = st.columns([1.2, 1], gap="large")
    with plot_a:
        st.markdown("**Historical RUL labels / this training engine**")
        use_capped = st.toggle("Show optional 125-cycle cap", value=False, key="m3_target_cap")
        target_name = "rul_capped_125" if use_capped else "rul_true"
        rul_fig = go.Figure()
        rul_fig.add_trace(go.Scatter(
            x=full_labels["cycle"], y=full_labels[target_name],
            name="Known RUL target", mode="lines",
            line=dict(color=ORANGE, width=3.5),
            fill="tozeroy", fillcolor="rgba(255,134,88,0.09)",
            hovertemplate="Cycle %{x}<br>Known RUL %{y} cycles<extra></extra>",
        ))
        rul_fig.add_vline(x=observation_cycle, line_color=LILAC, line_dash="dot")
        rul_fig.update_xaxes(title="Operating cycle")
        rul_fig.update_yaxes(title="Remaining useful life / cycles")
        st.plotly_chart(tune_plot(rul_fig, height=370), use_container_width=True)
        st.caption(
            "The 125-cycle target cap is an optional modeling convention; "
            "it is NOT the physical lifetime and does not change the true RUL."
        )

    with plot_b:
        st.markdown(f"**{chosen_sensor} / causal sensor statistics**")
        signal_fig = go.Figure()
        signal_fig.add_trace(go.Scatter(
            x=engine_features["cycle"], y=engine_features[chosen_sensor],
            name="Sensor reading", line=dict(color="#D7D4E8", width=1.3),
            hovertemplate="Cycle %{x}<br>Reading %{y:.3f}<extra></extra>",
        ))
        for window, color in ((5, ORANGE), (20, LILAC)):
            signal_fig.add_trace(go.Scatter(
                x=engine_features["cycle"],
                y=engine_features[f"{chosen_sensor}_mean_{window}"],
                name=f"Causal mean / {window}",
                line=dict(color=color, width=2.5),
                hovertemplate="Cycle %{x}<br>Average %{y:.3f}<extra></extra>",
            ))
        signal_fig.add_vline(x=observation_cycle, line_color="#FFBE9F", line_dash="dot")
        signal_fig.update_xaxes(title="Operating cycle")
        signal_fig.update_yaxes(title="Sensor measurement")
        st.plotly_chart(tune_plot(signal_fig, height=370), use_container_width=True)
        st.caption("Each average uses the selected cycle and earlier observations only.")

    section_title("08 /", "Model-ready features")
    st.markdown(
        '<div class="ap-hint">Each sensor contributes 5 engineered columns: '
        '5- and 20-cycle mean, 5- and 20-cycle standard deviation, and the 5-cycle delta. '
        'Early short histories use available readings; unavailable 5-cycle deltas are set to zero.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("**Feature preview / selected engine**")
    sample_columns = ["engine_id", "cycle", chosen_sensor,
                      f"{chosen_sensor}_mean_5", f"{chosen_sensor}_std_5",
                      f"{chosen_sensor}_mean_20", f"{chosen_sensor}_std_20",
                      f"{chosen_sensor}_delta_5"]
    st.dataframe(
        engine_features[sample_columns].tail(12),
        use_container_width=True, hide_index=True, height=360,
    )
    st.caption("Engine ID identifies a trajectory. It is not included in the model feature list.")

    project_root = Path(__file__).resolve().parent
    manifest_path = project_root / "data" / "processed" / "feature_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("milestone") != 3:
                st.warning("Found an older feature manifest. Re-run the Milestone 3 pipeline.")
            else:
                st.success(
                    f"Feature pipeline ready: {manifest['train_rows']:,} training rows, "
                    f"{manifest['test_last_rows']:,} final test observations, "
                    f"{manifest['model_feature_count']} candidate numeric model features."
                )
        except (OSError, ValueError, KeyError) as error:
            st.warning(f"Feature manifest could not be read: {error}")
    else:
        st.info("Feature files not exported yet. Run the Milestone 3 command below in VS Code.")
        st.code(r".\.venv\Scripts\python.exe run_feature_engineering.py", language="powershell")

    st.markdown(
        '<div class="ap-quality-note"><b>Leakage safeguard:</b> Engine features contain '
        'no targets and use only current/past readings. In Model Lab, models '
        'validate on <b>different engine IDs</b>; NASA official test truth '
        'is not used for training or model selection.</div>',
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------
# Milestone 4 — ML Model Lab: strictly held-out TRAIN engine histories
# -------------------------------------------------------------
if selected_view == "Model Lab":
    section_title("09 /", "Model training & validation laboratory")
    st.markdown(
        '<div class="ap-model-callout"><span class="ap-status-badge">MILESTONE 04</span><br>'
        '<b>Predict remaining useful life with two machine-learning models.</b> '
        'Random Forest and XGBoost are trained on separate training-engine histories '
        'and compared using <b>unseen validation engines</b>. '
        'These results are <b>not</b> scores on the official NASA test set or '
        'evidence of safe real-aircraft operation.</div>',
        unsafe_allow_html=True,
    )
    root = Path(__file__).resolve().parent
    model_summary_path = root / "models" / "training_summary.json"
    prediction_file = root / "outputs" / "validation_predictions.csv"
    importance_file = root / "outputs" / "feature_importance.csv"
    if not all(path.is_file() for path in [model_summary_path, prediction_file, importance_file]):
        st.info("Your model-training outputs haven't been generated yet. Run the command below in VS Code after Milestone 3.")
        st.code(r".\.venv\Scripts\python.exe train_models.py", language="powershell")
        st.caption(r"Install new requirements first with: .\.venv\Scripts\python.exe -m pip install -r requirements.txt")
    else:
        try:
            model_summary = json.loads(model_summary_path.read_text(encoding="utf-8"))
            validation_preds = pd.read_csv(prediction_file)
            importance_data = pd.read_csv(importance_file)
            required_pred = {"engine_id", "cycle", "actual_rul", "pred_random_forest", "pred_xgboost"}
            required_importance = {"model", "feature", "importance"}
            if (model_summary.get("milestone") != 4
                or not required_pred.issubset(validation_preds.columns)
                or not required_importance.issubset(importance_data.columns)):
                raise ValueError("Missing/incompatible Milestone 4 model output columns")
            scores = model_summary["validation_metrics"]
            split_details = model_summary["split"]
            winner = model_summary["best_validation_model"]
        except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError) as error:
            st.error(f"The model output files could not be loaded: {error}")
            st.code(r".\.venv\Scripts\python.exe train_models.py", language="powershell")
        else:
            target_name = model_summary["target"]
            train_n = int(split_details["training_engines"])
            val_n = int(split_details["validation_engines"])
            st.markdown(
                f'<div class="ap-model-meta"><b>Training target:</b> {escape(target_name)} &nbsp;·&nbsp; '
                f'<b>Engine split:</b> {train_n} training / {val_n} validation &nbsp;·&nbsp; '
                f'<b>Feature candidates:</b> {int(model_summary["candidate_feature_count"])} '
                f'&nbsp;·&nbsp; <b>Best validation RMSE:</b> {escape(winner)}</div>',
                unsafe_allow_html=True,
            )
            st.caption(model_summary["target_description"])
            if model_summary.get("quick_training_mode"):
                st.warning("Quick training mode is enabled. Re-run without --quick for the full Milestone 4 models.")

            section_title("10 /", "Compare trained models")
            score_rows = []
            for model_name in ("Random Forest", "XGBoost"):
                record = scores[model_name]
                score_rows.append({"Model": model_name, "MAE (cycles) ↓": record["mae"],
                                   "RMSE (cycles) ↓": record["rmse"], "R² ↑": record["r2"]})
            st.dataframe(pd.DataFrame(score_rows), hide_index=True, use_container_width=True)
            st.caption("MAE and RMSE measure error in cycles on the chosen validation target; smaller is better. R² closer to 1 is better. These are scores over validation rows.")
            chosen_model = st.selectbox(
                "Explore predictions from", ["Random Forest", "XGBoost"],
                index=0 if winner == "Random Forest" else 1, key="m4_model_selector",
            )
            pred_col = "pred_random_forest" if chosen_model == "Random Forest" else "pred_xgboost"
            chosen_scores = scores[chosen_model]
            m4a, m4b, m4c, m4d = st.columns(4, gap="medium")
            with m4a:
                metric_card("Validation MAE", f'{chosen_scores["mae"]:.2f}', "CYCLES · LOWER IS BETTER")
            with m4b:
                metric_card("Validation RMSE", f'{chosen_scores["rmse"]:.2f}', "CYCLES · LOWER IS BETTER")
            with m4c:
                metric_card("Validation R²", f'{chosen_scores["r2"]:.3f}', "UNSEEN ENGINE HISTORIES")
            with m4d:
                metric_card("Validation engines", str(val_n), "NO ENGINE-LEVEL OVERLAP")

            section_title("11 /", "Actual versus predicted")
            valid_ids = sorted(map(int, validation_preds["engine_id"].unique()))
            selected_validation_engine = st.selectbox(
                "Inspect a held-out validation engine", valid_ids,
                key="m4_validation_engine", format_func=lambda eng: f"ENG-{eng:03d}",
            )
            valid_track = validation_preds.loc[
                validation_preds["engine_id"] == selected_validation_engine
            ].sort_values("cycle")
            track_graph = go.Figure()
            track_graph.add_trace(go.Scatter(
                x=valid_track["cycle"], y=valid_track["actual_rul"],
                mode="lines", name="Known validation RUL",
                line=dict(color=LILAC, width=3.2),
            ))
            track_graph.add_trace(go.Scatter(
                x=valid_track["cycle"], y=valid_track[pred_col],
                mode="lines", name=f"{chosen_model} prediction",
                line=dict(color=ORANGE, width=2.8),
            ))
            track_graph.update_xaxes(title="Operating cycle")
            track_graph.update_yaxes(title="RUL / cycles")
            st.plotly_chart(tune_plot(track_graph, height=365), use_container_width=True)
            st.caption(f"ENG-{selected_validation_engine:03d} belongs to the unseen validation group. The ground-truth line comes from historical training labels, not from official NASA test labels.")

            left, right = st.columns([1.18, 1], gap="large")
            with left:
                st.markdown("**Predicted versus actual / validation rows**")
                sampled = validation_preds.sample(n=min(3000, len(validation_preds)), random_state=42)
                scatter = go.Figure(go.Scattergl(
                    x=sampled["actual_rul"], y=sampled[pred_col],
                    mode="markers", marker=dict(color=LILAC, size=4, opacity=0.45),
                    name="Validation records",
                    hovertemplate="Actual %{x:.1f}<br>Predicted %{y:.1f}<extra></extra>",
                ))
                upper = float(max(validation_preds["actual_rul"].max(), validation_preds[pred_col].max(), 1))
                scatter.add_trace(go.Scatter(
                    x=[0, upper], y=[0, upper], mode="lines",
                    name="Perfect agreement", line=dict(color=ORANGE, dash="dash"),
                ))
                scatter.update_xaxes(title="Actual RUL / cycles")
                scatter.update_yaxes(title="Predicted RUL / cycles")
                st.plotly_chart(tune_plot(scatter, height=355), use_container_width=True)
            with right:
                st.markdown("**Residual distribution / validation rows**")
                residuals = validation_preds[pred_col] - validation_preds["actual_rul"]
                resid_fig = go.Figure(go.Histogram(
                    x=residuals, nbinsx=45, marker=dict(color=ORANGE, opacity=0.8),
                    name="Prediction minus actual",
                ))
                resid_fig.add_vline(x=0, line_color=LILAC, line_dash="dash")
                resid_fig.update_xaxes(title="Prediction error (cycles)")
                resid_fig.update_yaxes(title="Number of validation observations")
                st.plotly_chart(tune_plot(resid_fig, height=355), use_container_width=True)

            section_title("12 /", "Which features does the model rely on?")
            chosen_importance = importance_data.loc[
                importance_data["model"] == chosen_model
            ].nlargest(15, "importance").sort_values("importance")
            if len(chosen_importance):
                feat_plot = go.Figure(go.Bar(
                    x=chosen_importance["importance"],
                    y=chosen_importance["feature"], orientation="h",
                    marker=dict(color=ORANGE), name="Model feature importance",
                ))
                feat_plot.update_xaxes(title="Built-in model feature-importance score")
                feat_plot.update_yaxes(title="Feature")
                st.plotly_chart(tune_plot(feat_plot, height=485), use_container_width=True)
            st.caption("Model importances are descriptive, model-specific scores, not proof that a sensor causes degradation. Constant features are filtered using training engines only.")
            st.download_button(
                "Download validation predictions (CSV)",
                data=validation_preds.to_csv(index=False),
                file_name="aeropredict_validation_predictions.csv",
                mime="text/csv", key="m4_download_predictions",
            )
            st.markdown(
                '<div class="ap-quality-note"><b>Evaluation protocol:</b> All measurements here '
                'come from 20% held-out <b>training engines</b>. No engine appears in both '
                'training and validation. The official NASA FD001 <b>test RUL labels were not used</b> '
                'for any model fitting, hyperparameter selection, or these charts. '
                'This system is an educational simulation prototype, not a certified '
                'predictive maintenance product.</div>', unsafe_allow_html=True,
            )


# -------------------------------------------------------------
# Milestone 05 / optimization & explainability (no NASA final test)
# -------------------------------------------------------------
if selected_view == "Optimization Hub":
    section_title("13 /", "Optimization & explainability hub")
    st.markdown(
        '<div class="ap-model-callout"><b>MILESTONE 05 / UNDERSTAND THE ESTIMATE.</b> '
        'Compare grouped cross-validation candidates, inspect engine-specific errors, '
        'and explore SHAP attributions for individual historical predictions. '
        'All graphics use simulated <b>training trajectories</b>; the official NASA test '
        'is evaluated separately in the Milestone 7 results tab.</div>',
        unsafe_allow_html=True,
    )
    opt_root = Path(__file__).resolve().parent
    opt_summary_path = opt_root / "models" / "optimization_summary.json"
    opt_output_files = {
        "cv": opt_root / "outputs" / "m5_group_cv_results.csv",
        "pred": opt_root / "outputs" / "m5_validation_predictions.csv",
        "errors": opt_root / "outputs" / "m5_engine_errors.csv",
        "global": opt_root / "outputs" / "m5_shap_global.csv",
        "local": opt_root / "outputs" / "m5_shap_local.csv",
    }
    if not opt_summary_path.is_file() or not all(path.is_file() for path in opt_output_files.values()):
        st.info("Milestone 5 files not generated yet. Install requirements and run the optimization script in your project folder.")
        st.code(r".\.venv\Scripts\python.exe -m pip install -r requirements.txt" + "\n" +
                r".\.venv\Scripts\python.exe optimize_models.py", language="powershell")
    else:
        try:
            opt_summary = json.loads(opt_summary_path.read_text(encoding="utf-8"))
            opt_cv = pd.read_csv(opt_output_files["cv"])
            opt_predictions = pd.read_csv(opt_output_files["pred"])
            opt_engine_errors = pd.read_csv(opt_output_files["errors"])
            opt_global = pd.read_csv(opt_output_files["global"])
            opt_local = pd.read_csv(opt_output_files["local"])
            if opt_summary.get("milestone") != 5:
                raise ValueError("Expected Milestone 5 optimization summary")
            for df, required in (
                (opt_cv, {"model","candidate","fold","rmse","mae"}),
                (opt_predictions, {"engine_id","cycle","actual_rul","pred_random_forest","pred_xgboost"}),
                (opt_engine_errors, {"model","engine_id","engine_mae","end_cycle_abs_error"}),
                (opt_global, {"model","feature","mean_absolute_shap"}),
                (opt_local, {"model","engine_id","cycle","feature","shap_value","base_value",
                             "raw_prediction","display_prediction","actual_rul"}),
            ):
                if not required.issubset(df.columns):
                    raise ValueError("Some Milestone 5 exports are missing expected columns")
            selected_models = opt_summary["selected_cv_candidates"]
            opt_metrics = opt_summary["validation_metrics"]
        except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError) as exc:
            st.error(f"Could not read Milestone 5 outputs: {exc}")
            st.code(r".\.venv\Scripts\python.exe optimize_models.py", language="powershell")
        else:
            if opt_summary.get("quick_mode"):
                st.warning("Quick mode was used. Run optimize_models.py without --quick for normal training.")
            st.caption(f"Target: {opt_summary['target']} · {opt_summary['split']['training_engines']} training engines · "
                       f"{opt_summary['split']['validation_engines']} held-out engines · 3-fold inner GroupKFold")

            section_title("14 /", "Cross-validation candidate search")
            cv_display = (opt_cv.groupby(["model", "candidate"], as_index=False)
                          .agg(mean_cv_rmse=("rmse", "mean"), mean_cv_mae=("mae", "mean"),
                               folds=("fold", "nunique")))
            st.dataframe(cv_display.round(3), hide_index=True, use_container_width=True)
            st.caption("Candidates are selected by mean validation RMSE across three GroupKFold folds "
                       "from the training engines only. No validation-engine IDs are mixed into these folds.")

            compare_rows = []
            for family in ("Random Forest", "XGBoost"):
                metrics = opt_metrics[family]
                compare_rows.append({
                    "Model": family,
                    "CV selection": selected_models[family]["candidate"],
                    "Mean CV RMSE": selected_models[family]["cv_rmse"],
                    "Validation MAE": metrics["mae"],
                    "Validation RMSE": metrics["rmse"],
                    "Validation R²": metrics["r2"],
                })
            st.dataframe(pd.DataFrame(compare_rows), hide_index=True, use_container_width=True)
            st.caption("These validation engines were already inspected in Milestone 4; "
                       "they are not an untouched final benchmark. Official NASA test evaluation is shown separately in Milestone 7.")

            section_title("15 /", "Engine-by-engine reliability")
            opt_selected = st.selectbox(
                "Choose optimized model", ["Random Forest", "XGBoost"],
                index=0 if opt_summary["best_validation_rmse_model"] == "Random Forest" else 1,
                key="m5_family",
            )
            score = opt_metrics[opt_selected]
            m51, m52, m53, m54 = st.columns(4, gap="medium")
            with m51:
                metric_card("Validation MAE", f"{score['mae']:.2f}", "CYCLES / LOWER IS BETTER")
            with m52:
                metric_card("Validation RMSE", f"{score['rmse']:.2f}", "CYCLES / LOWER IS BETTER")
            with m53:
                metric_card("Validation R²", f"{score['r2']:.3f}", "HELD-OUT TRAINING ENGINES")
            with m54:
                metric_card("Engine groups", str(opt_summary["split"]["validation_engines"]), "NO TRAIN/VALIDATION OVERLAP")

            err_df = opt_engine_errors.loc[opt_engine_errors["model"] == opt_selected].sort_values("engine_mae", ascending=False)
            engine_bar = go.Figure(go.Bar(
                x=[f"ENG-{int(v):03d}" for v in err_df["engine_id"]],
                y=err_df["engine_mae"],
                marker=dict(color=ORANGE), name="Per-engine MAE",
                hovertemplate="%{x}<br>MAE: %{y:.2f} cycles<extra></extra>",
            ))
            engine_bar.update_xaxes(title="Validation engine", tickangle=-45)
            engine_bar.update_yaxes(title="Mean absolute error / cycles")
            st.plotly_chart(tune_plot(engine_bar, height=365), use_container_width=True)
            st.caption("Per-engine MAE averages prediction errors within each held-out engine history. "
                       "This may differ from row-weighted fleet-wide MAE.")
            st.dataframe(err_df[["engine_id", "observations", "engine_mae", "engine_rmse", "end_cycle_abs_error"]].round(3),
                         hide_index=True, use_container_width=True)

            section_title("16 /", "Explain predictions with SHAP")
            st.markdown('<div class="ap-note"><b>Interpretation:</b> Positive SHAP contributions raise the '
                        'raw predicted RUL, while negative ones lower it. These are model explanations—not '
                        'proof that changing any sensor will prolong engine life.</div>',
                        unsafe_allow_html=True)
            global_df = (opt_global.loc[opt_global["model"] == opt_selected]
                         .nlargest(15, "mean_absolute_shap").sort_values("mean_absolute_shap"))
            shap_chart = go.Figure(go.Bar(
                x=global_df["mean_absolute_shap"], y=global_df["feature"],
                orientation="h", marker=dict(color=LILAC),
                name="Mean absolute SHAP contribution",
            ))
            shap_chart.update_xaxes(title="Mean |SHAP| / predicted RUL cycles")
            shap_chart.update_yaxes(title="Sensor feature")
            st.plotly_chart(tune_plot(shap_chart, height=470), use_container_width=True)
            st.caption("Global SHAP values summarize a small deterministic sample of held-out engine cycles; "
                       "they are not calculated across every validation row.")

            local_df = opt_local.loc[opt_local["model"] == opt_selected]
            available_engines = sorted(int(x) for x in local_df["engine_id"].unique())
            if available_engines:
                left_sel, right_sel = st.columns(2, gap="large")
                with left_sel:
                    local_eng = st.selectbox("Explain engine", available_engines,
                                             format_func=lambda e: f"ENG-{e:03d}", key="m5_engine")
                with right_sel:
                    cycles = sorted(int(x) for x in local_df.loc[
                        local_df["engine_id"] == local_eng, "cycle"].unique())
                    local_cycle = st.selectbox("Explain observed cycle", cycles, key="m5_cycle")
                chosen_explanation = local_df.loc[
                    (local_df["engine_id"] == local_eng) & (local_df["cycle"] == local_cycle)
                ].sort_values("shap_value")
                chosen_row = chosen_explanation.iloc[0]
                stat1, stat2, stat3 = st.columns(3)
                with stat1:
                    metric_card("Observed RUL", f"{chosen_row['actual_rul']:.1f}", "VALIDATION LABEL")
                with stat2:
                    metric_card("Model estimate", f"{chosen_row['display_prediction']:.1f}", "CAPPED / NONNEGATIVE")
                with stat3:
                    metric_card("Raw tree estimate", f"{chosen_row['raw_prediction']:.1f}", "WHAT SHAP EXPLAINS")
                contrib_fig = go.Figure(go.Bar(
                    x=chosen_explanation["shap_value"],
                    y=chosen_explanation["feature"], orientation="h",
                    marker=dict(color=[ORANGE if v > 0 else LILAC for v in chosen_explanation["shap_value"]]),
                    customdata=chosen_explanation[["feature_value"]].to_numpy(),
                    hovertemplate="%{y}<br>SHAP: %{x:+.2f} cycles<br>Feature: %{customdata[0]:.3f}<extra></extra>",
                ))
                contrib_fig.add_vline(x=0, line_color=SILVER, line_width=1)
                contrib_fig.update_xaxes(title="Contribution to raw predicted RUL / cycles")
                contrib_fig.update_yaxes(title="Feature / highest 10 |SHAP| values")
                st.plotly_chart(tune_plot(contrib_fig, height=425), use_container_width=True)
                st.caption(f"SHAP base value: {chosen_row['base_value']:.2f} cycles. "
                           "The chart shows only the ten largest contributions; other features are omitted. "
                           "Raw tree prediction may differ from the capped/nonnegative displayed RUL.")

            st.download_button("Download CV results (CSV)", opt_cv.to_csv(index=False),
                               "aeropredict_m5_cv.csv", "text/csv", key="m5_dl_cv")
            st.download_button("Download engine-error diagnostics (CSV)",
                               opt_engine_errors.to_csv(index=False),
                               "aeropredict_m5_engine_errors.csv", "text/csv", key="m5_dl_engines")
            st.markdown('<div class="ap-quality-note"><b>Model integrity:</b> All features use '
                        'the current/earlier sensor measurements only. The grouped CV folds and final '
                        'validation engines are distinct. SHAP uses the variance-filtered features '
                        'from the fitted model, and does not infer physical causality. '
                        '<b>Final NASA test scores are intentionally not shown here.</b></div>',
                        unsafe_allow_html=True)



# -------------------------------------------------------------
# Milestone 06 / historical inference and health replay
# -------------------------------------------------------------
if selected_view == "Health Monitor":
    section_title("17 /", "Engine health monitoring / historical replay")
    st.markdown(
        '<div class="ap-m6-banner"><div class="ap-m6-kicker">MILESTONE 06 / OFFLINE FLIGHT-DECK SIMULATOR</div>'
        '<div class="ap-m6-title">REPLAY THE SIGNAL. ESTIMATE THE LIFE.</div>'
        '<div class="ap-m6-description">Step through a previously unseen validation engine history, '
        'inspect sensor values available at that cycle, and review its saved model estimates. '
        'These are NASA simulated historical recordings — <b>not live telemetry</b>. '
        'Urgency bands are educational rules, not validated failure probabilities or real maintenance instructions.</div></div>',
        unsafe_allow_html=True,
    )

    root_m6 = Path(__file__).resolve().parent
    replay_path = root_m6 / "outputs" / "m6_monitoring_replay.csv"
    meta_path = root_m6 / "outputs" / "m6_monitoring_summary.json"
    if not (replay_path.is_file() and meta_path.is_file()):
        st.info("Monitoring files have not been generated yet. Your existing data and models are unchanged.")
        st.code(r".\.venv\Scripts\python.exe prepare_monitoring.py", language="powershell")
        st.caption("This uses your Milestone 5 optimized models and the same validation-engine histories. "
                   "No official NASA test data are opened.")
    else:
        try:
            m6_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            replay = pd.read_csv(replay_path)
            expected = {"engine_id", "cycle", "actual_rul", "pred_random_forest", "pred_xgboost", *MONITORED_SENSORS}
            if m6_meta.get("milestone") != 6 or not expected.issubset(replay.columns):
                raise ValueError("Monitoring file is missing expected M6 fields")
            if replay.duplicated(["engine_id", "cycle"]).any():
                raise ValueError("Duplicate engine-cycle observations")
            allowed_engines = sorted(map(int, replay["engine_id"].unique()))
            if allowed_engines != sorted(map(int, m6_meta["validation_engines"])):
                raise ValueError("Monitoring history does not match the validation engines")
            if replay.empty or not np.isfinite(replay[list(expected - {"engine_id", "cycle"})].to_numpy(dtype=float)).all():
                raise ValueError("Monitoring data is empty or has invalid numeric values")
        except (ValueError, KeyError, TypeError, OSError, pd.errors.ParserError) as m6_error:
            st.error(f"Monitoring files could not be loaded: {m6_error}")
            st.code(r".\.venv\Scripts\python.exe prepare_monitoring.py", language="powershell")
        else:
            if m6_meta.get("quick_mode"):
                st.warning("The optimized models came from M5 --quick. Rerun optimize_models.py normally, "
                           "then prepare_monitoring.py for standard artifacts.")
            st.caption(f"Replay source: {len(allowed_engines)} validation engines from the Milestone 5 engine split. "
                       "Official NASA test evaluation is handled separately in Milestone 7.")
            first_c, second_c, third_c = st.columns([1.15, 1.08, 1], gap="large")
            with first_c:
                monitor_engine = st.selectbox(
                    "Replay validation engine", allowed_engines,
                    format_func=lambda eid: f"ENG-{eid:03d}", key="m6_validation_engine",
                )
            with second_c:
                monitor_model = st.selectbox(
                    "RUL model", ["XGBoost", "Random Forest"],
                    index=0 if m6_meta.get("chosen_default_model") == "XGBoost" else 1,
                    key="m6_model_choice",
                )
            with third_c:
                monitor_sensor = st.selectbox(
                    "Monitor sensor", list(MONITORED_SENSORS),
                    index=1, format_func=lambda s: f"Sensor {s.split('_')[1].zfill(2)}",
                    key="m6_sensor_choice",
                )

            history_all = (replay.loc[replay["engine_id"] == monitor_engine]
                           .sort_values("cycle").reset_index(drop=True))
            cycle_min = int(history_all["cycle"].min())
            cycle_max = int(history_all["cycle"].max())
            if st.session_state.get("m6_previous_engine") != monitor_engine:
                st.session_state["m6_replay_cycle"] = cycle_min
                st.session_state["m6_previous_engine"] = monitor_engine

            left_btn, right_btn, latest_btn = st.columns([1, 1, 1], gap="medium")
            with left_btn:
                if st.button("◀ Previous cycle", disabled=st.session_state["m6_replay_cycle"] <= cycle_min,
                             key="m6_back", use_container_width=True):
                    st.session_state["m6_replay_cycle"] -= 1
            with right_btn:
                if st.button("Next cycle ▶", disabled=st.session_state["m6_replay_cycle"] >= cycle_max,
                             key="m6_next", use_container_width=True):
                    st.session_state["m6_replay_cycle"] += 1
            with latest_btn:
                if st.button("Jump to last recorded cycle", key="m6_latest", use_container_width=True):
                    st.session_state["m6_replay_cycle"] = cycle_max
            replay_cycle = st.slider("Recorded cycle / manual historical replay", min_value=cycle_min,
                                     max_value=cycle_max, step=1, key="m6_replay_cycle")
            visible = history_all.loc[history_all["cycle"] <= replay_cycle].copy()
            current = visible.iloc[-1]
            pred_column = "pred_xgboost" if monitor_model == "XGBoost" else "pred_random_forest"
            alternate = "pred_random_forest" if monitor_model == "XGBoost" else "pred_xgboost"
            pred = float(current[pred_column])

            section_title("18 /", "Interpret the engine estimate")
            urgent_col, watch_col = st.columns(2, gap="large")
            with urgent_col:
                urgent_threshold = st.slider("Priority-review threshold / estimated cycles",
                                             min_value=5, max_value=55, value=30, step=5,
                                             key="m6_urgent_band")
            with watch_col:
                watch_threshold = st.slider("Watch-band upper boundary / estimated cycles",
                                            min_value=60, max_value=120,
                                            value=65, step=5,
                                            key="m6_watch_band")
            band = health_band(pred, urgent_at=urgent_threshold, watch_at=watch_threshold)
            shown_pred = f"{pred:.1f}" + (" (capped)" if m6_meta.get("target_cap") == 125 and pred >= 124.99 else "")
            current_last = bool(replay_cycle == cycle_max)
            m61, m62, m63, m64 = st.columns(4, gap="medium")
            with m61:
                metric_card("Estimated RUL", shown_pred, "CYCLES / MODEL ESTIMATE")
            with m62:
                metric_card("Replay position", f"{replay_cycle:,}", "LAST AVAILABLE OBSERVATION")
            with m63:
                metric_card("Sensor reading", f"{float(current[monitor_sensor]):,.2f}", monitor_sensor.upper())
            with m64:
                metric_card("Other model's RUL", f"{float(current[alternate]):.1f}", "COMPARISON / CYCLES")
            st.markdown(
                '<div class="ap-m6-status ' + band["tone"] + '"><b>ESTIMATED PRIORITY:</b> '
                + escape(band["label"]) + ' &nbsp;·&nbsp; ' + escape(band["note"]) + '</div>',
                unsafe_allow_html=True,
            )
            if m6_meta.get("target_cap") == 125:
                st.caption("The model was trained on RUL targets clipped to 125 cycles. A displayed estimate near 125 "
                           "is a capped model value, not evidence that exactly 125 cycles remain.")
            st.caption("The priority bands above are adjustable demonstration thresholds. Their labels "
                       "are not calibrated failure likelihoods and should not guide aircraft maintenance.")

            section_title("19 /", "Telemetry and model evolution — up to selected cycle")
            sensor_col, forecast_col = st.columns([1.13, 1], gap="large")
            with sensor_col:
                sensor_plot = go.Figure()
                sensor_plot.add_trace(go.Scatter(x=visible["cycle"], y=visible[monitor_sensor],
                                                 name="Observed sensor", mode="lines",
                                                 line=dict(color=LILAC, width=2.3)))
                sensor_plot.add_trace(go.Scatter(
                    x=visible["cycle"], y=visible[monitor_sensor].rolling(5, min_periods=1).mean(),
                    name="Past 5-cycle mean", mode="lines", line=dict(color=ORANGE, width=3)))
                sensor_plot.update_xaxes(title="Recorded cycle")
                sensor_plot.update_yaxes(title=f"{monitor_sensor} / raw units")
                st.plotly_chart(tune_plot(sensor_plot, height=385), use_container_width=True)
            with forecast_col:
                estimate_plot = go.Figure()
                estimate_plot.add_trace(go.Scatter(
                    x=visible["cycle"], y=visible[pred_column], mode="lines",
                    name=f"{monitor_model} estimate", line=dict(color=ORANGE, width=3)))
                estimate_plot.add_trace(go.Scatter(
                    x=visible["cycle"], y=visible[alternate], mode="lines",
                    name="Other model", line=dict(color=LILAC, width=2, dash="dot")))
                estimate_plot.update_xaxes(title="Recorded cycle")
                estimate_plot.update_yaxes(title="Estimated RUL / cycles")
                st.plotly_chart(tune_plot(estimate_plot, height=385), use_container_width=True)
            st.caption("Only readings and previously computed predictions through the selected cycle are plotted. "
                       "Moving the slider forward reveals additional historical observations; it is not a live feed.")

            with st.expander("Reveal historical ground truth (validation only)", expanded=False):
                reveal_truth = st.toggle("Show actual RUL for comparison", value=False, key="m6_show_truth")
                if reveal_truth:
                    truth_val = float(current["actual_rul"])
                    truth_error = pred - truth_val
                    stat_a, stat_b = st.columns(2)
                    with stat_a:
                        metric_card("Known validation RUL", f"{truth_val:.1f}", "GROUND TRUTH / NOT A MODEL FEATURE")
                    with stat_b:
                        metric_card("Current prediction error", f"{truth_error:+.1f}", "ESTIMATE MINUS VALIDATION TRUTH")
                    truth_fig = go.Figure()
                    truth_fig.add_trace(go.Scatter(x=visible["cycle"], y=visible["actual_rul"],
                                                   name="Known historical RUL", line=dict(color=LILAC, width=2.6)))
                    truth_fig.add_trace(go.Scatter(x=visible["cycle"], y=visible[pred_column],
                                                   name="Model RUL estimate", line=dict(color=ORANGE, width=2.6)))
                    truth_fig.update_xaxes(title="Recorded cycle")
                    truth_fig.update_yaxes(title="RUL / cycles")
                    st.plotly_chart(tune_plot(truth_fig, height=335), use_container_width=True)
                    st.caption("These are validation labels from NASA complete training histories, not official test labels.")

            st.markdown("**Recorded observations currently visible**")
            show_cols = ["engine_id", "cycle", *MONITORED_SENSORS,
                         "pred_random_forest", "pred_xgboost"]
            st.dataframe(visible[show_cols].tail(30).round(3), hide_index=True, use_container_width=True)
            st.download_button("Download visible replay (CSV)",
                               data=visible[show_cols].to_csv(index=False),
                               file_name=f"aeropredict_validation_engine_{monitor_engine:03d}_through_{replay_cycle}.csv",
                               mime="text/csv", key="m6_csv")
            st.markdown(
                '<div class="ap-quality-note"><b>Scientific and operational boundaries:</b> '
                'Validation engines were not used to fit these optimized models, but their results were '
                'already inspected during Milestones 4–5. There is no new generalization benchmark here. '
                'The alert bands use a regression estimate, not calibrated failure probabilities. '
                'Do not use this prototype for real maintenance or safety decisions. '
                '<b>Official NASA test scores are reported separately in the Milestone 7 tab.</b></div>',
                unsafe_allow_html=True,
            )

if selected_view == "Final Evaluation":
    section_title("21 /", "NASA FD001 · official final test evaluation")
    st.markdown(
        '<div class="ap-m6-banner">'
        '<div class="ap-m6-kicker">MILESTONE 07 / OFFICIAL NASA BENCHMARK</div>'
        '<div class="ap-m6-title">THE FINAL CHECK. ONE RESULT PER ENGINE.</div>'
        '<div class="ap-m6-description">Frozen models from Milestone 5 are evaluated on '
        'the final recorded observation of each official FD001 test engine. '
        'The ground-truth RUL labels are only consulted to measure the finished predictions. '
        'Model selection was made earlier using validation engines, not this official test.</div>'
        '</div>', unsafe_allow_html=True,
    )
    m7_root = Path(__file__).resolve().parent
    m7_summary_file = m7_root / "outputs" / "m7_official_test_summary.json"
    m7_metrics_file = m7_root / "outputs" / "m7_official_test_metrics.csv"
    m7_preds_file = m7_root / "outputs" / "m7_official_test_predictions.csv"
    if not all(p.is_file() for p in (m7_summary_file, m7_metrics_file, m7_preds_file)):
        st.info("Official evaluation has not been generated yet. Run the command below after your Milestone 5 models are ready.")
        st.code(r".\.venv\Scripts\python.exe evaluate_official_test.py", language="powershell")
        st.caption("Do this only after model tuning is finalized. This opens the reserved NASA official test labels.")
    else:
        try:
            m7_info = json.loads(m7_summary_file.read_text(encoding="utf-8"))
            m7_scores = pd.read_csv(m7_metrics_file)
            m7_preds = pd.read_csv(m7_preds_file).sort_values("engine_id").reset_index(drop=True)
            if m7_info.get("milestone") != 7 or not set([
                "engine_id", "cycle", "actual_rul_true", "actual_rul_capped_125",
                "pred_random_forest", "pred_xgboost"
            ]).issubset(m7_preds.columns):
                raise ValueError("Milestone 7 report metadata or prediction columns are invalid")
            if len(m7_preds) != int(m7_info["test_engine_count"]):
                raise ValueError("Milestone 7 test engine count mismatch")
            if set(m7_scores["evaluation_target"]) != {"capped_target", "true_uncapped_rul"}:
                raise ValueError("Missing capped/uncapped test score conventions")
        except (OSError, KeyError, ValueError, pd.errors.ParserError, json.JSONDecodeError) as ex:
            st.error(f"Unable to load Milestone 7 reports: {ex}. Run evaluate_official_test.py again.")
        else:
            st.markdown(
                '<div class="ap-model-callout"><b>Important:</b> The headline test evaluation is '
                'on real, uncapped remaining cycles. Models trained on a 125-cycle clipped target '
                'cannot distinguish longer horizons accurately. The capped-target comparison '
                'is reported separately. These scores are a final benchmark snapshot; '
                'do not adjust models based on this result.</div>', unsafe_allow_html=True,
            )
            section_title("22 /", "Read the official test score")
            first_sel, second_sel = st.columns(2, gap="large")
            default_family = m7_info["model_selected_before_official_test"]
            with first_sel:
                final_family = st.selectbox(
                    "Inspect model (selection was made before test)",
                    ["XGBoost", "Random Forest"],
                    index=0 if default_family == "XGBoost" else 1,
                    key="m7_family",
                )
            with second_sel:
                eval_mode = st.radio(
                    "Scoring convention",
                    options=["True physical RUL (uncapped)", "125-cycle capped reference"],
                    horizontal=True, key="m7_mode",
                )
            metric_target = "true_uncapped_rul" if eval_mode.startswith("True") else "capped_target"
            m7_subset = m7_scores.loc[
                (m7_scores["model"] == final_family) &
                (m7_scores["evaluation_target"] == metric_target)
            ]
            if len(m7_subset) != 1:
                st.error("Metric row missing for this model and scoring convention")
            else:
                met_row = m7_subset.iloc[0]
                mc1, mc2, mc3, mc4 = st.columns(4, gap="medium")
                with mc1:
                    metric_card("Official test MAE", f"{float(met_row['mae']):.2f}", "CYCLES · LOWER IS BETTER")
                with mc2:
                    metric_card("Official test RMSE", f"{float(met_row['rmse']):.2f}", "CYCLES · LOWER IS BETTER")
                with mc3:
                    metric_card("Official test R²", f"{float(met_row['r2']):.3f}", "ENGINE-LEVEL SCORE")
                with mc4:
                    metric_card("Official test engines", str(len(m7_preds)), "ONE LAST-CYCLE PREDICTION EACH")

                st.markdown(
                    f'<div class="ap-quality-note"><b>Preselected model:</b> '
                    f'{escape(default_family)} &nbsp;·&nbsp; '
                    f'<b>Training target:</b> {escape(str(m7_info["model_training_target"]))} '
                    f'&nbsp;·&nbsp; <b>True RUL above 125:</b> '
                    f'{int(m7_info["true_rul_exceeds_cap_count"])} engines. '
                    f'Both models are presented for transparency; the official test is NOT used '
                    f'to choose a new winner.</div>', unsafe_allow_html=True,
                )
                st.markdown("**Model comparison on the selected scoring convention**")
                model_compare = (m7_scores.loc[m7_scores["evaluation_target"] == metric_target,
                                              ["model", "mae", "rmse", "r2", "test_engines"]]
                                 .rename(columns={"model": "Model", "mae": "MAE / cycles",
                                                  "rmse": "RMSE / cycles", "r2": "R²", "test_engines": "Engines"}))
                st.dataframe(model_compare, hide_index=True, use_container_width=True)
                st.caption("Do not interpret whichever test model has the lower RMSE as a newly selected model. "
                           "The declared selection rule was fixed during Milestone 5.")

                section_title("23 /", "Predictions versus known final RUL")
                pcol = "pred_xgboost" if final_family == "XGBoost" else "pred_random_forest"
                truecol = "actual_rul_true" if metric_target == "true_uncapped_rul" else "actual_rul_capped_125"
                comparison_left, comparison_right = st.columns([1.12, 1], gap="large")
                with comparison_left:
                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(
                        x=m7_preds[truecol], y=m7_preds[pcol],
                        mode="markers", name="Engine last-cycle estimate",
                        marker=dict(color=ORANGE, size=9, opacity=.8, line=dict(color="#FFBEA2", width=.5)),
                        customdata=m7_preds[["engine_id", "cycle"]].to_numpy(),
                        hovertemplate="ENG-%{customdata[0]:03d} / cycle %{customdata[1]}<br>Actual %{x:.1f}<br>Predicted %{y:.1f}<extra></extra>",
                    ))
                    span = max(125, float(m7_preds[[truecol, pcol]].max().max()))
                    fig_scatter.add_trace(go.Scatter(
                        x=[0, span], y=[0, span], mode="lines", name="Perfect prediction",
                        line=dict(color=LILAC, dash="dash", width=1.7)))
                    fig_scatter.update_xaxes(title="Actual RUL / cycles", range=[-5, span + 5])
                    fig_scatter.update_yaxes(title="Predicted RUL / cycles", range=[-5, span + 5])
                    st.plotly_chart(tune_plot(fig_scatter, height=400), use_container_width=True)
                with comparison_right:
                    signed_error = m7_preds[pcol] - m7_preds[truecol]
                    fig_error = go.Figure()
                    fig_error.add_trace(go.Histogram(x=signed_error, nbinsx=20, name="Signed error",
                                                     marker_color=LILAC, opacity=.9))
                    fig_error.add_vline(x=0, line_color=ORANGE, line_dash="dash", line_width=2)
                    fig_error.update_xaxes(title="Estimate minus actual / cycles")
                    fig_error.update_yaxes(title="Number of test engines")
                    st.plotly_chart(tune_plot(fig_error, height=400), use_container_width=True)
                st.caption("Overestimates appear to the right of zero in the error histogram; "
                           "underestimates appear to the left. The scatter chart uses one engine per point.")

                section_title("24 /", "Investigate difficult test engines")
                plot_errors = m7_preds.copy()
                plot_errors["absolute_error"] = (plot_errors[pcol] - plot_errors[truecol]).abs()
                worst = plot_errors.sort_values("absolute_error", ascending=False).head(15)
                bar_fig = go.Figure()
                bar_fig.add_trace(go.Bar(
                    x=[f"ENG-{v:03d}" for v in worst["engine_id"].astype(int)],
                    y=worst["absolute_error"], marker_color=ORANGE, name="Absolute error"))
                bar_fig.update_xaxes(title="Test engine / 15 largest absolute errors")
                bar_fig.update_yaxes(title="Absolute error / cycles")
                st.plotly_chart(tune_plot(bar_fig, height=370), use_container_width=True)

                one, two = st.columns(2, gap="large")
                with one:
                    chosen_e = st.selectbox(
                        "Inspect official test engine",
                        m7_preds["engine_id"].astype(int).tolist(),
                        format_func=lambda eid: f"ENG-{eid:03d}", key="m7_engine",
                    )
                detail = m7_preds.loc[m7_preds["engine_id"] == chosen_e].iloc[0]
                with two:
                    st.markdown(f"**Final recorded cycle:** {int(detail['cycle'])}")
                    st.markdown(f"**True RUL:** {int(detail['actual_rul_true'])} cycles")
                    st.markdown(f"**{final_family} prediction:** {float(detail[pcol]):.1f} cycles")
                st.caption("The displayed true RUL is held-out NASA ground truth, never a model input.")
                with st.expander("View all 100 official test predictions", expanded=False):
                    show_frame = m7_preds[["engine_id", "cycle", "actual_rul_true", "actual_rul_capped_125",
                                          "pred_random_forest", "pred_xgboost"]].copy()
                    st.dataframe(show_frame.round(2), hide_index=True, use_container_width=True)
                    st.download_button("Download official test comparison (CSV)",
                                       data=show_frame.to_csv(index=False),
                                       file_name="aeropredict_fd001_official_test_evaluation.csv",
                                       mime="text/csv", key="m7_export")
                st.markdown(
                    '<div class="ap-quality-note"><b>Evaluation integrity:</b> '
                    'The models were frozen before test labels were opened. '
                    'The default model was chosen in Milestone 5 based on validation RMSE. '
                    'Every NASA test engine contributes exactly one final-cycle prediction. '
                    'RUL values above 125 are not resolved by capped-target models. '
                    'The official test is now considered used; further tuning to these results would '
                    'invalidate its untouched-holdout status. NASA C-MAPSS is simulated; '
                    'this app is not intended for real aviation maintenance.</div>', unsafe_allow_html=True,
                )

st.markdown(
    '<div class="ap-footer">AEROPREDICT AI  /  OBSIDIAN 2.0 &nbsp;·&nbsp; '
    'NASA C-MAPSS FD001 &nbsp;·&nbsp; RESEARCH DEMONSTRATION / FD001<br>'
    'Simulated historical engine data. Final holdout evaluated after model selection; not certified maintenance guidance.</div>',
    unsafe_allow_html=True,
)
