import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="AI Learning Navigator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# CUSTOM CSS
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"], .stApp, body {
    background: linear-gradient(135deg, #0b1220, #111827);
    color: #e5e7eb;
}

.main {
    background: linear-gradient(135deg, #0b1220, #111827);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

h1, h2, h3 {
    color: #f9fafb;
    font-weight: 700;
    letter-spacing: -0.15px;
}

.dashboard-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.2rem;
}

.dashboard-subtitle {
    font-size: 1.05rem;
    color: #cbd5e1;
    margin-bottom: 1.5rem;
}

.hero-box {
    background: linear-gradient(135deg, #1e3a8a, #0f766e);
    border: 1px solid rgba(255,255,255,0.14);
    padding: 1.6rem 1.8rem;
    border-radius: 18px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.32);
}

.card {
    background: rgba(17, 24, 39, 0.82);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 1.2rem;
    border-radius: 18px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.20);
    margin-bottom: 1rem;
}

.viz-card {
    background: #f3f4f6;
    border: 1px solid rgba(255,255,255,0.12);
    padding: 0.7rem;
    border-radius: 12px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.20);
    margin-bottom: 0.5rem;
}

.kpi-card {
    background: linear-gradient(180deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
    border: 1px solid rgba(255,255,255,0.08);
    padding: 1rem 1.1rem;
    border-radius: 18px;
    text-align: left;
    box-shadow: 0 6px 18px rgba(0,0,0,0.22);
    min-height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.kpi-label {
    color: #9ca3af;
    font-size: 0.95rem;
    font-weight: 500;
}

.kpi-value {
    color: #f8fafc;
    font-size: 2rem;
    font-weight: 800;
    margin-top: 0.35rem;
}

.kpi-row {
    margin: 0.35rem 0 1rem 0;
}

.ai-tip {
    background: rgba(14, 165, 233, 0.16);
    border-left: 5px solid #38bdf8;
    padding: 1rem;
    border-radius: 12px;
    color: #e0f2fe;
    font-weight: 600;
    margin-top: 0.5rem;
}

.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 1rem;
}

.insight-good {
    background: rgba(6, 95, 70, 0.75);
    border-left: 5px solid #10b981;
    padding: 1rem;
    border-radius: 12px;
    color: #ecfdf5;
    font-weight: 600;
}

.insight-mid {
    background: rgba(146, 64, 14, 0.75);
    border-left: 5px solid #f59e0b;
    padding: 1rem;
    border-radius: 12px;
    color: #fffbeb;
    font-weight: 600;
}

.insight-bad {
    background: rgba(127, 29, 29, 0.78);
    border-left: 5px solid #ef4444;
    padding: 1rem;
    border-radius: 12px;
    color: #fef2f2;
    font-weight: 600;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f9fafb !important;
    font-weight: 600;
}

section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: #e2e8f0 !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #e5e7eb;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Metric style override */
div[data-testid="metric-container"] {
    background: linear-gradient(180deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
    border: 1px solid rgba(255,255,255,0.08);
    padding: 16px;
    border-radius: 18px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.22);
}

div[data-testid="metric-container"] label {
    color: #9ca3af !important;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.8rem;
    font-weight: 800;
}

div[data-testid="stMetricLabel"] {
    color: #cbd5e1 !important;
}

div[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #14b8a6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}

.stButton > button:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
    transition: 0.2s ease;
}

/* Top navigation (dashboard pages) */
.top-nav-title {
    font-size: 0.95rem;
    color: #cbd5e1;
    margin-bottom: 0.35rem;
    font-weight: 600;
}

div[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 0.5rem;
}

div[data-testid="stRadio"] div[role="radiogroup"] label {
    background: rgba(30, 41, 59, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 12px;
    padding: 0.4rem 0.85rem;
    transition: transform 0.2s ease, border-color 0.2s ease, background 0.2s ease;
    cursor: pointer;
}

div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    transform: translateY(-1px);
    border-color: rgba(56, 189, 248, 0.9);
    background: rgba(30, 64, 175, 0.35);
}

div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    border-color: #38bdf8;
    background: rgba(14, 165, 233, 0.22);
    box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.5) inset;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("student_learning_dataset.xlsx")

    df["test_preparation"] = df["test_preparation"].fillna("Not Completed").str.lower()
    df["avg_score"] = (df["math_score"] + df["reading_score"] + df["writing_score"]) / 3

    features = df[[
        "study_time_weekly",
        "absences",
        "math_score",
        "reading_score",
        "writing_score"
    ]]

    scaled_features = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(scaled_features)

    cluster_means = df.groupby("cluster")["avg_score"].mean().sort_values()
    cluster_labels = {
        cluster_means.index[0]: "Low Performer",
        cluster_means.index[1]: "Average (At Risk)",
        cluster_means.index[2]: "High Performer"
    }
    df["performance_level"] = df["cluster"].map(cluster_labels)

    df["skill_gap"] = np.select(
        [
            df["math_score"] < 50,
            (df["math_score"] >= 50) & (df["math_score"] < 75)
        ],
        ["Weak Fundamentals", "Needs Improvement"],
        default="Strong"
    )
    return df


df = load_data()

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.markdown("## 🎛 Dashboard Controls")

perf = st.sidebar.selectbox(
    "Performance Level",
    ["All", "Low Performer", "Average (At Risk)", "High Performer"]
)

gender = st.sidebar.selectbox(
    "Gender",
    ["All"] + list(df["gender"].dropna().unique())
)

score = st.sidebar.slider("Math Score Range", 0, 100, (0, 100))
abs_range = st.sidebar.slider("Absences Range", 0, int(df["absences"].max()), (0, int(df["absences"].max())))

df_f = df.copy()

if perf != "All":
    df_f = df_f[df_f["performance_level"] == perf]

if gender != "All":
    df_f = df_f[df_f["gender"] == gender]

df_f = df_f[
    (df_f["math_score"] >= score[0])
    & (df_f["math_score"] <= score[1])
    & (df_f["absences"] >= abs_range[0])
    & (df_f["absences"] <= abs_range[1])
]

avg = df_f["math_score"].mean() if not df_f.empty else 0


def ai_group(filtered_df):
    if filtered_df.empty:
        return "No data available for recommendation."

    avg_score = filtered_df["math_score"].mean()

    if avg_score > 85:
        return "🚀 High performance group - focus on advanced learning paths."
    if avg_score > 70:
        return "📈 Good progress - improve consistency with regular revision."
    if avg_score > 50:
        return "⚠️ Moderate risk - strengthen fundamentals and mentor support."
    return "❌ High risk - immediate intervention plan is recommended."


def ai_student(row):
    if row["math_score"] < 50:
        return "Focus on core basics"
    if row["absences"] > 8:
        return "Reduce absences and attend classes regularly"
    if row["math_score"] > 85:
        return "Try advanced problem-solving tasks"
    return "Maintain consistency with weekly practice"

# ---------------------------
# HEADER
# ---------------------------
st.markdown('<div class="dashboard-title">🎯 AI Learning Navigator</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">AI-Powered Student Performance Analytics and Personalization Dashboard</div>', unsafe_allow_html=True)

st.markdown('<div class="top-nav-title">Dashboard Navigation</div>', unsafe_allow_html=True)
page = st.radio(
    "Dashboard Navigation",
    ["Overview", "Performance", "Behavior", "Insights", "Student Tables"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("""
<div class="hero-box">
    <div style="font-size:2.1rem; font-weight:700; color:white; margin-bottom:0.4rem;">
        Smarter educational insights with data-driven segmentation
    </div>
    <div style="font-size:1.98rem; color:#cbd5e1; line-height:1.7;">
        Monitor academic performance, identify at-risk students, analyze behavior patterns,
        and generate actionable learning insights through clustering and visual analytics.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# KPI CARDS
# ---------------------------
st.markdown('<div class="kpi-row">', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

total_students = len(df_f)
avg_math_score = round(df_f["math_score"].mean(), 2) if not df_f.empty else 0
high_performers = (df_f["performance_level"] == "High Performer").sum()
at_risk_students = (df_f["performance_level"] == "Average (At Risk)").sum()

with col1:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Students</div>
            <div class="kpi-value">{total_students}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">Average Math Score</div>
            <div class="kpi-value">{avg_math_score}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">High Performers</div>
            <div class="kpi-value">{high_performers}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">At Risk Students</div>
            <div class="kpi-value">{at_risk_students}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------
# CHART STYLE
# ---------------------------
sns.set_theme(style="whitegrid")
PLOTLY_THEME = dict(
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    font=dict(color="#e2e8f0")
)

PLOTLY_CONFIG = dict(
    displayModeBar=True,
    scrollZoom=True,
    doubleClick="reset",
    responsive=True
)

COLORS = {
    "Low Performer": "#ef4444",
    "Average (At Risk)": "#f59e0b",
    "High Performer": "#22c55e"
}

palette = COLORS

def render_smart_insight():
    st.markdown('<div class="section-title">🧠 Smart Insight</div>', unsafe_allow_html=True)

    if df_f.empty:
        st.info("No students match the selected filters.")
        return

    if avg > 80:
        st.markdown('<div class="insight-good">Strong overall performance detected. Students are showing excellent academic outcomes.</div>', unsafe_allow_html=True)
    elif avg > 60:
        st.markdown('<div class="insight-mid">Moderate performance observed. Some student groups may need targeted academic support.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="insight-bad">Low performance trend detected. Immediate intervention is recommended for at-risk students.</div>', unsafe_allow_html=True)

def render_overview_page():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    render_smart_insight()
    st.markdown('<div class="section-title">🤖 AI Recommendation</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ai-tip">{ai_group(df_f)}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_performance_page():
    st.markdown('<div class="section-title">📊 Performance Overview</div>', unsafe_allow_html=True)

    if df_f.empty:
        st.info("No students match the selected filters.")
        return

    perf_counts = df_f["performance_level"].value_counts().reset_index()
    perf_counts.columns = ["performance_level", "count"]

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            perf_counts,
            x="performance_level",
            y="count",
            color="performance_level",
            color_discrete_map=COLORS,
            title="Student Distribution"
        )
        fig.update_layout(**PLOTLY_THEME, dragmode="pan", hovermode="closest")
        fig.update_traces(hovertemplate="Category: %{x}<br>Count: %{y}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with col2:
        fig = px.pie(
            perf_counts,
            names="performance_level",
            values="count",
            color="performance_level",
            color_discrete_map=COLORS,
            title="Performance Share"
        )
        fig.update_layout(**PLOTLY_THEME, hovermode="closest")
        fig.update_traces(hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

def render_behavior_page():
    st.markdown('<div class="section-title">📈 Behavior Analysis</div>', unsafe_allow_html=True)

    if df_f.empty:
        st.info("No students match the selected filters.")
        return

    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            df_f,
            x="study_time_weekly",
            y="math_score",
            color="performance_level",
            color_discrete_map=COLORS,
            title="Study Time vs Math Score"
        )
        fig.update_layout(**PLOTLY_THEME, dragmode="pan", hovermode="closest")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

    with col2:
        fig = px.scatter(
            df_f,
            x="absences",
            y="math_score",
            color="performance_level",
            color_discrete_map=COLORS,
            title="Absences vs Math Score"
        )
        fig.update_layout(**PLOTLY_THEME, dragmode="pan", hovermode="closest")
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

def render_insights_page():
    st.markdown('<div class="section-title">📉 Learning Insights</div>', unsafe_allow_html=True)

    if df_f.empty:
        st.info("No students match the selected filters.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Test Preparation Impact</div>', unsafe_allow_html=True)
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.boxplot(
            data=df_f,
            x="test_preparation",
            y="math_score",
            order=["completed", "not completed"],
            palette=["#8fb3d9", "#e4ac84"],
            width=0.58,
            ax=ax
        )
        ax.set_facecolor("#e8e8e8")
        fig.patch.set_facecolor("#f3f4f6")
        ax.set_title("")
        ax.set_xlabel("test_preparation", fontsize=10)
        ax.set_ylabel("math_score", fontsize=10)
        ax.grid(axis="y", color="#bdbdbd", linewidth=0.8, alpha=0.8)
        ax.grid(axis="x", visible=False)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
        st.markdown('<div class="viz-card">', unsafe_allow_html=True)
        corr_order = [
            "student_id",
            "age",
            "study_time_weekly",
            "absences",
            "math_score",
            "reading_score",
            "writing_score",
            "science_score",
            "cluster"
        ]
        available_corr_cols = [col for col in corr_order if col in df_f.columns]
        corr_df = df_f[available_corr_cols].corr(numeric_only=True) if available_corr_cols else df_f.select_dtypes(include="number").corr()

        fig, ax = plt.subplots(figsize=(8.3, 5.5))
        sns.heatmap(
            corr_df,
            annot=True,
            cmap="coolwarm",
            fmt=".3g",
            linewidths=0.3,
            linecolor="#d1d5db",
            cbar=True,
            ax=ax
        )
        ax.set_facecolor("#f3f4f6")
        fig.patch.set_facecolor("#f3f4f6")
        ax.set_title("")
        ax.tick_params(axis="x", rotation=90)
        ax.tick_params(axis="y", rotation=0)
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def render_tables_page():
    st.markdown('<div class="section-title">🏆 Student Tables</div>', unsafe_allow_html=True)

    temp = df_f.copy()
    if not temp.empty:
        temp["AI_Recommendation"] = temp.apply(ai_student, axis=1)
    else:
        temp["AI_Recommendation"] = []

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Personalized Recommendations")
    st.dataframe(
        temp[["gender", "study_time_weekly", "absences", "math_score", "performance_level", "skill_gap", "AI_Recommendation"]].head(20),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Top Performers")
    st.dataframe(df_f.sort_values(by="math_score", ascending=False).head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("At Risk Students")
    st.dataframe(df_f[df_f["performance_level"] == "Average (At Risk)"].head(10), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# PAGE ROUTING
# ---------------------------
if page == "Overview":
    render_overview_page()
elif page == "Performance":
    render_performance_page()
elif page == "Behavior":
    render_behavior_page()
elif page == "Insights":
    render_insights_page()
else:
    render_tables_page()

# ---------------------------
# FOOTER
# ---------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#9ca3af; font-size:.95rem;'>Developed by <b>Syed Azmath</b> & <b>Mohammed Afnan S</b></div>",
    unsafe_allow_html=True
)