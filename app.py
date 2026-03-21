import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from utils import preprocess_dataset, handle_imbalance
from ml_utils import train_and_evaluate, get_feature_importance_df, detect_task_type, CLASSIFIERS, REGRESSORS
from report_utils import generate_pdf_report

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG & CUSTOM THEME
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CleanD AI",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
code, pre, .stCode {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Dark industrial background */
.stApp {
    background: #080c14;
    color: #e2e8f0;
}

/* Main header */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border: 1px solid #1d4ed8;
    border-radius: 4px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #1d4ed8, #06b6d4, #1d4ed8);
}
.main-header h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    color: #f8fafc;
    margin: 0;
    letter-spacing: -0.02em;
}
.main-header p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0.4rem 0 0 0;
    font-family: 'JetBrains Mono', monospace;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-left: 3px solid #1d4ed8;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    flex: 1;
}
.metric-card .label {
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'JetBrains Mono', monospace;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #f1f5f9;
    font-family: 'Syne', sans-serif;
}
.metric-card .delta {
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}
.delta-pos { color: #22c55e; }
.delta-neg { color: #ef4444; }

/* Section headers */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #1d4ed8;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.3rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.3rem;
    color: #f1f5f9;
    margin-bottom: 1rem;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 0.5rem;
}

/* Log entries */
.log-entry {
    background: #0d1117;
    border-left: 3px solid #22c55e;
    padding: 0.4rem 0.8rem;
    margin: 0.3rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #86efac;
    border-radius: 0 4px 4px 0;
}

/* Model cards */
.model-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    padding: 1.2rem;
    margin: 0.7rem 0;
}
.model-name {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    color: #38bdf8;
    margin-bottom: 0.6rem;
}
.model-metric {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 3px;
    padding: 0.2rem 0.6rem;
    margin: 0.2rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #e2e8f0;
}

/* Insight box */
.insight-box {
    background: linear-gradient(135deg, #0c1a3a, #0f2a4a);
    border: 1px solid #1d4ed8;
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #cbd5e1;
}
.insight-box .insight-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #60a5fa;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}

/* Diff compare */
.compare-better { color: #22c55e; font-weight: 700; }
.compare-worse  { color: #ef4444; font-weight: 700; }
.compare-same   { color: #94a3b8; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0a0f1e;
    border-right: 1px solid #1e293b;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    color: #64748b !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
}

/* Buttons */
.stButton > button {
    background: #1d4ed8;
    color: white;
    border: none;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.4rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #2563eb;
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(29, 78, 216, 0.4);
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e293b;
    border-radius: 4px;
}

/* Plotly charts transparent bg */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
defaults = {
    "uploaded_df": None, "df_processed": None, "target_col": None,
    "preprocess_log": [], "ml_results": None, "ai_insights": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>⚡ CleanD AI</h1>
  <p>// smart dataset CleanD AI + preprocessor + ml evaluator for data scientists</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SIDEBAR — FILE UPLOAD + SETTINGS
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded_file:
        df_new = pd.read_csv(uploaded_file)
        df_new = df_new.apply(pd.to_numeric, errors='ignore')
        if st.session_state.uploaded_df is None or uploaded_file.name != st.session_state.get("fname", ""):
            st.session_state.uploaded_df = df_new
            st.session_state.df_processed = None
            st.session_state.target_col = None
            st.session_state.preprocess_log = []
            st.session_state.ml_results = None
            st.session_state.ai_insights = None
            st.session_state["fname"] = uploaded_file.name

    st.markdown("---")
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        st.markdown("### 🎯 Global Target Column")
        target_col = st.selectbox(
            "Select target", ["— None —"] + list(df.columns),
            label_visibility="collapsed", key="global_target"
        )
        st.session_state.target_col = None if target_col == "— None —" else target_col

        st.markdown("---")
        st.markdown("### ⚙️ Quick Stats")
        st.markdown(f"**Rows:** {df.shape[0]:,}")
        st.markdown(f"**Cols:** {df.shape[1]}")
        st.markdown(f"**Missing:** {int(df.isna().sum().sum()):,}")
        st.markdown(f"**Numerics:** {len(df.select_dtypes(include=np.number).columns)}")
        st.markdown(f"**Categoricals:** {len(df.select_dtypes(include='object').columns)}")

        if st.session_state.df_processed is not None:
            st.markdown("---")
            st.markdown("### 📥 Export")
            csv = st.session_state.df_processed.to_csv(index=False).encode()
            st.download_button("⬇ Processed CSV", csv, "processed.csv", "text/csv")

            if st.session_state.preprocess_log:
                try:
                    pdf_bytes = generate_pdf_report(
                        df_original=st.session_state.uploaded_df,
                        df_processed=st.session_state.df_processed,
                        log=st.session_state.preprocess_log,
                        ml_results=st.session_state.ml_results,
                        dataset_name=st.session_state.get("fname", "Dataset")
                    )
                    st.download_button("⬇ PDF Report", pdf_bytes, "cleand_report.pdf", "application/pdf")
                except Exception as e:
                    st.warning(f"PDF unavailable: {e}")

# ─────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────
if st.session_state.uploaded_df is None:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #475569;">
        <div style="font-size:4rem; margin-bottom:1rem;">📂</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.4rem; font-weight:700; color:#94a3b8;">
            Upload a CSV to begin
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; margin-top:0.5rem;">
            // use the sidebar to upload your dataset
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = st.session_state.uploaded_df
target_col = st.session_state.target_col

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  CleanD AI", "⚙️  Preprocess", "🔬  Compare", "🤖  ML Models", "💡  AI Insights"
])

PLOT_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(8,12,20,0.8)",
    font=dict(family="JetBrains Mono", color="#94a3b8", size=11),
    margin=dict(l=40, r=20, t=40, b=40),
)

# ═══════════════════════════════════════════════════════════════
# TAB 1 — PROFILE REPORT
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">// overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Dataset CleanD AI</div>', unsafe_allow_html=True)

    # Metric cards
    num_missing = int(df.isna().sum().sum())
    num_dupes = int(df.duplicated().sum())
    mem_kb = round(df.memory_usage(deep=True).sum() / 1024, 2)
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Rows</div>
            <div class="value">{df.shape[0]:,}</div>
        </div>
        <div class="metric-card">
            <div class="label">Columns</div>
            <div class="value">{df.shape[1]}</div>
        </div>
        <div class="metric-card">
            <div class="label">Missing Values</div>
            <div class="value">{num_missing:,}</div>
            <div class="delta {'delta-neg' if num_missing > 0 else 'delta-pos'}">
                {round(num_missing / (df.shape[0]*df.shape[1])*100, 1)}% of cells
            </div>
        </div>
        <div class="metric-card">
            <div class="label">Duplicates</div>
            <div class="value">{num_dupes}</div>
        </div>
        <div class="metric-card">
            <div class="label">Memory</div>
            <div class="value">{mem_kb}</div>
            <div class="delta compare-same">KB</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Column details table
    st.markdown('<div class="section-label">// column details</div>', unsafe_allow_html=True)
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.values,
        "Non-Null": df.notnull().sum().values,
        "Missing": df.isnull().sum().values,
        "Missing %": (df.isnull().sum().values / len(df) * 100).round(1),
        "Unique": [df[c].nunique() for c in df.columns],
        "Sample": [str(df[c].dropna().iloc[0]) if len(df[c].dropna()) > 0 else "—" for c in df.columns]
    })
    st.dataframe(info_df, use_container_width=True, hide_index=True)
    st.markdown("---")

    # Summary stats
    st.markdown('<div class="section-label">// summary statistics</div>', unsafe_allow_html=True)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe().T.round(4), use_container_width=True)
    st.markdown("---")

    # Column visualizations — Plotly
    st.markdown('<div class="section-label">// distributions</div>', unsafe_allow_html=True)
    cols_vis = st.columns(2)
    for i, col in enumerate(df.columns[:16]):
        with cols_vis[i % 2]:
            try:
                if df[col].dtype in [np.int64, np.float64, float, int]:
                    fig = px.histogram(df, x=col, nbins=30, title=col,
                                       color_discrete_sequence=["#1d4ed8"])
                    fig.update_layout(**PLOT_THEME, height=260, title_font_size=12)
                    fig.update_traces(marker_line_width=0)
                else:
                    vc = df[col].value_counts().head(12).reset_index()
                    vc.columns = ["value", "count"]
                    fig = px.bar(vc, x="count", y="value", orientation="h",
                                 title=col, color_discrete_sequence=["#0891b2"])
                    fig.update_layout(**PLOT_THEME, height=260, title_font_size=12)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            except Exception:
                st.caption(f"Cannot plot {col}")

    if len(df.columns) > 16:
        st.info(f"Showing first 16 of {len(df.columns)} columns.")
    st.markdown("---")

    # Correlation heatmap
    st.markdown('<div class="section-label">// correlation matrix</div>', unsafe_allow_html=True)
    df_enc = df.copy()
    for c in df_enc.select_dtypes(include=['object','category']).columns:
        try:
            df_enc[c] = LabelEncoder().fit_transform(df_enc[c].astype(str))
        except Exception:
            df_enc.drop(columns=[c], inplace=True)

    num_enc = df_enc.select_dtypes(include=np.number).columns.tolist()
    if len(num_enc) >= 2:
        corr_mat = df_enc[num_enc].corr()
        fig = px.imshow(corr_mat, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Feature Correlation Matrix", aspect="auto")
        fig.update_layout(**PLOT_THEME, height=500)
        st.plotly_chart(fig, use_container_width=True)

        if target_col and target_col in df_enc.columns:
            tc = corr_mat[target_col].drop(target_col, errors='ignore').abs().sort_values(ascending=True)
            fig2 = px.bar(x=tc.values, y=tc.index, orientation="h",
                          title=f"Feature Correlation with '{target_col}'",
                          color=tc.values, color_continuous_scale="Blues")
            fig2.update_layout(**PLOT_THEME, height=max(300, len(tc)*25), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation.")

    # Missing value heatmap
    if num_missing > 0:
        st.markdown("---")
        st.markdown('<div class="section-label">// missing value map</div>', unsafe_allow_html=True)
        miss_df = df.isnull().astype(int)
        fig = px.imshow(miss_df.T, color_continuous_scale=["#0f172a", "#ef4444"],
                        title="Missing Value Map (red = missing)", aspect="auto")
        fig.update_layout(**PLOT_THEME, height=max(200, len(df.columns)*18))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">// preprocessing pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Configure & Run</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**🗑️ Drop Columns**")
        drop_cols = st.multiselect("Columns to drop", options=df.columns, key="pp_drop")

        st.markdown("**📉 Low-Correlation Drop**")
        drop_low_corr = st.checkbox("Enable low-correlation filter", key="pp_low_corr")
        corr_threshold = 0.05
        if drop_low_corr:
            if not target_col:
                st.warning("Select a target column in the sidebar first.")
                drop_low_corr = False
            else:
                corr_threshold = st.slider("Drop threshold", 0.0, 0.5, 0.05, 0.01, key="pp_corr_thresh")

        st.markdown("**🔧 Missing Values**")
        handle_missing = st.selectbox("Strategy", ["None", "Drop Rows", "Fill Values", "KNN Impute", "Iterative Impute"], key="pp_miss")
        num_fill = None
        if handle_missing == "Fill Values":
            num_fill = st.radio("Numeric fill", ["Mean", "Median"], horizontal=True, key="pp_fill")

        st.markdown("**♻️ Duplicates**")
        remove_dupes = st.checkbox("Remove duplicate rows", value=True, key="pp_dupes")

    with c2:
        st.markdown("**🚫 Outlier Handling**")
        outlier_method = st.selectbox("Method", ["None", "IQR Method", "Z-Score Method", "Isolation Forest", "Winsorization"], key="pp_out")

        st.markdown("**📊 Normalization**")
        normalization = st.selectbox("Technique",
            ["None", "Min-Max Scaler", "Standard Scaler", "Robust Scaler", "Z-Score (Custom)", "Log Transform"],
            key="pp_norm")

        st.markdown("**🔡 Encoding**")
        encoding_type = st.selectbox("Technique",
            ["None", "Label Encoding", "One-Hot Encoding", "Ordinal Encoding"],
            key="pp_enc")

    st.markdown("---")
    if st.button("⚡ Run Preprocessing Pipeline", use_container_width=True, key="run_pp"):
        progress = st.progress(0)
        status = st.empty()

        def update(step, total):
            progress.progress(step / total)

        with st.spinner("Running pipeline..."):
            df_proc, log = preprocess_dataset(
                df=df,
                drop_cols=drop_cols,
                handle_missing=handle_missing,
                num_fill=num_fill,
                remove_duplicates=remove_dupes,
                outlier_method=outlier_method,
                encoding_type=encoding_type,
                normalization=normalization,
                target_col=target_col,
                drop_low_corr=drop_low_corr,
                corr_threshold=corr_threshold,
                progress_callback=update
            )
            time.sleep(0.3)
            st.session_state.df_processed = df_proc
            st.session_state.preprocess_log = log
            st.session_state.ml_results = None
            st.session_state.ai_insights = None

        progress.progress(1.0)
        st.success("✅ Pipeline complete!")

        st.markdown('<div class="section-label">// pipeline log</div>', unsafe_allow_html=True)
        for entry in log:
            st.markdown(f'<div class="log-entry">▸ {entry}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-label">// processed preview</div>', unsafe_allow_html=True)
        st.dataframe(df_proc.head(20), use_container_width=True, hide_index=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df_proc.shape[0], delta=df_proc.shape[0] - df.shape[0])
        col2.metric("Columns", df_proc.shape[1], delta=df_proc.shape[1] - df.shape[1])
        col3.metric("Missing", int(df_proc.isna().sum().sum()),
                    delta=-(int(df.isna().sum().sum()) - int(df_proc.isna().sum().sum())))


# ═══════════════════════════════════════════════════════════════
# TAB 3 — BEFORE / AFTER COMPARISON
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">// dataset comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Before vs. After Preprocessing</div>', unsafe_allow_html=True)

    if st.session_state.df_processed is None:
        st.info("Run the preprocessing pipeline first (Tab 2).")
    else:
        df_proc = st.session_state.df_processed

        # Summary comparison table
        def quick_stats(d):
            return {
                "Rows": d.shape[0],
                "Columns": d.shape[1],
                "Missing Values": int(d.isna().sum().sum()),
                "Duplicates": int(d.duplicated().sum()),
                "Numeric Cols": len(d.select_dtypes(include=np.number).columns),
                "Categorical Cols": len(d.select_dtypes(include='object').columns),
                "Memory (KB)": round(d.memory_usage(deep=True).sum() / 1024, 2),
            }

        s_orig = quick_stats(df)
        s_proc = quick_stats(df_proc)

        compare_rows = []
        for key in s_orig:
            orig_v, proc_v = s_orig[key], s_proc[key]
            diff = proc_v - orig_v if isinstance(orig_v, (int, float)) else "—"
            if key == "Missing Values":
                tag = "compare-better" if diff <= 0 else "compare-worse"
            elif key in ("Rows",):
                tag = "compare-same"
            else:
                tag = "compare-same"
            compare_rows.append({"Metric": key, "Original": orig_v, "Processed": proc_v, "Δ": diff})

        compare_df = pd.DataFrame(compare_rows)
        st.dataframe(compare_df, use_container_width=True, hide_index=True)
        st.markdown("---")

        # Distribution comparison for numeric columns
        st.markdown('<div class="section-label">// distribution comparison</div>', unsafe_allow_html=True)
        common_num = [c for c in df_proc.select_dtypes(include=np.number).columns if c in df.columns]

        if common_num:
            sel_col = st.selectbox("Select column to compare", common_num, key="compare_col")
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["Before Preprocessing", "After Preprocessing"])
            fig.add_trace(go.Histogram(x=df[sel_col].dropna(), name="Before",
                                       marker_color="#ef4444", opacity=0.75,
                                       nbinsx=30), row=1, col=1)
            fig.add_trace(go.Histogram(x=df_proc[sel_col].dropna(), name="After",
                                       marker_color="#22c55e", opacity=0.75,
                                       nbinsx=30), row=1, col=2)
            fig.update_layout(**PLOT_THEME, height=350, showlegend=False)
            fig.update_xaxes(title_text=sel_col)
            st.plotly_chart(fig, use_container_width=True)

            # Box plot comparison
            try:
                fig2 = go.Figure()
                fig2.add_trace(go.Box(y=df[sel_col].dropna(), name="Before",
                                      marker_color="#ef4444", boxmean=True))
                fig2.add_trace(go.Box(y=df_proc[sel_col].dropna(), name="After",
                                      marker_color="#22c55e", boxmean=True))
                fig2.update_layout(**PLOT_THEME, height=320,
                                   title=f"Box Plot — {sel_col}")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                pass

        # Imbalance handling section
        st.markdown("---")
        st.markdown('<div class="section-label">// class balance</div>', unsafe_allow_html=True)
        if target_col and target_col in df_proc.columns:
            before_dist = df[target_col].value_counts() if target_col in df.columns else None
            after_dist = df_proc[target_col].value_counts()

            fig_bal = make_subplots(rows=1, cols=2,
                                    subplot_titles=["Class Distribution — Before", "Class Distribution — After"])
            if before_dist is not None:
                fig_bal.add_trace(go.Bar(x=before_dist.index.astype(str), y=before_dist.values,
                                         marker_color="#f59e0b", name="Before"), row=1, col=1)
            fig_bal.add_trace(go.Bar(x=after_dist.index.astype(str), y=after_dist.values,
                                     marker_color="#06b6d4", name="After"), row=1, col=2)
            fig_bal.update_layout(**PLOT_THEME, height=320, showlegend=False)
            st.plotly_chart(fig_bal, use_container_width=True)

            bal_method = st.selectbox("Apply class balancing",
                                      ["None", "SMOTE Oversampling", "Random Undersampling"],
                                      key="bal_method_tab3")
            if st.button("Apply Balancing", key="apply_bal_tab3"):
                df_bal, msg = handle_imbalance(df_proc, target_col, bal_method)
                if msg == "Success":
                    st.session_state.df_processed = df_bal
                    st.success(f"✅ {bal_method} applied. New shape: {df_bal.shape}")
                else:
                    st.error(f"❌ {msg}")
        else:
            st.info("Select a target column in the sidebar to see class distribution.")

        # PCA Section
        st.markdown("---")
        st.markdown('<div class="section-label">// dimensionality reduction</div>', unsafe_allow_html=True)
        if st.checkbox("Run PCA", key="pca_check"):
            pca_target = target_col
            num_pca_cols = [c for c in df_proc.select_dtypes(include=np.number).columns
                            if c != pca_target]
            if len(num_pca_cols) < 2:
                st.warning("Need at least 2 numeric feature columns for PCA.")
            else:
                n_comp = st.slider("Components", 2, min(len(num_pca_cols), 15), 2, key="pca_n")
                try:
                    pca_data = df_proc[num_pca_cols].dropna()
                    pca = PCA(n_components=n_comp)
                    pca_result = pca.fit_transform(pca_data)
                    pca_df = pd.DataFrame(pca_result, columns=[f"PC{i+1}" for i in range(n_comp)])

                    # Explained variance
                    var_df = pd.DataFrame({
                        "Component": [f"PC{i+1}" for i in range(n_comp)],
                        "Explained Variance %": np.round(pca.explained_variance_ratio_ * 100, 2),
                        "Cumulative %": np.round(np.cumsum(pca.explained_variance_ratio_) * 100, 2)
                    })

                    fig_var = go.Figure()
                    fig_var.add_trace(go.Bar(x=var_df["Component"], y=var_df["Explained Variance %"],
                                             name="Individual", marker_color="#1d4ed8"))
                    fig_var.add_trace(go.Scatter(x=var_df["Component"], y=var_df["Cumulative %"],
                                                 name="Cumulative", mode="lines+markers",
                                                 marker_color="#06b6d4", line=dict(width=2)))
                    fig_var.update_layout(**PLOT_THEME, height=320, title="PCA Explained Variance")
                    st.plotly_chart(fig_var, use_container_width=True)

                    # 2D scatter
                    scatter_kwargs = dict(x=pca_df["PC1"], y=pca_df["PC2"])
                    if pca_target and pca_target in df_proc.columns:
                        hue = df_proc.loc[pca_data.index, pca_target].reset_index(drop=True).astype(str)
                        fig_sc = px.scatter(x=pca_df["PC1"], y=pca_df["PC2"], color=hue,
                                            title="PCA — PC1 vs PC2", labels={"x":"PC1","y":"PC2"},
                                            color_discrete_sequence=px.colors.qualitative.Bold)
                    else:
                        fig_sc = px.scatter(x=pca_df["PC1"], y=pca_df["PC2"],
                                            title="PCA — PC1 vs PC2", labels={"x":"PC1","y":"PC2"})
                    fig_sc.update_layout(**PLOT_THEME, height=400)
                    st.plotly_chart(fig_sc, use_container_width=True)

                    # Download PCA
                    if pca_target and pca_target in df_proc.columns:
                        pca_df[pca_target] = df_proc.loc[pca_data.index, pca_target].values
                    csv_pca = pca_df.to_csv(index=False).encode()
                    st.download_button("📥 Download PCA Dataset", csv_pca, "pca_data.csv", "text/csv",
                                       key="dl_pca")
                except Exception as e:
                    st.error(f"PCA failed: {e}")


# ═══════════════════════════════════════════════════════════════
# TAB 4 — ML MODELS
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-label">// model training & evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AutoML Evaluator</div>', unsafe_allow_html=True)

    if st.session_state.df_processed is None:
        st.info("Run preprocessing first (Tab 2).")
    elif not target_col:
        st.info("Select a target column in the sidebar.")
    else:
        df_ml = st.session_state.df_processed
        task_type = detect_task_type(df_ml[target_col]) if target_col in df_ml.columns else "unknown"
        st.markdown(f'<div class="log-entry">▸ Detected task: <strong>{task_type.upper()}</strong> — target: <strong>{target_col}</strong></div>',
                    unsafe_allow_html=True)

        model_pool = CLASSIFIERS if task_type == "classification" else REGRESSORS
        c1, c2 = st.columns([2, 1])
        with c1:
            selected_models = st.multiselect(
                "Select models to train",
                options=list(model_pool.keys()),
                default=list(model_pool.keys())[:3],
                key="ml_model_select"
            )
        with c2:
            test_size = st.slider("Test split %", 10, 40, 20, 5, key="ml_test_size") / 100

        if st.button("🚀 Train & Evaluate Models", use_container_width=True, key="train_btn"):
            if not selected_models:
                st.warning("Select at least one model.")
            else:
                with st.spinner("Training models..."):
                    results, err = train_and_evaluate(df_ml, target_col, selected_models, test_size)
                if err:
                    st.error(err)
                else:
                    st.session_state.ml_results = results
                    st.success("✅ Training complete!")

        if st.session_state.ml_results:
            results = st.session_state.ml_results
            st.markdown("---")
            st.markdown('<div class="section-label">// model results</div>', unsafe_allow_html=True)

            # Collect metrics for comparison chart
            metric_rows = []
            for name, res in results.items():
                if "error" in res:
                    st.error(f"{name}: {res['error']}")
                    continue
                m = res["metrics"]
                row = {"Model": name}
                for k, v in m.items():
                    if k not in ("Report", "Confusion Matrix", "Feature Importance"):
                        row[k] = v
                metric_rows.append(row)

                # Model card
                metric_html = ""
                for k, v in m.items():
                    if k not in ("Report", "Confusion Matrix", "Feature Importance"):
                        metric_html += f'<span class="model-metric">{k}: {v}</span>'

                report_html = ""
                if "Report" in m:
                    report_html = f'<details><summary style="cursor:pointer; color:#64748b; font-family:JetBrains Mono; font-size:0.75rem; margin-top:0.5rem;">▸ Classification Report</summary><pre style="font-size:0.72rem; color:#94a3b8; background:#0d1117; padding:0.8rem; border-radius:4px; overflow-x:auto;">{m["Report"]}</pre></details>'

                st.markdown(f"""
                <div class="model-card">
                    <div class="model-name">◆ {name}</div>
                    {metric_html}
                    {report_html}
                </div>
                """, unsafe_allow_html=True)

            # Comparison bar chart
            if len(metric_rows) > 1:
                st.markdown("---")
                st.markdown('<div class="section-label">// model comparison</div>', unsafe_allow_html=True)
                metrics_df = pd.DataFrame(metric_rows)
                numeric_metrics = [c for c in metrics_df.columns if c != "Model" and metrics_df[c].apply(lambda x: isinstance(x, (int, float))).all()]
                if numeric_metrics:
                    sel_metric = st.selectbox("Metric to compare", numeric_metrics, key="compare_metric")
                    fig_cmp = px.bar(metrics_df, x="Model", y=sel_metric,
                                     color="Model", title=f"Model Comparison — {sel_metric}",
                                     color_discrete_sequence=px.colors.qualitative.Bold,
                                     text_auto=True)
                    fig_cmp.update_layout(**PLOT_THEME, height=360, showlegend=False)
                    fig_cmp.update_traces(textposition="outside")
                    st.plotly_chart(fig_cmp, use_container_width=True)

            # Feature importance
            st.markdown("---")
            st.markdown('<div class="section-label">// feature importance</div>', unsafe_allow_html=True)
            fi_df = get_feature_importance_df(results)
            if fi_df is not None:
                top_n = min(20, len(fi_df))
                fi_plot = fi_df["Average"].head(top_n).sort_values(ascending=True)
                fig_fi = px.bar(x=fi_plot.values, y=fi_plot.index, orientation="h",
                                title=f"Top {top_n} Features by Average Importance",
                                color=fi_plot.values,
                                color_continuous_scale="Blues", text_auto=True)
                fig_fi.update_layout(**PLOT_THEME, height=max(320, top_n * 28), showlegend=False)
                st.plotly_chart(fig_fi, use_container_width=True)

                # Per-model importance breakdown
                if len(fi_df.columns) > 2:
                    with st.expander("View per-model feature importance breakdown"):
                        model_cols = [c for c in fi_df.columns if c != "Average"]
                        fig_multi = go.Figure()
                        for mc in model_cols:
                            top_feats = fi_df[mc].dropna().head(10).sort_values(ascending=True)
                            fig_multi.add_trace(go.Bar(x=top_feats.values, y=top_feats.index,
                                                       orientation="h", name=mc))
                        fig_multi.update_layout(**PLOT_THEME, barmode="group", height=420,
                                               title="Feature Importance by Model")
                        st.plotly_chart(fig_multi, use_container_width=True)
            else:
                st.info("No feature importances available for selected models.")

            # Confusion matrix heatmaps (classification)
            for name, res in results.items():
                if "error" in res:
                    continue
                cm = res["metrics"].get("Confusion Matrix")
                if cm:
                    with st.expander(f"Confusion Matrix — {name}"):
                        fig_cm = px.imshow(cm, text_auto=True, title=f"{name} — Confusion Matrix",
                                           color_continuous_scale="Blues",
                                           labels=dict(x="Predicted", y="Actual"))
                        fig_cm.update_layout(**PLOT_THEME, height=380)
                        st.plotly_chart(fig_cm, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 5 — AI INSIGHTS
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-label">// ai-powered analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">AI Data Insights</div>', unsafe_allow_html=True)

    if st.session_state.uploaded_df is None:
        st.info("Upload a dataset first.")
    else:
        df_ai = st.session_state.df_processed if st.session_state.df_processed is not None else df

        # Build context summary for Claude
        def build_context(df_orig, df_clean, target, ml_res):
            num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
            cat_cols = df_clean.select_dtypes(include='object').columns.tolist()
            missing = df_clean.isna().sum().sum()
            dupes = df_clean.duplicated().sum()

            ctx = f"""Dataset: {st.session_state.get('fname','unknown')}
Shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} cols (original: {df_orig.shape[0]} × {df_orig.shape[1]})
Numeric columns ({len(num_cols)}): {', '.join(num_cols[:15])}{'...' if len(num_cols)>15 else ''}
Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:10])}{'...' if len(cat_cols)>10 else ''}
Missing values: {missing}
Duplicates: {dupes}
Target column: {target if target else 'None selected'}
"""
            if num_cols:
                stats = df_clean[num_cols[:8]].describe().round(3).to_string()
                ctx += f"\nSummary stats (first 8 numeric cols):\n{stats}\n"

            if ml_res:
                ctx += "\nML Results:\n"
                for name, res in ml_res.items():
                    if "error" not in res:
                        m = {k: v for k, v in res["metrics"].items()
                             if k not in ("Report", "Confusion Matrix", "Feature Importance")}
                        ctx += f"  {name}: {m}\n"
            return ctx

        context = build_context(df, df_ai, target_col, st.session_state.ml_results)

        st.markdown('<div class="section-label">// ask ai</div>', unsafe_allow_html=True)

        preset_questions = [
            "What are the most important insights from this dataset?",
            "What preprocessing steps do you recommend and why?",
            "Which features are most likely to be predictive of the target?",
            "Are there any data quality issues I should address?",
            "What ML algorithms would work best for this data?",
            "Summarize the dataset in plain English.",
        ]

        preset = st.selectbox("Quick questions", ["— custom —"] + preset_questions, key="ai_preset")
        user_q = st.text_area(
            "Your question",
            value="" if preset == "— custom —" else preset,
            height=80,
            placeholder="Ask anything about your dataset...",
            key="ai_question"
        )

        if st.button("💡 Get AI Insights", use_container_width=True, key="ai_btn"):
            if not user_q.strip():
                st.warning("Enter a question first.")
            else:
                try:
                    import anthropic
                    client = anthropic.Anthropic()

                    prompt = f"""You are a senior data scientist reviewing a dataset. Here is the dataset context:

{context}

User question: {user_q}

Provide a concise, actionable response focused on data science best practices. Use bullet points where helpful. Be specific to the numbers provided."""

                    with st.spinner("Analyzing..."):
                        response = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=1200,
                            messages=[{"role": "user", "content": prompt}]
                        )
                    answer = response.content[0].text
                    st.session_state.ai_insights = {"q": user_q, "a": answer}

                except Exception as e:
                    st.error(f"AI unavailable: {e}")
                    st.info("Make sure your ANTHROPIC_API_KEY environment variable is set.")

        if st.session_state.ai_insights:
            ins = st.session_state.ai_insights
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-header">◆ AI Response — "{ins['q'][:60]}..."</div>
                {ins['a'].replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)