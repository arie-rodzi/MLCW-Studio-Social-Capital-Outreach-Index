"""
MLCW Studio — Multi-LLM Consensus Weighting for the Social Capital Outreach Index.
Interactive companion to the paper. Run:  streamlit run app.py
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

import io
import mlcw_core as mc
import report as rpt


def template_bytes():
    """Excel template pre-filled with the real 53-economy OIC dataset + a sources sheet."""
    tmpl = pd.read_csv(DATA_PATH)
    src = pd.DataFrame({
        "column": ["iso", "economy", "cpi", "account_ownership", "internet_use"],
        "meaning": ["ISO code (optional)", "Economy / unit name",
                    "Transparency proxy: Corruption Perceptions Index 2024 (0-100, higher=cleaner)",
                    "Financial inclusion proxy: account ownership %, age 15+ (Global Findex 2021/2024)",
                    "Digital readiness proxy: internet users % (ITU/World Bank, 2025; optional)"],
        "source": ["-", "-", "Transparency International", "World Bank Global Findex",
                   "ITU / World Bank WDI"],
    })
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        tmpl.to_excel(xw, index=False, sheet_name="data")
        src.to_excel(xw, index=False, sheet_name="sources")
    return buf.getvalue()


def excel_report(ranking, weights, methods, diagnostics):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        ranking.to_excel(xw, index=False, sheet_name="Ranking")
        weights.to_excel(xw, index=False, sheet_name="Weights")
        methods.to_excel(xw, index=False, sheet_name="Method comparison")
        diagnostics.to_excel(xw, index=False, sheet_name="Diagnostics")
    return buf.getvalue()

st.set_page_config(page_title="MLCW Studio — Social Capital Outreach Index",
                   page_icon="🧭", layout="wide")

CRIT2 = ["Transparency", "Financial inclusion"]
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "oic_data.csv")


# ---------------- data ----------------
@st.cache_data
def load_default():
    return pd.read_csv(DATA_PATH)

if "df" not in st.session_state:
    st.session_state.df = load_default()
if "panel" not in st.session_state:
    st.session_state.panel = pd.DataFrame(mc.DEFAULT_PANEL, index=mc.CRITERIA_3).T
if "panel_source" not in st.session_state:
    st.session_state.panel_source = "Default 5-assessor panel (paper)"


# ---------------- sidebar ----------------
st.sidebar.title("🧭 MLCW Studio")
st.sidebar.caption("Social Capital Outreach Index for blockchain-enabled Islamic microfinance.")

df_all = st.session_state.df.copy()
has_digital = ("internet_use" in df_all.columns
               and pd.to_numeric(df_all["internet_use"], errors="coerce").notna().any())
opts = ["2-indicator core"] + (["3-indicator (with digital readiness)"] if has_digital else [])
model = st.sidebar.radio("Indicator set", opts)
use3 = model.startswith("3")
crit = mc.CRITERIA_3 if use3 else CRIT2
cols = ["cpi", "account_ownership"] + (["internet_use"] if use3 else [])

obj_method = st.sidebar.selectbox("Objective method", list(mc.OBJECTIVE_METHODS), index=1)
alpha = st.sidebar.slider("Fusion α  (0 = objective only · 1 = semantic only)", 0.0, 1.0, 0.0, 0.05)
lam = st.sidebar.slider("Disagreement penalty λ", 0.0, 15.0, 5.0, 0.5)
draws = st.sidebar.select_slider("Bootstrap draws", [200, 500, 1000, 2000, 5000], value=2000)

focal = st.sidebar.multiselect("Highlight economies",
                               df_all["economy"].tolist(),
                               default=[e for e in ["Malaysia", "Maldives"] if e in df_all["economy"].tolist()])

# build matrix for active model (complete cases)
work = df_all.dropna(subset=cols).copy()
work = work[(work[cols] != "").all(axis=1)]
for c in cols:
    work[c] = pd.to_numeric(work[c], errors="coerce")
work = work.dropna(subset=cols).reset_index(drop=True)
M = work[cols].to_numpy(float)
Z = mc.minmax(M)

# weights
obj_fn = mc.OBJECTIVE_METHODS[obj_method]
wobj = obj_fn(Z)
panel = st.session_state.panel.to_numpy(float)
sem = mc.semantic_weights(panel, lam=lam)
wsem_full = sem["wsem"]
wsem = wsem_full[:len(crit)]
wsem = wsem / wsem.sum()
wfuse = mc.fuse(wsem, wobj, alpha)

st.sidebar.markdown("---")
st.sidebar.caption(f"Active panel: {st.session_state.panel_source}")
st.sidebar.caption(f"K = {len(work)} economies · {len(crit)} indicators")


# ---------------- header ----------------
st.title("Social Capital Outreach Index")
st.markdown("Measure the social-capital conditions for **blockchain-enabled Islamic microfinance** "
            "outreach across OIC economies, weighted by **Multi-LLM Consensus Weighting (MLCW)**.")

tabs = st.tabs(["📊 Data", "⚖️ Weights", "🏆 Index & Ranking",
                "🔬 Sensitivity & Robustness", "🧪 Diagnostics", "📄 HTML Report",
                "🤖 Live elicitation"])


# ---------------- TAB: Data ----------------
with tabs[0]:
    st.subheader("Indicator data")
    source = st.radio("Data source", ["Built-in OIC dataset", "Upload your own (CSV / Excel)"],
                      horizontal=True)

    if source.startswith("Upload"):
        st.download_button("⬇ Download Excel template", template_bytes(),
                           "mlcw_template.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        up = st.file_uploader("Upload .xlsx, .xls or .csv", type=["xlsx", "xls", "csv"])
        if up is not None:
            try:
                raw = (pd.read_csv(up) if up.name.lower().endswith(".csv")
                       else pd.read_excel(up))
            except Exception as e:
                st.error(f"Could not read file: {e}")
                raw = None
            if raw is not None and len(raw.columns):
                st.caption("Preview of your file:")
                st.dataframe(raw.head(), width='stretch')
                cols = list(raw.columns)

                def guess(opts, *keys):
                    for k in keys:
                        for c in opts:
                            if k in str(c).lower():
                                return opts.index(c)
                    return 0
                st.markdown("**Map your columns to the index roles** (digital readiness is optional):")
                m1, m2 = st.columns(2)
                ecol = m1.selectbox("Unit / economy name", cols, index=guess(cols, "econ", "country", "name", "unit"))
                tcol = m2.selectbox("Transparency proxy (e.g. CPI)", cols, index=guess(cols, "cpi", "transp", "corrupt"))
                icol = m1.selectbox("Financial inclusion proxy", cols, index=guess(cols, "account", "inclus", "findex"))
                dopts = ["(none)"] + cols
                dcol = m2.selectbox("Digital readiness proxy (optional)", dopts,
                                    index=(dopts.index(cols[guess(cols, "internet", "digital", "net")]) if cols else 0))
                if st.button("Load this data", type="primary"):
                    new = pd.DataFrame({
                        "economy": raw[ecol].astype(str),
                        "cpi": pd.to_numeric(raw[tcol], errors="coerce"),
                        "account_ownership": pd.to_numeric(raw[icol], errors="coerce"),
                    })
                    if dcol != "(none)":
                        new["internet_use"] = pd.to_numeric(raw[dcol], errors="coerce")
                    new.insert(0, "iso", new["economy"].str[:3].str.upper())
                    new = new.dropna(subset=["cpi", "account_ownership"]).reset_index(drop=True)
                    if len(new) < 2:
                        st.error("Need at least 2 valid rows after mapping. Check your columns.")
                    else:
                        st.session_state.df = new
                        st.success(f"Loaded {len(new)} units. See the other tabs for results.")
                        st.rerun()

    st.markdown("---")
    st.caption("Edit cells, add or delete rows; downstream results update automatically. "
               "All indicators are benefit-type (higher = better).")
    edited = st.data_editor(st.session_state.df, num_rows="dynamic", width='stretch', key="editor")
    c1, c2 = st.columns(2)
    if c1.button("Apply edits"):
        st.session_state.df = edited.copy()
        st.success("Data updated.")
        st.rerun()
    if c2.button("Reset to built-in dataset"):
        load_default.clear()
        st.session_state.df = load_default()
        st.rerun()


# ---------------- TAB: Weights ----------------
with tabs[1]:
    st.subheader("Criterion weights")
    wdf = pd.DataFrame({
        "Criterion": crit,
        f"Objective ({obj_method})": np.round(wobj, 3),
        "Semantic (LLM panel)": np.round(wsem, 3),
        f"Fused (α={alpha:g})": np.round(wfuse, 3),
    })
    st.dataframe(wdf, width='stretch', hide_index=True)

    long = wdf.melt(id_vars="Criterion", var_name="Source", value_name="Weight")
    fig = px.bar(long, x="Criterion", y="Weight", color="Source", barmode="group",
                 color_discrete_sequence=["#1b7a7a", "#13315c", "#8a5a0a"])
    fig.update_layout(height=380, legend_title="", yaxis_range=[0, max(0.6, long.Weight.max() + 0.05)])
    st.plotly_chart(fig, width='stretch')

    if alpha == 0:
        st.info("α = 0: the headline uses the **objective** weights only (most conservative, fully "
                "reproducible). Raise α to inject the semantic panel; compare both in *Sensitivity*.")


# ---------------- TAB: Index & Ranking ----------------
with tabs[2]:
    st.subheader("Social Capital Outreach Index")
    scores = mc.sci(Z, wfuse)
    rk = mc.ranks(scores)
    weight_fn = lambda Zb: mc.fuse(wsem, obj_fn(Zb), alpha)
    lo, hi = mc.bootstrap_ci(M, weight_fn, draws=int(draws))

    res = work[["economy"]].copy()
    res["SCI"] = np.round(scores, 3)
    res["Rank"] = rk
    res["CI low"] = np.round(lo, 3)
    res["CI high"] = np.round(hi, 3)
    res = res.sort_values("Rank").reset_index(drop=True)

    fcol = res["economy"].isin(focal)
    topn = st.slider("Show top N", 5, len(res), min(20, len(res)))
    show = res.head(topn)
    colors = ["#8a5a0a" if e in focal else "#13315c" for e in show["economy"]]
    fig = go.Figure(go.Bar(
        x=show["SCI"], y=show["economy"], orientation="h", marker_color=colors,
        error_x=dict(type="data", symmetric=False,
                     array=show["CI high"] - show["SCI"],
                     arrayminus=show["SCI"] - show["CI low"], color="rgba(0,0,0,.35)")))
    fig.update_layout(height=22 * len(show) + 80, yaxis=dict(autorange="reversed"),
                      xaxis_title="Social Capital Index (95% bootstrap CI)", margin=dict(l=10, r=10))
    st.plotly_chart(fig, width='stretch')

    def hl(row):
        return ["background-color:#fdf2d8" if row["economy"] in focal else "" for _ in row]
    st.dataframe(res.style.apply(hl, axis=1), width='stretch', hide_index=True)

    # assemble multi-sheet Excel report
    wdf_x = pd.DataFrame({"Criterion": crit,
                          f"Objective ({obj_method})": np.round(wobj, 3),
                          "Semantic (LLM panel)": np.round(wsem, 3),
                          f"Fused (alpha={alpha:g})": np.round(wfuse, 3)})
    meth_x = {"Criterion": crit}
    for nm, fn in mc.OBJECTIVE_METHODS.items():
        meth_x[nm] = np.round(fn(Z), 3)
    meth_x = pd.DataFrame(meth_x)
    P = st.session_state.panel.to_numpy(float)
    diag_x = pd.DataFrame({"Diagnostic": ["Kendall W", "Cronbach alpha", "ICC(2,k)", "alpha (fusion)", "lambda", "K economies", "Indicators"],
                           "Value": [round(mc.kendall_w(P), 3), round(mc.cronbach_alpha(P), 3),
                                     round(mc.icc2k(P), 3), alpha, lam, len(work), len(crit)]})
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Download ranking (CSV)", res.to_csv(index=False).encode(),
                       "sci_ranking.csv", "text/csv")
    c2.download_button("⬇ Download full report (Excel)",
                       excel_report(res, wdf_x, meth_x, diag_x),
                       "mlcw_results.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ---------------- TAB: Sensitivity & Robustness ----------------
with tabs[3]:
    st.subheader("Fusion sweep (α)")
    A = np.linspace(0, 1, 21)
    foc_idx = [work.index[work["economy"] == e][0] for e in focal if e in work["economy"].values]
    rows = []
    for a in A:
        w = mc.fuse(wsem, wobj, a)
        rr = mc.ranks(mc.sci(Z, w))
        for e in focal:
            if e in work["economy"].values:
                i = work.index[work["economy"] == e][0]
                rows.append({"α": a, "economy": e, "rank": int(rr[i])})
    if rows:
        sw = pd.DataFrame(rows)
        fig = px.line(sw, x="α", y="rank", color="economy", markers=True,
                      color_discrete_sequence=px.colors.qualitative.Safe)
        fig.update_yaxes(autorange="reversed", title="Rank (1 = best)")
        fig.update_layout(height=360, legend_title="")
        st.plotly_chart(fig, width='stretch')
        st.caption("If the highlighted economies hold their rank across α, the conclusion is "
                   "robust to the semantic contribution.")

    st.subheader("Weighting-method comparison")
    rank_by_method = {}
    table = {"Criterion": crit}
    for name, fn in mc.OBJECTIVE_METHODS.items():
        w = fn(Z)
        table[name] = np.round(w, 3)
        rank_by_method[name] = mc.ranks(mc.sci(Z, w))
    st.dataframe(pd.DataFrame(table), width='stretch', hide_index=True)

    names = list(mc.OBJECTIVE_METHODS)
    S = np.eye(len(names))
    for i in range(len(names)):
        for j in range(len(names)):
            S[i, j] = mc.spearman(rank_by_method[names[i]], rank_by_method[names[j]])
    fig = px.imshow(np.round(S, 3), x=names, y=names, text_auto=True,
                    color_continuous_scale="Teal", zmin=0.9, zmax=1.0,
                    title="Pairwise Spearman of rankings")
    fig.update_layout(height=360, coloraxis_showscale=False)
    st.plotly_chart(fig, width='stretch')


# ---------------- TAB: Diagnostics ----------------
with tabs[4]:
    st.subheader("LLM-assessor panel")
    st.caption("Each row is an assessor's a-priori weight vector (sums to 1). "
               "Edit to explore; or replace via the Live elicitation tab.")
    ped = st.data_editor(st.session_state.panel, width='stretch', key="paneledit")
    if st.button("Apply panel edits"):
        st.session_state.panel = ped.copy()
        st.session_state.panel_source = "User-edited panel"
        st.rerun()

    P = st.session_state.panel.to_numpy(float)
    c1, c2, c3 = st.columns(3)
    c1.metric("Kendall's W", f"{mc.kendall_w(P):.3f}")
    c2.metric("Cronbach's α", f"{mc.cronbach_alpha(P):.3f}")
    c3.metric("ICC(2,k)", f"{mc.icc2k(P):.3f}")
    st.caption("W: concordance of criterion orderings · α: internal consistency · ICC(2,k): "
               "average-measures inter-assessor reliability.")

    sem_full = mc.semantic_weights(P, lam=lam)
    dd = pd.DataFrame({"Criterion": mc.CRITERIA_3,
                       "Consensus": np.round(sem_full["wbar"], 3),
                       "Dispersion σ": np.round(sem_full["sigma"], 3),
                       "Semantic w": np.round(sem_full["wsem"], 3)})
    st.dataframe(dd, width='stretch', hide_index=True)
    rel = pd.DataFrame({"Assessor": st.session_state.panel.index,
                        "Reliability r": np.round(sem_full["r"], 3)})
    st.dataframe(rel, width='stretch', hide_index=True)


# ---------------- TAB: HTML Report ----------------
with tabs[5]:
    st.subheader("Beautiful HTML report")
    st.caption("Generate a polished, standalone HTML report of the current results "
               "(interactive charts included). Open it in any browser or share it.")
    if st.button("Generate report", type="primary"):
        with st.spinner("Building report…"):
            sc = mc.sci(Z, wfuse)
            rkk = mc.ranks(sc)
            wfn = lambda Zb: mc.fuse(wsem, obj_fn(Zb), alpha)
            lo2, hi2 = mc.bootstrap_ci(M, wfn, draws=int(draws))
            rank_df = work[["economy"]].copy()
            rank_df["SCI"] = np.round(sc, 3)
            rank_df["Rank"] = rkk
            rank_df["CI low"] = np.round(lo2, 3)
            rank_df["CI high"] = np.round(hi2, 3)
            rank_df = rank_df.sort_values("Rank").reset_index(drop=True)

            wdf_r = pd.DataFrame({"Criterion": crit,
                                  f"Objective ({obj_method})": np.round(wobj, 3),
                                  "Semantic (LLM)": np.round(wsem, 3),
                                  f"Fused (α={alpha:g})": np.round(wfuse, 3)})
            meth_r = {"Criterion": crit}
            for nm, fn in mc.OBJECTIVE_METHODS.items():
                meth_r[nm] = np.round(fn(Z), 3)
            meth_r = pd.DataFrame(meth_r)
            Pr = st.session_state.panel.to_numpy(float)
            diag_r = {"W": mc.kendall_w(Pr), "alpha": mc.cronbach_alpha(Pr), "icc": mc.icc2k(Pr)}
            meta_r = {"K": len(work), "n_ind": len(crit), "alpha": alpha, "obj": obj_method}
            html = rpt.build_html_report(rank_df, wdf_r, meth_r, diag_r, meta_r, focal)
        st.success("Report ready.")
        st.download_button("⬇ Download HTML report", html.encode("utf-8"),
                           "SocialCapitalOutreach_report.html", "text/html")
        with st.expander("Preview"):
            components.html(html, height=680, scrolling=True)


# ---------------- TAB: Live elicitation ----------------
with tabs[6]:
    st.subheader("Live multi-vendor elicitation (optional)")
    st.caption("Run the fair protocol against real LLMs to replace the panel with genuine multi-vendor "
               "weights. Keys are used only in this session and never stored. Install the SDKs you need: "
               "`pip install anthropic openai google-generativeai`.")
    try:
        import llm_elicit as le
        with st.expander("API keys (session only)", expanded=True):
            for label, (env, _, _) in le.PROVIDERS.items():
                val = st.text_input(label, type="password", key="k_" + env,
                                    value=os.environ.get(env, ""))
                if val:
                    os.environ[env] = val
        avail = le.available_providers()
        st.write("Detected providers:", ", ".join(avail) if avail else "none (enter a key above)")
        sel = st.multiselect("Providers to query", list(le.PROVIDERS), default=avail)
        runs = st.number_input("Runs per provider", 1, 10, 1)
        temp = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        if st.button("Run elicitation", type="primary", disabled=not sel):
            try:
                with st.spinner("Querying models under the fair protocol…"):
                    labels, mat, rats = le.elicit(sel, runs=int(runs), temperature=temp)
                st.session_state.panel = pd.DataFrame(mat, index=labels, columns=mc.CRITERIA_3)
                st.session_state.panel_source = f"Live: {', '.join(sel)} ({len(labels)} runs)"
                st.success(f"Panel updated with {len(labels)} live assessor runs. "
                           "See Diagnostics and Weights tabs.")
                st.dataframe(st.session_state.panel.round(3), width='stretch')
            except Exception as e:
                st.error(f"Elicitation failed: {e}")
    except Exception:
        st.warning("Provider SDKs not installed. Add `anthropic`, `openai`, or `google-generativeai` "
                   "to requirements and redeploy to enable live elicitation.")

st.markdown("---")
st.caption("MLCW Studio · companion to the Social Capital Outreach Index manuscript. "
           "Headline results use real, fully traceable secondary data; no values are imputed.")
