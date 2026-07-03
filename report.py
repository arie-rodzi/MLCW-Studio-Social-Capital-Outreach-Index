"""
Beautiful standalone HTML report for the Social Capital Outreach Index.
Produces a self-contained, styled HTML string with interactive Plotly charts
(plotly.js loaded from CDN) that the user can download and open in any browser.
"""
from __future__ import annotations
import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go

NAVY = "#13315c"
TEAL = "#1b7a7a"
AMBER = "#b3801a"
INK = "#1a2230"
PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _fig_html(fig):
    return fig.to_html(full_html=False, include_plotlyjs=False,
                       config={"displayModeBar": False, "responsive": True})


def _rank_fig(ranking, focal, topn=20):
    show = ranking.head(topn).iloc[::-1]
    colors = [AMBER if e in focal else NAVY for e in show["economy"]]
    fig = go.Figure(go.Bar(
        x=show["SCI"], y=show["economy"], orientation="h", marker_color=colors,
        error_x=dict(type="data", symmetric=False,
                     array=show["CI high"] - show["SCI"],
                     arrayminus=show["SCI"] - show["CI low"], color="rgba(0,0,0,.28)")))
    fig.update_layout(height=26 * len(show) + 60, margin=dict(l=8, r=8, t=8, b=8),
                      xaxis_title="Social Capital Index (95% bootstrap CI)",
                      plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="Georgia, serif", color=INK, size=12))
    fig.update_xaxes(gridcolor="#eee")
    return fig


def _weights_fig(weights):
    long = weights.melt(id_vars="Criterion", var_name="Source", value_name="Weight")
    palette = {s: c for s, c in zip(weights.columns[1:], [TEAL, NAVY, AMBER])}
    fig = go.Figure()
    for s in weights.columns[1:]:
        d = long[long.Source == s]
        fig.add_bar(x=d["Criterion"], y=d["Weight"], name=s, marker_color=palette[s])
    fig.update_layout(barmode="group", height=340, margin=dict(l=8, r=8, t=8, b=8),
                      legend=dict(orientation="h", y=1.12, x=0),
                      plot_bgcolor="white", paper_bgcolor="white",
                      font=dict(family="Georgia, serif", color=INK, size=12))
    fig.update_yaxes(gridcolor="#eee", rangemode="tozero")
    return fig


def _table(df, focal=None, rank_col="Rank"):
    focal = focal or []
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        hl = " class='hl'" if (focal and r.get("economy") in focal) else ""
        cells = "".join(f"<td>{r[c]}</td>" for c in df.columns)
        rows.append(f"<tr{hl}>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _stat(label, value, sub=""):
    return (f"<div class='stat'><div class='stat-v'>{value}</div>"
            f"<div class='stat-l'>{label}</div>"
            f"{f'<div class=stat-s>{sub}</div>' if sub else ''}</div>")


def build_html_report(ranking, weights, methods, diagnostics, meta, focal):
    date = datetime.date.today().strftime("%d %B %Y")
    top = ranking.iloc[0]
    focal_bits = " · ".join(
        f"{e}: #{int(ranking.loc[ranking.economy == e, 'Rank'].iloc[0])}"
        for e in focal if e in ranking.economy.values)

    stats = "".join([
        _stat("Economies ranked", meta["K"]),
        _stat("Indicators", meta["n_ind"]),
        _stat("Top economy", top["economy"], f"SCI {top['SCI']:.3f}"),
        _stat("Fusion α", f"{meta['alpha']:g}", f"{meta['obj']} + LLM panel"),
    ])
    if focal_bits:
        stats += _stat("Highlighted", focal_bits.split(" · ")[0].split(":")[0] + " …", focal_bits)

    diag = "".join([
        _stat("Kendall's W", f"{diagnostics['W']:.3f}", "panel concordance"),
        _stat("Cronbach's α", f"{diagnostics['alpha']:.3f}", "internal consistency"),
        _stat("ICC(2,k)", f"{diagnostics['icc']:.3f}", "inter-assessor reliability"),
    ])

    rank_tbl = _table(ranking.head(25), focal=focal)
    weights_tbl = _table(weights.round(3))
    methods_tbl = _table(methods.round(3))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Social Capital Outreach Index — Report</title>
<script src="{PLOTLY_CDN}"></script>
<style>
  :root{{--navy:{NAVY};--teal:{TEAL};--amber:{AMBER};--ink:{INK};}}
  *{{box-sizing:border-box}}
  body{{margin:0;font-family:'Segoe UI',system-ui,-apple-system,sans-serif;color:var(--ink);
       background:#f5f7fa;line-height:1.55}}
  .wrap{{max-width:1000px;margin:0 auto;padding:0 20px 60px}}
  header.hero{{background:linear-gradient(135deg,#0e2445,#13315c 55%,#1b7a7a);color:#fff;
       padding:54px 20px 46px;text-align:center}}
  .hero h1{{font-family:Georgia,serif;font-size:30px;margin:0 auto 10px;max-width:820px;line-height:1.25}}
  .hero p{{margin:4px 0;opacity:.9;font-size:14px}}
  .hero .badge{{display:inline-block;margin-top:14px;padding:6px 14px;border:1px solid rgba(255,255,255,.4);
       border-radius:20px;font-size:12.5px;letter-spacing:.3px}}
  .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:-34px 0 8px}}
  .stat{{background:#fff;border-radius:14px;padding:16px 18px;box-shadow:0 6px 22px rgba(19,49,92,.08);
       border:1px solid #eef1f5}}
  .stat-v{{font-family:Georgia,serif;font-size:22px;font-weight:700;color:var(--navy)}}
  .stat-l{{font-size:12px;text-transform:uppercase;letter-spacing:.5px;color:#6b7686;margin-top:3px}}
  .stat-s{{font-size:12px;color:#93a;margin-top:2px;color:var(--teal)}}
  section.card{{background:#fff;border-radius:16px;padding:24px 26px;margin:22px 0;
       box-shadow:0 6px 22px rgba(19,49,92,.07);border:1px solid #eef1f5}}
  h2{{font-family:Georgia,serif;color:var(--navy);font-size:20px;margin:0 0 4px;
      border-left:4px solid var(--teal);padding-left:12px}}
  .sub{{color:#6b7686;font-size:13.5px;margin:0 0 16px;padding-left:16px}}
  table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:6px}}
  th{{background:var(--navy);color:#fff;text-align:left;padding:9px 11px;font-weight:600;
      position:sticky;top:0}}
  td{{padding:8px 11px;border-bottom:1px solid #eef1f5}}
  tbody tr:nth-child(even){{background:#fafbfc}}
  tr.hl td{{background:#fdf3dc;font-weight:600}}
  .tbl-scroll{{max-height:520px;overflow:auto;border-radius:10px;border:1px solid #eef1f5}}
  .two{{display:grid;grid-template-columns:1fr 1fr;gap:22px}}
  @media(max-width:760px){{.two{{grid-template-columns:1fr}}}}
  footer{{text-align:center;color:#8a93a1;font-size:12px;margin-top:30px;line-height:1.7}}
  .pill{{display:inline-block;background:#eef6f6;color:var(--teal);border-radius:12px;
        padding:2px 10px;font-size:12px;margin:2px}}
</style></head>
<body>
<header class="hero">
  <h1>Social Capital Outreach Index</h1>
  <p>Blockchain-enabled Islamic microfinance across OIC economies</p>
  <p>Weighted by Multi-LLM Consensus Weighting (MLCW)</p>
  <div class="badge">Generated {date}</div>
</header>
<div class="wrap">
  <div class="grid">{stats}</div>

  <section class="card">
    <h2>Ranking</h2>
    <p class="sub">Social Capital Index with 95% bootstrap confidence intervals. Highlighted bars are your focal economies.</p>
    {_fig_html(_rank_fig(ranking, focal))}
  </section>

  <section class="card">
    <div class="two">
      <div>
        <h2>Criterion weights</h2>
        <p class="sub">Objective, semantic (LLM panel), and fused weights.</p>
        {_fig_html(_weights_fig(weights))}
      </div>
      <div>
        <h2>Weights &amp; methods</h2>
        <p class="sub">Fused weights and objective-method comparison.</p>
        {weights_tbl}
        <div style="height:10px"></div>
        {methods_tbl}
      </div>
    </div>
  </section>

  <section class="card">
    <h2>Panel agreement diagnostics</h2>
    <p class="sub">Reliability of the LLM-assessor panel behind the semantic weights.</p>
    <div class="grid" style="margin:0">{diag}</div>
  </section>

  <section class="card">
    <h2>Full ranking</h2>
    <p class="sub">Top 25 economies. Download the app's Excel report for the complete table and all sheets.</p>
    <div class="tbl-scroll">{rank_tbl}</div>
  </section>

  <footer>
    <div><span class="pill">real data</span><span class="pill">no imputation</span>
    <span class="pill">reproducible</span></div>
    Headline uses the objective operating point (α = 0) unless changed; all indicators are
    public and fully traceable (Transparency International, World Bank Global Findex, ITU/World Bank).<br>
    Generated by MLCW Studio · companion to the Social Capital Outreach Index manuscript.
  </footer>
</div>
</body></html>"""
