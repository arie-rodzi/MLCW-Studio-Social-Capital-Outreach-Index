# MLCW Studio — Social Capital Outreach Index

Interactive Streamlit companion to the manuscript *"A Multi-LLM Consensus Weighting
(MCDM) Approach to Redefining Microfinance Through Blockchain."* It operationalises the
paper's method: enter indicator data, derive **objective** (Equal/Entropy/CRITIC/MEREC)
and **semantic** (LLM-assessor panel) weights, **fuse** them with a single parameter α,
and rank OIC economies with bootstrap confidence intervals, sensitivity, and agreement
diagnostics.

## Features
- Editable indicator table (ships with the real 53-economy OIC dataset; no imputation).
- Objective weights: Equal, Entropy, CRITIC, MEREC.
- Semantic weights from an LLM-assessor panel (default = the paper's five-assessor panel;
  editable; or replaced by a **live multi-vendor elicitation**).
- Fusion slider α and disagreement penalty λ.
- Ranking with 95% bootstrap CIs, top-N chart, CSV export.
- Fusion sweep, weighting-method comparison (pairwise Spearman).
- Diagnostics: Kendall's W, Cronbach's α, ICC(2,k).

## Use your own data (upload) and download results
In the **📊 Data** tab choose **Upload your own (CSV / Excel)**:
1. Use `data/mlcw_template.xlsx` (bundled) or click **Download Excel template** in the app.
2. Upload your `.xlsx`/`.xls`/`.csv`.
3. Map your columns to the index roles (economy name, transparency proxy, financial-inclusion
   proxy, and optionally a digital-readiness proxy) — the app auto-guesses common names.
4. Click **Load this data**; every tab recomputes on your data.

Download results from the **🏆 Index & Ranking** tab:
- **Download ranking (CSV)** — the ranked index.
- **Download full report (Excel)** — a workbook with sheets *Ranking, Weights,
  Method comparison, Diagnostics*.

## HTML report (beautiful, standalone)
The **📄 HTML Report** tab generates a polished, self-contained HTML report of the current
results (interactive Plotly charts, ranking with confidence intervals, weights, method
comparison, and panel diagnostics). Click **Generate report**, preview it, then
**Download HTML report** — open it in any browser or share it.

## Excel template (53 economies)
In the **📊 Data** tab, **Download Excel template** now gives you the full real 53-economy
OIC dataset (sheet `data`) plus a `sources` sheet. Edit it, or replace the numbers with your
own, and upload it back. A ready copy is also bundled at `data/mlcw_53countries.xlsx`.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open http://localhost:8501

## Deploy to Streamlit Community Cloud (free)
1. Push this folder to a public GitHub repo.
2. On https://share.streamlit.io → **New app** → pick the repo → main file `app.py`.
3. Deploy. (No secrets required for the core app.)

## Optional: live multi-vendor LLM elicitation
The *Live elicitation* tab queries real models under the fair protocol and replaces the
panel with genuine multi-vendor weights.
1. Uncomment the SDK lines in `requirements.txt` (whichever you use).
2. Provide keys either in the app sidebar (session only) or via Streamlit **Secrets**:
   ```toml
   ANTHROPIC_API_KEY = "..."
   OPENAI_API_KEY = "..."
   GOOGLE_API_KEY = "..."
   ```
Keys are read from the environment / session and are never written to disk by the app.

## Files
```
app.py            Streamlit UI
mlcw_core.py      weighting, fusion, index, bootstrap, diagnostics (pure NumPy)
llm_elicit.py     optional live multi-vendor elicitation (lazy SDK imports)
data/oic_data.csv real OIC indicator dataset (CPI, account ownership, internet use)
.streamlit/config.toml  theme
requirements.txt
```

## Note
Headline figures use the objective operating point (α = 0), the most conservative and
fully reproducible choice; raise α to inject the semantic panel. All values are real and
traceable; missing cells are dropped (complete-case), never imputed.
