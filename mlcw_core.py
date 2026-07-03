"""
MLCW core — Multi-LLM Consensus Weighting for the Social Capital Outreach Index.

Pure NumPy implementation of every method used in the paper:
  - min-max normalisation (benefit indicators)
  - objective weights: equal, entropy, CRITIC, MEREC
  - semantic weights from an LLM-assessor panel (reliability -> consensus ->
    disagreement penalty -> renormalise)
  - convex fusion of semantic and objective weights via alpha
  - Social Capital Index, ranking, bootstrap confidence intervals
  - agreement diagnostics: Kendall's W, Cronbach's alpha, ICC(2,k)

No external services required; the LLM panel can be supplied as a matrix
(default = the five-assessor panel reported in the paper) or replaced by a
live elicitation (see llm_elicit.py).
"""
from __future__ import annotations
import numpy as np

# Default a-priori LLM-assessor panel from the paper (rows sum to 1 over
# [transparency, financial_inclusion, digital_readiness]).
DEFAULT_PANEL = {
    "A": [0.35, 0.45, 0.20],
    "B": [0.42, 0.38, 0.20],
    "C": [0.28, 0.40, 0.32],
    "D": [0.33, 0.40, 0.27],
    "E": [0.40, 0.36, 0.24],
}
CRITERIA_3 = ["Transparency", "Financial inclusion", "Digital readiness"]


# ---------- normalisation ----------
def minmax(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, float)
    lo, hi = M.min(0), M.max(0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    return (M - lo) / rng


# ---------- objective weighting methods ----------
def w_equal(Z: np.ndarray) -> np.ndarray:
    k = Z.shape[1]
    return np.full(k, 1.0 / k)


def w_entropy(Z: np.ndarray) -> np.ndarray:
    n = len(Z)
    col = Z.sum(0)
    P = np.divide(Z, np.where(col > 0, col, 1.0))
    P = np.clip(P, 1e-12, None)
    E = -(P * np.log(P)).sum(0) / np.log(n)
    d = 1 - E
    return d / d.sum() if d.sum() > 0 else w_equal(Z)


def w_critic(Z: np.ndarray) -> np.ndarray:
    std = Z.std(0, ddof=1)
    if Z.shape[1] < 2:
        return w_equal(Z)
    corr = np.corrcoef(Z, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    info = std * (1 - corr).sum(0)
    return info / info.sum() if info.sum() > 0 else w_equal(Z)


def w_merec(Z: np.ndarray) -> np.ndarray:
    X = np.clip(Z, 1e-6, None)
    k = X.shape[1]
    lnX = np.log(X)
    S = np.log1p(np.abs(np.mean(lnX, axis=1)))
    removal = np.zeros(k)
    for j in range(k):
        keep = [c for c in range(k) if c != j]
        Sj = np.log1p(np.abs(np.mean(lnX[:, keep], axis=1)))
        removal[j] = np.abs(Sj - S).sum()
    return removal / removal.sum() if removal.sum() > 0 else w_equal(Z)


OBJECTIVE_METHODS = {
    "Equal": w_equal,
    "Entropy": w_entropy,
    "CRITIC": w_critic,
    "MEREC": w_merec,
}


# ---------- semantic (LLM panel) aggregation ----------
def semantic_weights(panel: np.ndarray, lam: float = 5.0):
    """panel: (M assessors x K criteria), rows ~sum to 1.
    Returns dict with reliability r, consensus wbar, dispersion sigma, semantic w."""
    W = np.asarray(panel, float)
    W = W / W.sum(1, keepdims=True)
    med = np.median(W, 0)
    delta = np.abs(W - med).mean(1)
    delta = np.where(delta > 0, delta, 1e-9)
    r = (1 / delta) / (1 / delta).sum()
    wbar = (r[:, None] * W).sum(0)
    sigma = np.sqrt((r[:, None] * (W - wbar) ** 2).sum(0))
    wtil = wbar * np.exp(-lam * sigma)
    wsem = wtil / wtil.sum()
    return {"r": r, "wbar": wbar, "sigma": sigma, "wsem": wsem}


def fuse(wsem: np.ndarray, wobj: np.ndarray, alpha: float) -> np.ndarray:
    w = alpha * np.asarray(wsem) + (1 - alpha) * np.asarray(wobj)
    return w / w.sum()


# ---------- index, ranking, bootstrap ----------
def sci(Z: np.ndarray, w: np.ndarray) -> np.ndarray:
    return Z @ np.asarray(w)


def ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    rk = np.empty(len(scores), int)
    rk[order] = np.arange(1, len(scores) + 1)
    return rk


def bootstrap_ci(M: np.ndarray, weight_fn, draws: int = 2000, seed: int = 42):
    """Resample the economy set; recompute normalisation+weights+scores each draw.
    weight_fn(Z) -> weight vector. Returns (lo, hi) 95% CI per economy on SCI."""
    rng = np.random.default_rng(seed)
    n = len(M)
    acc = np.zeros((draws, n))
    for b in range(draws):
        idx = rng.integers(0, n, n)
        Zb = minmax(M[idx])
        wb = weight_fn(Zb)
        # score the ORIGINAL economies using the resampled normalisation range
        lo, hi = M[idx].min(0), M[idx].max(0)
        rng_ = np.where(hi > lo, hi - lo, 1.0)
        Zall = np.clip((M - lo) / rng_, 0, 1)
        acc[b] = Zall @ wb
    lo = np.percentile(acc, 2.5, axis=0)
    hi = np.percentile(acc, 97.5, axis=0)
    return lo, hi


# ---------- agreement diagnostics ----------
def kendall_w(panel: np.ndarray) -> float:
    W = np.asarray(panel, float)
    m, k = W.shape
    if k < 2:
        return float("nan")
    R = np.array([k - np.argsort(np.argsort(row)) for row in W], float)
    Rs = R.sum(0)
    S = ((Rs - Rs.mean()) ** 2).sum()
    return float(12 * S / (m ** 2 * (k ** 3 - k)))


def cronbach_alpha(panel: np.ndarray) -> float:
    X = np.asarray(panel, float).T  # criteria x assessors
    k = X.shape[1]
    tot = X.sum(1).var(ddof=1)
    if k < 2 or tot == 0:
        return float("nan")
    return float((k / (k - 1)) * (1 - X.var(0, ddof=1).sum() / tot))


def icc2k(panel: np.ndarray) -> float:
    X = np.asarray(panel, float).T  # targets(criteria) x raters(assessors)
    n, k = X.shape
    if n < 2 or k < 2:
        return float("nan")
    gm = X.mean()
    MSR = k * ((X.mean(1) - gm) ** 2).sum() / (n - 1)
    MSC = n * ((X.mean(0) - gm) ** 2).sum() / (k - 1)
    MSE = ((X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    denom = MSR + (MSC - MSE) / n
    return float((MSR - MSE) / denom) if denom != 0 else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])
