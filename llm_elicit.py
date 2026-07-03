"""
Optional live multi-vendor elicitation for MLCW.

Sends the FAIR prompt (no country data; symmetric criteria; anti-anchoring;
randomised criterion order) to whichever providers have an API key set, parses
the returned weights, and returns a panel matrix for mlcw_core.semantic_weights.

Provider SDKs are imported lazily, so the app deploys and runs fine WITHOUT them.
Install only what you use:  pip install anthropic openai google-generativeai
"""
from __future__ import annotations
import os, re, json, random
import numpy as np

CRITERIA = {
    "transparency": "Transparency / institutional integrity. Proxy: Corruption Perceptions Index.",
    "financial_inclusion": "Financial inclusion. Proxy: account ownership (Global Findex).",
    "digital_readiness": "Digital readiness / connectivity. Proxy: individuals using the internet.",
}
SYSTEM = ("You are an independent expert in development economics, Islamic finance, financial "
          "inclusion, and composite social indicators, serving as ONE assessor on a panel. Reason "
          "independently. Do not assume any criterion is more important by default. You will NOT be "
          "shown any country data - your judgement must be conceptual and a priori.")


def _user(order):
    lines = "\n".join(f"- ({chr(97+i)}) {CRITERIA[k]}" for i, k in enumerate(order))
    return (
        "We are building a composite index of the SOCIAL-CAPITAL CONDITIONS that enable the OUTREACH "
        "of blockchain-enabled Islamic microfinance across OIC economies (breadth and depth of services "
        "reaching the underserved).\n\nCRITERIA (order randomised, does NOT imply importance):\n"
        f"{lines}\n\nAssign each a non-negative weight summing to exactly 1.00. Do NOT anchor on equal "
        'weights or on order. Return ONLY JSON: {"weights": {"transparency": x, "financial_inclusion": y, '
        '"digital_readiness": z}, "rationale": {...}, "confidence": c}.')


def _parse(text):
    obj = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    w = obj["weights"]
    v = np.array([w["transparency"], w["financial_inclusion"], w["digital_readiness"]], float)
    v = np.clip(v, 0, None)
    return v / v.sum(), obj.get("rationale", {})


def _anthropic(sys, usr, model, temp):
    import anthropic
    r = anthropic.Anthropic().messages.create(
        model=model, max_tokens=600, temperature=temp, system=sys,
        messages=[{"role": "user", "content": usr}])
    return "".join(b.text for b in r.content if b.type == "text")


def _openai(sys, usr, model, temp):
    from openai import OpenAI
    r = OpenAI().chat.completions.create(
        model=model, temperature=temp,
        messages=[{"role": "system", "content": sys}, {"role": "user", "content": usr}])
    return r.choices[0].message.content


def _google(sys, usr, model, temp):
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    gm = genai.GenerativeModel(model, system_instruction=sys)
    return gm.generate_content(usr, generation_config={"temperature": temp}).text


PROVIDERS = {
    "Anthropic (Claude)": ("ANTHROPIC_API_KEY", _anthropic, "claude-3-5-sonnet-20241022"),
    "OpenAI (GPT)":       ("OPENAI_API_KEY", _openai, "gpt-4o"),
    "Google (Gemini)":    ("GOOGLE_API_KEY", _google, "gemini-1.5-pro"),
}


def available_providers():
    return [name for name, (env, _, _) in PROVIDERS.items() if os.environ.get(env)]


def elicit(selected, runs=1, temperature=0.7, seed=0):
    """Return (labels, panel_matrix Mx3, rationales). Raises if a call fails."""
    random.seed(seed)
    labels, rows, rats = [], [], []
    for name in selected:
        env, fn, model = PROVIDERS[name]
        for run in range(runs):
            order = random.sample(list(CRITERIA), 3)
            txt = fn(SYSTEM, _user(order), model, temperature)
            v, rat = _parse(txt)
            labels.append(f"{name.split()[0]}#{run+1}")
            rows.append(v.tolist())
            rats.append(rat)
    return labels, np.array(rows, float), rats
