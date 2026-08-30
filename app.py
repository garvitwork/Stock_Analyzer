"""
Stock Investment Analyzer — upload-driven decision-support tool.

You upload the actual source document for each analysis parameter
(financial statements, filings, news, etc.). The app extracts the
content and sends it to your chosen model, which reasons through a
fixed 10-category institutional-style framework and returns a
structured verdict: what the data supports, what's missing, and open
risks.

This is a decision-support tool, not investment advice — the output
is a reasoned synthesis of what you gave it, not a guarantee.

Two providers supported:
  - GPT-OSS-120B via Groq   — get a key at https://console.groq.com/keys
  - Gemini 3 Flash via Google — free tier, get a key at https://aistudio.google.com/apikey

SETUP
------
pip install streamlit groq google-genai pandas openpyxl pypdf tabulate
export GROQ_API_KEY=gsk_...
export GEMINI_API_KEY=AIza...
streamlit run app.py
"""

import io
import json
import os
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
from groq import Groq, APIStatusError
from google import genai
from google.genai.errors import ClientError as GeminiClientError

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# ---------------------------------------------------------------------------
# 1. The framework: one entry per analysis parameter.
#    Edit this list freely — it's the single source of truth for both the
#    UI (what upload slots appear) and the prompt sent to the model.
# ---------------------------------------------------------------------------

CATEGORIES = [
    {
        "key": "fundamentals",
        "title": "Fundamental / Financial Analysis",
        "ask": "Income statement, balance sheet, cash flow statement (last 3-5 years).",
        "source_hint": "Company 10-K/10-Q (SEC EDGAR, free) or exported financials from your broker.",
        "filetypes": ["xlsx", "csv", "pdf"],
    },
    {
        "key": "valuation",
        "title": "Valuation & Margin of Safety",
        "ask": "Any valuation workbook you have (DCF, comps, historical P/E band) — or the price/ratio history from your broker export.",
        "source_hint": "Your own model, or a broker/data-provider export (P/E, P/B, EV/EBITDA history).",
        "filetypes": ["xlsx", "csv", "pdf"],
    },
    {
        "key": "moat",
        "title": "Business Quality & Moat",
        "ask": "Annual report narrative section, investor day deck, or an analyst note discussing competitive position.",
        "source_hint": "Company 10-K 'Business' section, investor relations site, or a research note.",
        "filetypes": ["pdf", "txt"],
    },
    {
        "key": "macro",
        "title": "Macro & Cycle Positioning",
        "ask": "Industry/sector outlook note or macro report relevant to this company's sector.",
        "source_hint": "Sell-side sector report, central bank/industry association publication, or a news roundup you've saved.",
        "filetypes": ["pdf", "txt"],
    },
    {
        "key": "sentiment",
        "title": "Market Psychology & Sentiment",
        "ask": "Recent news clippings, short-interest report, or options-flow snapshot.",
        "source_hint": "FINRA short-interest data (free), your broker's sentiment tab, or saved news articles.",
        "filetypes": ["xlsx", "pdf", "txt", "csv"],
    },
    {
        "key": "risk",
        "title": "Risk Management Inputs",
        "ask": "Your current portfolio holdings export (for correlation/position-sizing context).",
        "source_hint": "Export from your brokerage account.",
        "filetypes": ["xlsx", "csv"],
    },
    {
        "key": "technicals",
        "title": "Technical & Price Action",
        "ask": "Price/volume history export or a chart screenshot description.",
        "source_hint": "Broker or charting platform export (e.g. TradingView CSV export).",
        "filetypes": ["csv", "xlsx"],
    },
    {
        "key": "catalyst",
        "title": "Catalyst Identification",
        "ask": "Earnings call transcript, product-launch announcement, or regulatory filing tied to an upcoming event.",
        "source_hint": "Company IR page, SEC 8-K filings, or a news article on the specific catalyst.",
        "filetypes": ["pdf", "txt"],
    },
]

# Hard ceiling on any single file's raw extracted text before budgeting kicks in.
MAX_CHARS_PER_FILE = 120_000

# ---------------------------------------------------------------------------
# 2. Provider configuration — this is the part that actually varies between
#    Groq and Gemini: different SDKs, different auth, and very different
#    rate-limit ceilings (Gemini's free tier is far more generous on TPM).
# ---------------------------------------------------------------------------

PROVIDERS = {
    "GPT-OSS-120B (Groq, free tier)": {
        "provider": "groq",
        "model": "openai/gpt-oss-120b",
        "tpm_limit": 8_000,        # tight — this is the one that kept 413ing
        "key_env": "GROQ_API_KEY",
        "key_help": "Free at https://console.groq.com/keys",
        "key_prefix_hint": "gsk_...",
    },
    "Gemini 3.5 Flash (Google, free tier)": {
        "provider": "gemini",
        "model": "gemini-3.5-flash",
        "tpm_limit": 200_000,      # generous free-tier TPM (~250k), kept a safety margin
        "key_env": "GEMINI_API_KEY",
        "key_help": "Free at https://aistudio.google.com/apikey",
        "key_prefix_hint": "AIza...",
    },
}

SAFETY_MARGIN = 0.75  # only use 75% of the stated limit to leave headroom


def estimate_tokens(text: str) -> int:
    """Rough token estimate — good enough for budgeting, no extra dependency needed."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Analysis history — local SQLite file, no server/credentials involved.
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_history.db")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            timestamp TEXT,
            provider TEXT,
            verdict TEXT,
            score REAL,
            result_json TEXT
        )
    """)
    conn.commit()
    conn.close()


def compute_score(result: dict) -> float:
    """Rough 0-100 score: share of assessed categories with a positive signal."""
    signal_points = {"Positive": 1.0, "Neutral": 0.5, "Negative": 0.0}
    scored = [signal_points[c["signal"]] for c in result.get("categories", []) if c.get("signal") in signal_points]
    return round(sum(scored) / len(scored) * 100, 1) if scored else 0.0


def save_history(ticker: str, provider: str, result: dict):
    score = compute_score(result)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO history (ticker, timestamp, provider, verdict, score, result_json) VALUES (?, ?, ?, ?, ?, ?)",
        (ticker, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), provider, result.get("verdict", ""), score, json.dumps(result)),
    )
    conn.commit()
    conn.close()


def load_history() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM history ORDER BY timestamp DESC", conn)
    conn.close()
    return df


def delete_history_row(row_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history WHERE id = ?", (row_id,))
    conn.commit()
    conn.close()


init_db()


# ---------------------------------------------------------------------------
# 3. File extraction helpers
# ---------------------------------------------------------------------------

def extract_text(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith((".xlsx", ".xls")):
        try:
            sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
            parts = []
            for sheet_name, df in sheets.items():
                parts.append(f"[Sheet: {sheet_name}]\n{df.to_markdown(index=False)}")
            return "\n\n".join(parts)[:MAX_CHARS_PER_FILE]
        except Exception as e:
            return f"[Could not parse Excel file: {e}]"

    if name.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(data))
            return df.to_markdown(index=False)[:MAX_CHARS_PER_FILE]
        except Exception as e:
            return f"[Could not parse CSV file: {e}]"

    if name.endswith(".pdf"):
        if PdfReader is None:
            return "[pypdf not installed — cannot parse PDF]"
        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text[:MAX_CHARS_PER_FILE]
        except Exception as e:
            return f"[Could not parse PDF file: {e}]"

    # plain text / anything else
    try:
        return data.decode("utf-8", errors="ignore")[:MAX_CHARS_PER_FILE]
    except Exception as e:
        return f"[Could not read file: {e}]"


def budget_category_texts(category_texts: dict, input_token_budget: int) -> dict:
    """Proportionally trims each category's text so the TOTAL input stays within budget."""
    total_chars = sum(len(t) for t in category_texts.values())
    if total_chars == 0:
        return category_texts

    total_tokens = estimate_tokens("x" * total_chars)
    if total_tokens <= input_token_budget:
        return category_texts

    trimmed = {}
    for key, text in category_texts.items():
        share = len(text) / total_chars
        char_budget = int(input_token_budget * share * 4)
        if len(text) > char_budget:
            trimmed[key] = text[:char_budget] + "\n[...trimmed to fit rate limit...]"
        else:
            trimmed[key] = text
    return trimmed


# ---------------------------------------------------------------------------
# 4. Prompt + provider-specific calls
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an equity-analysis assistant helping an individual investor \
think through a potential stock investment using an institutional-style framework. \
You have been given source documents the user uploaded, organized by analysis category. \
Some categories may have no uploaded material — do not invent data for those; mark them \
as insufficient instead.

Respond with ONLY valid JSON (no markdown fences, no preamble), in this exact shape:
{
  "categories": [
    {"category": "<category title>", "signal": "Positive|Negative|Neutral|Insufficient",
     "inferred": "<what the evidence shows, <25 words>", "reason": "<why it matters, <25 words>"}
    ... one entry per category, same order as given, every category included even if insufficient ...
  ],
  "verdict": "Favorable|Unfavorable|Insufficient information",
  "top_reasons": ["<=4 short bullets"],
  "risks": ["<=4 short bullets on open risks/uncertainties, including gaps from missing categories"],
  "missing_docs": ["<=4 short bullets on what documents would most improve this"]
}

Be direct and specific — reference actual numbers/facts from the documents rather than \
generic statements. If material for a category is too thin, set signal to "Insufficient" \
and say so plainly in "inferred" rather than padding. This is a decision-support summary, \
not financial advice — never phrase verdict as a guaranteed buy/sell instruction. Keep \
every field concise since the response budget is limited."""


def call_groq(api_key, model, system_prompt, user_content, max_output_tokens):
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_output_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    choice = response.choices[0]
    text = choice.message.content or ""
    truncated = choice.finish_reason == "length"
    return text, truncated


def call_gemini(api_key, model, system_prompt, user_content, max_output_tokens):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=user_content,
        config={
            "system_instruction": system_prompt,
            "max_output_tokens": max_output_tokens,
        },
    )
    text = response.text or ""
    truncated = getattr(response.candidates[0], "finish_reason", None) == "MAX_TOKENS"
    return text, truncated


def run_analysis(ticker, category_texts, api_key, provider_key, max_output_tokens):
    cfg = PROVIDERS[provider_key]
    tpm_limit = cfg["tpm_limit"]
    usable_budget = int(tpm_limit * SAFETY_MARGIN)
    system_tokens = estimate_tokens(SYSTEM_PROMPT)

    input_budget = usable_budget - system_tokens - max_output_tokens
    if input_budget < 500:
        max_output_tokens = max(300, usable_budget - system_tokens - 500)
        input_budget = 500

    trimmed_texts = budget_category_texts(category_texts, input_budget)

    sections = []
    for cat in CATEGORIES:
        text = trimmed_texts.get(cat["key"])
        if text:
            sections.append(f"## {cat['title']}\n{text}")
        else:
            sections.append(f"## {cat['title']}\n[No document uploaded for this category]")

    user_content = f"Ticker / company: {ticker}\n\n" + "\n\n".join(sections)
    estimated_total = system_tokens + estimate_tokens(user_content) + max_output_tokens

    try:
        if cfg["provider"] == "groq":
            text, truncated = call_groq(api_key, cfg["model"], SYSTEM_PROMPT, user_content, max_output_tokens)
        else:
            text, truncated = call_gemini(api_key, cfg["model"], SYSTEM_PROMPT, user_content, max_output_tokens)
    except APIStatusError as e:
        if e.status_code == 413 or "rate_limit_exceeded" in str(e):
            raise RuntimeError(
                f"Still too large even after trimming (~{estimated_total} estimated tokens "
                f"vs a ~{tpm_limit} token/min limit for {provider_key}). Try: unchecking a "
                f"category or two below, switching provider, or lowering 'Max response length'."
            ) from e
        raise
    except GeminiClientError as e:
        if getattr(e, "code", None) == 429:
            raise RuntimeError(
                f"Gemini rate limit hit (~{estimated_total} estimated tokens sent). "
                f"Free tier also caps requests/minute and requests/day — wait a moment "
                f"and retry, or switch to the Groq provider."
            ) from e
        raise RuntimeError(f"Gemini request failed: {e}") from e

    if truncated:
        raise RuntimeError(
            "Response was cut off at the token limit before finishing the JSON. "
            "Raise 'Max response length' in the sidebar, or trim an uploaded document."
        )

    clean = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Model response wasn't valid JSON ({e}). Try again.") from e

    return parsed, estimated_total, tpm_limit


# ---------------------------------------------------------------------------
# 5. Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Stock Investment Analyzer", layout="wide")
st.title("Stock Investment Analyzer")
st.caption(
    "Upload the source document for each parameter below. The verdict is a "
    "synthesis of what you provide — not financial advice."
)

with st.sidebar:
    st.header("Setup")

    provider_key = st.selectbox("Model", list(PROVIDERS.keys()), index=0)
    cfg = PROVIDERS[provider_key]

    api_key = st.text_input(
        f"{provider_key.split('(')[0].strip()} API key",
        type="password",
        value=os.environ.get(cfg["key_env"], ""),
        help=f"{cfg['key_help']} — format: {cfg['key_prefix_hint']}. Set via the "
        f"{cfg['key_env']} environment variable instead of pasting here when possible.",
    )
    st.caption(f"Rate limit for this model: ~{cfg['tpm_limit']:,} tokens/min (est.)")

    max_output_tokens = st.slider(
        "Max response length (output tokens)",
        min_value=500,
        max_value=10000,
        value=1500 if cfg["provider"] == "groq" else 2500,
        step=250,
        help="Lower this if you keep hitting rate limits — output tokens count "
        "against the same per-minute budget as your uploaded documents (mainly "
        "matters for Groq; Gemini's free tier has far more headroom).",
    )
    ticker = st.text_input("Ticker / company name", placeholder="e.g. AAPL")

category_texts = {}

for cat in CATEGORIES:
    with st.expander(f"{cat['title']}", expanded=False):
        st.write(cat["ask"])
        st.caption(f"Typical source: {cat['source_hint']}")
        files = st.file_uploader(
            "Upload file(s)",
            type=cat["filetypes"],
            accept_multiple_files=True,
            key=f"upload_{cat['key']}",
        )
        if files:
            combined = []
            for f in files:
                combined.append(f"### {f.name}\n{extract_text(f)}")
            category_texts[cat["key"]] = "\n\n".join(combined)
            st.success(f"{len(files)} file(s) loaded (~{estimate_tokens(''.join(combined)):,} tokens before budgeting).")

st.divider()

uploaded_count = len(category_texts)
st.write(f"**{uploaded_count} / {len(CATEGORIES)}** categories have uploaded material.")

raw_total_tokens = sum(estimate_tokens(t) for t in category_texts.values())
if raw_total_tokens:
    st.caption(
        f"Raw uploaded content is ~{raw_total_tokens:,} tokens. This will be "
        f"automatically trimmed to fit the selected model's rate limit before sending."
    )

if st.button("Run Analysis", type="primary", disabled=not api_key or not ticker):
    with st.spinner("Analyzing..."):
        try:
            result, est_tokens, tpm_limit = run_analysis(
                ticker, category_texts, api_key, provider_key, max_output_tokens
            )
            st.session_state["result"] = result
            st.session_state["est_tokens"] = est_tokens
            st.session_state["tpm_limit"] = tpm_limit
            st.session_state["ticker"] = ticker
            st.session_state["provider_key"] = provider_key
            save_history(ticker, provider_key, result)
        except RuntimeError as e:
            st.session_state.pop("result", None)
            st.error(str(e))
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"Analysis failed: {e}")

# Rendered from session_state (not directly under the button) so that
# clicking a download button — which also triggers a Streamlit rerun —
# doesn't wipe the results off the screen.
# Scope reruns from download-button clicks to just this block, instead of
# re-running the whole script (which was causing the "still running" flicker
# and re-executing the file uploaders above).
_fragment = getattr(st, "fragment", None) or getattr(st, "experimental_fragment")


@_fragment
def render_results():
    result = st.session_state["result"]
    est_tokens = st.session_state["est_tokens"]
    tpm_limit = st.session_state["tpm_limit"]
    ticker = st.session_state["ticker"]
    provider_key = st.session_state["provider_key"]

    st.caption(f"Sent ~{est_tokens:,} estimated tokens (limit ~{tpm_limit:,}/min for {provider_key}).")

    st.subheader(f"Verdict: {result.get('verdict', 'N/A')}")

    df = pd.DataFrame(result.get("categories", []))
    df = df.rename(columns={
        "category": "Category", "signal": "Signal",
        "inferred": "What Was Inferred", "reason": "Reason",
    })
    st.dataframe(df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        for label, items in [("Top Reasons", "top_reasons"), ("Key Risks", "risks")]:
            st.markdown(f"**{label}**")
            for item in result.get(items, []):
                st.markdown(f"- {item}")
    with col2:
        st.markdown("**Docs That Would Help Most**")
        for item in result.get("missing_docs", []):
            st.markdown(f"- {item}")

    # --- Downloads ---
    summary_rows = [
        {"Category": "OVERALL VERDICT", "Signal": result.get("verdict", ""), "What Was Inferred": "", "Reason": ""}
    ]
    full_df = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)

    csv_bytes = full_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download as CSV", csv_bytes,
        file_name=f"{ticker}_analysis.csv", mime="text/csv",
        key="csv_download",
    )

    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        full_df.to_excel(writer, index=False, sheet_name="Analysis")
        extras = pd.DataFrame({
            "Top Reasons": pd.Series(result.get("top_reasons", [])),
            "Key Risks": pd.Series(result.get("risks", [])),
            "Docs That Would Help": pd.Series(result.get("missing_docs", [])),
        })
        extras.to_excel(writer, index=False, sheet_name="Notes")
    st.download_button(
        "Download as Excel", excel_buf.getvalue(),
        file_name=f"{ticker}_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="excel_download",
    )


if "result" in st.session_state:
    render_results()

# ---------------------------------------------------------------------------
# Analysis History
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Analysis History")

hist_df = load_history()
if hist_df.empty:
    st.caption("No past analyses yet — run one above and it'll show up here.")
else:
    display_df = hist_df[["id", "ticker", "timestamp", "provider", "verdict", "score"]].rename(columns={
        "id": "ID", "ticker": "Ticker", "timestamp": "When",
        "provider": "Model", "verdict": "Verdict", "score": "Score",
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        options = [f"{r.id} — {r.ticker} ({r.timestamp})" for r in hist_df.itertuples()]
        picked = st.selectbox("View a past run in detail", options, key="history_picker")
    with col2:
        if st.button("Load into view above"):
            picked_id = int(picked.split(" — ")[0])
            row = hist_df[hist_df["id"] == picked_id].iloc[0]
            st.session_state["result"] = json.loads(row["result_json"])
            st.session_state["est_tokens"] = 0
            st.session_state["tpm_limit"] = 0
            st.session_state["ticker"] = row["ticker"]
            st.session_state["provider_key"] = row["provider"]
            st.rerun()
    with col3:
        if st.button("Delete this run"):
            picked_id = int(picked.split(" — ")[0])
            delete_history_row(picked_id)
            st.rerun()

    st.download_button(
        "Download full history as CSV",
        hist_df.drop(columns=["result_json"]).to_csv(index=False).encode("utf-8"),
        file_name="analysis_history.csv", mime="text/csv",
    )

if not api_key:
    st.info(f"Enter your {provider_key.split('(')[0].strip()} API key in the sidebar to run an analysis.")