"""
NicheIQ — Streamlit UI (Production)
"""
from __future__ import annotations

import contextlib
import html as _html
import io
import math
import os
from pathlib import Path

import streamlit as st

# ── Streamlit Cloud secrets → os.environ ──────────────────────────────────────
# market_gap_engine.py reads keys via os.getenv(). On Streamlit Community Cloud
# there is no .env file — secrets live in st.secrets. Copy them into os.environ
# before the engine module is imported so load_dotenv() finds them.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass  # local dev: .env is loaded by market_gap_engine itself

import pandas as pd

from market_gap_engine import (
    CSV_COLUMNS,
    IdeaHarvester,
    NicheReport,
    OpportunityScorer,
    OUTPUT_FILE,
    ProductIdea,
    ProductIdeaGenerator,
    save_to_csv,
)

# ─── page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NicheIQ",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── global CSS ───────────────────────────────────────────────────────────────

st.markdown("""<style>
/* Hide Streamlit chrome */
#MainMenu, footer { visibility: hidden; }
.stDeployButton { display: none; }

/* Font */
html, body, [class*="css"] {
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

/* Overflow fix — required so CSS tooltips aren't clipped by column divs */
[data-testid="stMarkdownContainer"],
.element-container,
[data-testid="column"] { overflow: visible !important; }

/* ── Metric card ─────────────────────────────────────── */
.mcard {
  background: #181c2e;
  border: 1px solid #252a42;
  border-radius: 12px;
  padding: 16px 18px;
  min-height: 120px;
  position: relative;
}
.mcard-label {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: #5a6490;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 5px;
}
.mcard-val {
  font-size: 28px;
  font-weight: 800;
  line-height: 1;
  margin-bottom: 8px;
}
.mcard-bar { height: 3px; background: #252a42; border-radius: 2px; overflow: hidden; }
.mcard-bar-f { height: 3px; border-radius: 2px; }
.mcard-src { font-size: 10px; color: #34405c; margin-top: 8px; font-style: italic; }

/* ── CSS tooltip (formula explanation on hover) ────── */
.tip {
  position: relative;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 14px; height: 14px;
  border-radius: 50%;
  background: #252a42;
  color: #5a6490;
  font-size: 9px;
  font-weight: 700;
  cursor: help;
  flex-shrink: 0;
  font-style: normal;
  line-height: 1;
}
.tip::after {
  content: attr(data-tip);
  position: absolute;
  bottom: calc(100% + 8px);
  left: 50%;
  transform: translateX(-50%);
  background: #181c2e;
  border: 1px solid #353d66;
  border-radius: 8px;
  padding: 10px 13px;
  font-size: 11px;
  line-height: 1.65;
  color: #b4bed8;
  width: 250px;
  white-space: pre-line;
  text-transform: none;
  letter-spacing: 0;
  font-weight: 400;
  font-style: normal;
  visibility: hidden;
  opacity: 0;
  transition: opacity .15s ease;
  z-index: 9999;
  pointer-events: none;
}
.tip:hover::after { visibility: visible; opacity: 1; }

/* ── Score ring ──────────────────────────────────────── */
.ring-wrap { display: flex; flex-direction: column; align-items: center; }

/* ── Tier badge ──────────────────────────────────────── */
.tbadge {
  display: inline-block;
  padding: 3px 10px;
  border-radius: 20px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
}
.tbg { background: #0d2b1e; color: #34d399; border: 1px solid #1a4a35; }
.tby { background: #2b2308; color: #fbbf24; border: 1px solid #4a3c12; }
.tbr { background: #2b1010; color: #f87171; border: 1px solid #4a1a1a; }

/* ── Pain snippet card ───────────────────────────────── */
.snip {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 9px 13px;
  background: #131625;
  border: 1px solid #1c2035;
  border-left: 3px solid;
  border-radius: 8px;
  margin-bottom: 7px;
  font-size: 12.5px;
  color: #b4bed8;
  line-height: 1.55;
}
.stag {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 9px;
  font-size: 9px;
  font-weight: 700;
  letter-spacing: .07em;
  text-transform: uppercase;
  white-space: nowrap;
  flex-shrink: 0;
  margin-top: 2px;
}
.stag-r { background: #2b1808; color: #fb923c; border: 1px solid #4a2e14; }
.stag-h { background: #261d08; color: #fbbf24; border: 1px solid #4a3a14; }
.stag-w { background: #08142b; color: #60a5fa; border: 1px solid #142a4a; }

/* ── Formula block ───────────────────────────────────── */
.fblock {
  background: #131625;
  border: 1px solid #1c2035;
  border-radius: 10px;
  padding: 14px 18px;
  font-size: 12px;
  color: #6b7599;
  font-family: 'Fira Code', 'JetBrains Mono', 'Courier New', monospace;
  line-height: 1.9;
  white-space: pre;
  overflow-x: auto;
}
.fblock b  { color: #c0cae4; }
.fblock .g { color: #10b981; font-weight: 800; }
.fblock .w { color: #f59e0b; }

/* ── Idea card ───────────────────────────────────────── */
.icard {
  background: #181c2e;
  border: 1px solid #252a42;
  border-radius: 12px;
  padding: 20px;
  transition: transform .15s, box-shadow .15s;
  height: 100%;
}
.icard:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,.35); }
.icard-n  { font-size: 15px; font-weight: 800; color: #e8ecf5; margin-bottom: 4px; }
.icard-l  {
  font-size: 12px; color: #5a6490; font-style: italic;
  margin-bottom: 14px; padding-bottom: 14px;
  border-bottom: 1px solid #1c2035;
}
.icard-fl { font-size: 9.5px; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: #38436a; margin: 10px 0 3px; }
.icard-fv { font-size: 12.5px; color: #9ba4c0; line-height: 1.5; }

/* ── Sidebar score bar ───────────────────────────────── */
.sbbar { height: 3px; background: #1c2035; border-radius: 2px; margin: -5px 0 8px; overflow: hidden; }
.sbbar-f { height: 3px; border-radius: 2px; }

/* ── Landing page feature cards ──────────────────────── */
.lcard { background: #181c2e; border: 1px solid #252a42; border-radius: 12px; padding: 20px; }
.lcard-icon  { font-size: 26px; margin-bottom: 10px; }
.lcard-title { font-size: 14px; font-weight: 700; color: #e8ecf5; margin-bottom: 6px; }
.lcard-desc  { font-size: 12px; color: #5a6490; line-height: 1.6; }
</style>""", unsafe_allow_html=True)

# ─── metric tooltip definitions ───────────────────────────────────────────────

_TIPS: dict[str, tuple[str, str, str]] = {
    "solutionless": (
        "Solutionless Rate",
        "GPT-4o-mini reads each pain snippet and\n"
        "counts posts where no existing tool is\n"
        "mentioned as a solution.\n\n"
        "Rate × 10 → base score (0–10)\n"
        "+ HN Ask boost: +0.15 per Ask HN post\n"
        "  (capped at +1.5 total)",
        "LLM analysis + HN Algolia API",
    ),
    "wtp": (
        "Willingness to Pay",
        "Two blended signals (0–5 pts each):\n\n"
        "① Price anchors in snippets\n"
        "   Regex: $X/mo, costs $X, etc.\n\n"
        "② LinkedIn/Indeed job postings\n"
        "   for manual-role titles in the niche",
        "Regex on snippets · job board search",
    ),
    "momentum": (
        "Market Momentum",
        "Primary: Google Trends (pytrends)\n"
        "  last-12-week avg ÷ 52-week avg × 5\n"
        "  5.0 = flat | >5 = growing | <5 = declining\n\n"
        "Fallback (if Trends rate-limited):\n"
        "  Serper 30-day vs annual count ratio",
        "Google Trends (pytrends) · Serper fallback",
    ),
    "pain": (
        "Pain Intensity",
        "GPT-4o-mini rates financial/time cost\n"
        "with calibrated reference anchors:\n\n"
        "2  = cosmetic annoyance\n"
        "4  = 1–2 hrs/week manual work\n"
        "5  = $200–500/mo cost\n"
        "6  = $500–2,000/mo\n"
        "8  = $5,000+/mo or compliance risk\n"
        "10 = existential (business failure risk)",
        "LLM (GPT-4o-mini, calibrated anchors)",
    ),
    "density": (
        "Competitive Density",
        "65% SaaS saturation\n"
        "   dedicated product pages on SERP page 1\n"
        "   7+ tools = max score\n\n"
        "35% Competitor age blend\n"
        "   WHOIS / DetectionLayer founding year\n"
        "   younger avg age = more crowded\n\n"
        "LOWER score = more market space",
        "SERP analysis · WHOIS · DetectionLayer",
    ),
    "demand": (
        "Demand Signal",
        "Quality-weighted volume across sources:\n\n"
        "Serper result volume   (max 5.0)\n"
        "+ pain keyword bonus  (max 2.0)\n"
        "+ forum source bonus  (max 0.8)\n"
        "+ HN Ask count bonus  (max 0.8)\n"
        "+ Reddit community    (max 1.2)",
        "Serper · Reddit · HN Algolia",
    ),
    "hn_ask": (
        "Ask HN Posts",
        "'Ask HN: is there a tool for X?' posts\n"
        "matching the niche, from HN Algolia API.\n\n"
        "Each post = someone publicly searching\n"
        "for a product they couldn't find — a\n"
        "direct unsatisfied demand signal.\n\n"
        "Also boosts Solutionless Score\n"
        "(+0.15 per post, capped at +1.5).",
        "HN Algolia API (free, no key required)",
    ),
    "reddit_subs": (
        "Reddit Community",
        "Sum of subscriber counts across the\n"
        "top 3 subreddits matching the niche.\n"
        "Used as a TAM (market size) proxy.\n\n"
        ">1M   = large addressable market\n"
        "100k–1M = solid niche community\n"
        "<100k = small or emerging niche\n\n"
        "Also boosts Demand Score (up to +1.2).",
        "Reddit API (PRAW, read-only)",
    ),
}

# ─── cached resources ─────────────────────────────────────────────────────────


@st.cache_resource
def get_scorer() -> OpportunityScorer:
    return OpportunityScorer()


@st.cache_resource
def get_generator() -> ProductIdeaGenerator:
    return ProductIdeaGenerator()


# ─── session state ────────────────────────────────────────────────────────────


def _init_state() -> None:
    for key, val in {"reports": [], "selected": None, "log": ""}.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if "csv_loaded" not in st.session_state:
        st.session_state.csv_loaded = True
        p = Path(__file__).parent / OUTPUT_FILE
        if p.exists():
            try:
                st.session_state.csv_df = pd.read_csv(p)
            except Exception:
                pass


_init_state()

# ─── pure helpers ─────────────────────────────────────────────────────────────


def _color(v: float) -> str:
    """Score → hex color. Aligned with tier thresholds."""
    return "#10b981" if v >= 6.0 else "#f59e0b" if v >= 4.5 else "#ef4444"


def _tier(v: float) -> tuple[str, str]:
    """Score → (tier_label, css_class)."""
    if v >= 7.5:
        return "UNDERSERVED",    "tbg"
    if v >= 6.0:
        return "HIGH POTENTIAL", "tbg"
    if v >= 4.5:
        return "CONTESTED",      "tby"
    return "SATURATED", "tbr"


def _fmt_n(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


def _get_report(niche: str) -> NicheReport | None:
    return next((r for r in st.session_state.reports if r.niche == niche), None)


def _run_single(niche: str, scorer: OpportunityScorer) -> NicheReport | None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            report = scorer.analyze(niche)
        except Exception as e:
            buf.write(f"[error] {e}\n")
            report = None
    st.session_state.log = buf.getvalue()
    return report


def _merge(report: NicheReport) -> None:
    ex = {r.niche: r for r in st.session_state.reports}
    ex[report.niche] = report
    st.session_state.reports = list(ex.values())


# ─── HTML builders ────────────────────────────────────────────────────────────


def _tip_attr(text: str) -> str:
    """Encode tip text safe for a data-tip HTML attribute."""
    return _html.escape(text).replace("\n", "&#10;")


def _mcard(
    key: str,
    value: str,
    score_val: float | None = None,
    color_override: str | None = None,
) -> str:
    """Return HTML for a single metric card with CSS tooltip."""
    label, tip_raw, source = _TIPS.get(key, (key, "", ""))
    tip = _tip_attr(tip_raw)
    c = color_override or (_color(score_val) if score_val is not None else "#5a6490")
    bar = ""
    if score_val is not None:
        pct = max(0.0, min(100.0, score_val / 10 * 100))
        bar = (
            f'<div class="mcard-bar">'
            f'<div class="mcard-bar-f" style="width:{pct:.0f}%;background:{c}"></div>'
            f'</div>'
        )
    tip_icon = f'<i class="tip" data-tip="{tip}">i</i>' if tip_raw else ""
    return (
        f'<div class="mcard">'
        f'<div class="mcard-label">{_html.escape(label)}{tip_icon}</div>'
        f'<div class="mcard-val" style="color:{c}">{_html.escape(value)}</div>'
        f'{bar}'
        f'<div class="mcard-src">{_html.escape(source)}</div>'
        f'</div>'
    )


def _score_ring(score: float) -> str:
    """Return SVG circular gauge HTML for the final score."""
    r = 42
    circ = 2 * math.pi * r
    filled = (score / 10.0) * circ
    c = _color(score)
    return (
        f'<div class="ring-wrap">'
        f'<svg width="120" height="120" viewBox="0 0 100 100">'
        f'<circle cx="50" cy="50" r="{r}" fill="none" stroke="#1c2035" stroke-width="9"/>'
        f'<circle cx="50" cy="50" r="{r}" fill="none" stroke="{c}" stroke-width="9"'
        f' stroke-dasharray="{filled:.2f} {circ - filled:.2f}" stroke-linecap="round"'
        f' transform="rotate(-90 50 50)"/>'
        f'<text x="50" y="46" text-anchor="middle" fill="{c}"'
        f' font-size="20" font-weight="800" font-family="system-ui,sans-serif">{score:.1f}</text>'
        f'<text x="50" y="60" text-anchor="middle" fill="#3a4366"'
        f' font-size="9" font-family="system-ui,sans-serif">/ 10</text>'
        f'</svg>'
        f'</div>'
    )


def _snippet(text: str) -> str:
    """Return HTML for a single pain snippet with source tag and colored border."""
    if "[r/" in text[:12] or "reddit" in text[:25].lower():
        tag, border = '<span class="stag stag-r">Reddit</span>', "#c2410c"
    elif "[hn" in text[:12].lower() or "[ask hn" in text[:15].lower():
        tag, border = '<span class="stag stag-h">HN</span>',     "#d97706"
    else:
        tag, border = '<span class="stag stag-w">Web</span>',    "#2563eb"
    return (
        f'<div class="snip" style="border-left-color:{border}">'
        f'{tag}<span>{_html.escape(text)}</span>'
        f'</div>'
    )


def _formula(report: NicheReport) -> str:
    """Return the score formula breakdown as a monospace HTML block."""
    pain = report.pain_intensity
    sol  = getattr(report, "solutionless_score", 0.0)
    wtp  = getattr(report, "willingness_to_pay",  0.0)
    mom  = getattr(report, "momentum_score",       0.0)
    dens = report.competitive_density
    ai   = report.ai_native_count
    gap  = 10.0 - dens

    components = [
        ("Pain Intensity",     0.30, pain),
        ("Solutionless Rate",  0.25, sol),
        ("Willingness to Pay", 0.20, wtp),
        ("Market Momentum",    0.15, mom),
        ("Market Space",       0.10, gap),
    ]
    raw = sum(w * v for _, w, v in components)
    pen_map = {0: 1.0, 1: 0.90, 2: 0.75, 3: 0.55}
    pen = pen_map.get(ai, 0.40)
    final = round(min(raw * pen, 10.0), 2)

    sep = "─" * 50
    rows = "\n".join(
        f"  {n:<22} × {w:.2f}  ×  {v:<5.1f}  =  {w * v:.2f}"
        for n, w, v in components
    )
    if ai > 0:
        pen_html = (
            f'  <span class="w">× {pen:.2f}  '
            f'({ai} AI-native rival{"s" if ai != 1 else ""} detected)</span>'
        )
    else:
        pen_html = '  <span style="color:#10b981">× 1.00  (no AI-native penalty)</span>'

    return (
        f'<div class="fblock">'
        f"<b>Component              Weight    Score   Contribution</b>\n"
        f"  {sep}\n"
        f"{rows}\n"
        f"  {sep}\n"
        f"  <b>Raw total                               {raw:.2f}</b>\n"
        f"{pen_html}\n"
        f"  {sep}\n"
        f'  <b><span class="g">Final Score                           {final:.2f}</span></b>'
        f"</div>"
    )


def _idea_card(idx: int, idea: ProductIdea) -> str:
    """Return HTML for a single product idea card."""
    e = _html.escape
    return (
        f'<div class="icard">'
        f'<div style="font-size:10px;font-weight:700;letter-spacing:.1em;'
        f'text-transform:uppercase;color:#38436a;margin-bottom:6px">Idea {idx + 1}</div>'
        f'<div class="icard-n">{e(idea.name)}</div>'
        f'<div class="icard-l">{e(idea.one_liner)}</div>'
        f'<div class="icard-fl">Target User</div>    <div class="icard-fv">{e(idea.target_user)}</div>'
        f'<div class="icard-fl">Core Feature</div>   <div class="icard-fv">{e(idea.core_feature)}</div>'
        f'<div class="icard-fl">Differentiation</div><div class="icard-fv">{e(idea.differentiation)}</div>'
        f'<div class="icard-fl">Pricing</div>        <div class="icard-fv">{e(idea.pricing_model)}</div>'
        f'<div class="icard-fl">MVP Scope</div>      <div class="icard-fv">{e(idea.mvp_scope)}</div>'
        f'</div>'
    )


# ─── sidebar ──────────────────────────────────────────────────────────────────


def render_sidebar() -> None:
    with st.sidebar:
        st.image("logo.png", use_column_width=True)
        st.markdown(
            '<div style="font-size:11px;color:#3a4366;margin-top:-8px;margin-bottom:4px;text-align:center">'
            "Niche Intelligence Platform</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        reports = st.session_state.reports
        if not reports:
            st.caption("No results yet. Run an analysis to get started.")
            return

        sorted_r = sorted(reports, key=lambda r: r.final_os_score, reverse=True)
        scores = [r.final_os_score for r in reports]

        # Stats strip
        a, b, c = st.columns(3)
        a.metric("Analyzed", len(reports))
        b.metric("Avg Score", f"{sum(scores) / len(scores):.1f}")
        c.metric("Best", f"{max(scores):.1f}")
        st.divider()

        st.markdown(
            "<div style='font-size:10px;font-weight:700;letter-spacing:.1em;"
            "text-transform:uppercase;color:#3a4366;margin-bottom:10px'>"
            "Results — best first</div>",
            unsafe_allow_html=True,
        )

        for r in sorted_r:
            is_sel = st.session_state.selected == r.niche
            c_col  = _color(r.final_os_score)
            pct    = r.final_os_score / 10 * 100
            nm     = r.niche[:28] + "…" if len(r.niche) > 28 else r.niche
            lbl    = f"{'▶ ' if is_sel else ''}{nm}  ·  {r.final_os_score:.1f}"
            if st.button(
                lbl,
                key=f"sb_{r.niche}",
                use_container_width=True,
                type="primary" if is_sel else "secondary",
            ):
                st.session_state.selected = r.niche
                st.rerun()
            st.markdown(
                f'<div class="sbbar">'
                f'<div class="sbbar-f" style="width:{pct:.0f}%;background:{c_col}"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.divider()
        with st.expander("📋 Comparison table"):
            df = pd.DataFrame([r.to_csv_row() for r in sorted_r])
            st.dataframe(df, use_container_width=True, hide_index=True)

        if st.button("⬇️ Export CSV", use_container_width=True):
            save_to_csv(st.session_state.reports)
            st.success(f"Saved → {OUTPUT_FILE}")


# ─── detail view ──────────────────────────────────────────────────────────────


def render_detail(report: NicheReport) -> None:
    sol         = getattr(report, "solutionless_score", 0.0)
    wtp         = getattr(report, "willingness_to_pay",  0.0)
    mom         = getattr(report, "momentum_score",       0.0)
    hn_ask      = getattr(report, "hn_ask_count",          0)
    reddit_subs = getattr(report, "reddit_subscribers",    0)
    tier_lbl, tier_cls = _tier(report.final_os_score)

    # ── Hero ──────────────────────────────────────────────────────────────
    h_left, h_right = st.columns([4, 1])
    with h_left:
        st.markdown(
            f'<span class="tbadge {tier_cls}">{tier_lbl}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(f"## {report.niche}")
        meta: list[str] = []
        if report.ai_native_count:
            s = "s" if report.ai_native_count != 1 else ""
            meta.append(f"🤖 {report.ai_native_count} AI-native rival{s} detected")
        if reddit_subs:
            meta.append(f"🌐 Reddit community: {_fmt_n(reddit_subs)}")
        if hn_ask:
            meta.append(f"❓ {hn_ask} Ask HN posts")
        if meta:
            st.markdown(
                f"<div style='color:#5a6490;font-size:13px'>"
                + " &nbsp;·&nbsp; ".join(meta)
                + "</div>",
                unsafe_allow_html=True,
            )
    with h_right:
        st.markdown(_score_ring(report.final_os_score), unsafe_allow_html=True)

    with st.expander("📐 Score formula breakdown", expanded=False):
        st.markdown(_formula(report), unsafe_allow_html=True)

    st.divider()

    # ── Primary signals ────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:10px;font-weight:700;letter-spacing:.1em;"
        "text-transform:uppercase;color:#3a4366;margin-bottom:10px'>"
        "Primary Signals — drive 90% of the score</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(_mcard("solutionless", f"{sol:.1f}",                    sol),                   unsafe_allow_html=True)
    c2.markdown(_mcard("wtp",          f"{wtp:.1f}",                    wtp),                   unsafe_allow_html=True)
    c3.markdown(_mcard("momentum",     f"{mom:.1f}",                    mom),                   unsafe_allow_html=True)
    c4.markdown(_mcard("pain",         f"{report.pain_intensity:.1f}",  report.pain_intensity),  unsafe_allow_html=True)

    st.markdown("<div style='margin-top:14px'></div>", unsafe_allow_html=True)

    # ── Supporting signals ─────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:10px;font-weight:700;letter-spacing:.1em;"
        "text-transform:uppercase;color:#3a4366;margin-bottom:10px'>"
        "Supporting Signals — context &amp; validation</div>",
        unsafe_allow_html=True,
    )
    d1, d2, d3, d4 = st.columns(4)

    # Density: lower = better, so invert color logic
    dens_c = (
        "#10b981" if report.competitive_density < 5
        else "#f59e0b" if report.competitive_density < 7
        else "#ef4444"
    )
    d1.markdown(
        _mcard("density", f"{report.competitive_density:.1f}", report.competitive_density, dens_c),
        unsafe_allow_html=True,
    )
    d2.markdown(_mcard("demand", f"{report.demand_score:.1f}", report.demand_score), unsafe_allow_html=True)

    # HN Ask — scale: 0–20 posts = full bar
    hn_pct = min(10.0, hn_ask * 0.5)
    d3.markdown(_mcard("hn_ask", str(hn_ask), hn_pct, "#818cf8"), unsafe_allow_html=True)

    # Reddit subs — log scale for bar
    if reddit_subs:
        rs_pct = min(10.0, math.log10(max(1, reddit_subs)) / 7 * 10)
        d4.markdown(_mcard("reddit_subs", _fmt_n(reddit_subs), rs_pct, "#818cf8"), unsafe_allow_html=True)
    else:
        d4.markdown(_mcard("reddit_subs", "—", None, "#3a4366"), unsafe_allow_html=True)

    st.divider()

    # ── Competitors & Pain Evidence ────────────────────────────────────────
    cl, cr = st.columns(2)
    with cl:
        st.markdown("### 🏢 Competitive Landscape")
        if report.competitors:
            rows = [
                {
                    "Domain":  c.domain,
                    "Founded": str(c.founded_year) if c.founded_year else "—",
                    "Age":     f"{c.age_years:.1f} yr" if c.age_years else "—",
                    "Source":  c.founding_source,
                    "AI":      "🤖" if c.is_ai_native else "",
                }
                for c in report.competitors
            ]
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
                column_config={"AI": st.column_config.TextColumn("", width="small")},
            )
        else:
            st.caption("No competitor data collected.")

    with cr:
        st.markdown("### 💬 Pain Evidence")
        if report.pain_snippets:
            st.markdown(
                "".join(_snippet(s) for s in report.pain_snippets[:8]),
                unsafe_allow_html=True,
            )
        else:
            st.caption("No snippets collected.")

    # ── Product Ideas ──────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 💡 Product Ideas")
    key   = f"ideas__{report.niche}"
    ideas: list[ProductIdea] = st.session_state.get(key, [])

    gen_col, cnt_col = st.columns([4, 1])
    with cnt_col:
        n_ideas = st.selectbox("Count", [2, 3, 4, 5], index=1, key=f"n_{report.niche}")
    with gen_col:
        if st.button(
            "💡 Generate Product Ideas",
            key=f"gen_{report.niche}",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(f"Generating {n_ideas} ideas with GPT-4o…"):
                new_ideas = get_generator().generate(report, n_ideas=n_ideas)
            if new_ideas:
                st.session_state[key] = new_ideas
                st.rerun()
            else:
                st.error("Generation failed — check your OpenAI key or try again.")

    if ideas:
        for i in range(0, len(ideas), 2):
            pair = ideas[i : i + 2]
            cols = st.columns(len(pair))
            for col, (j, idea) in zip(cols, enumerate(pair, i)):
                col.markdown(_idea_card(j, idea), unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)


# ─── main panel ───────────────────────────────────────────────────────────────


def render_main() -> None:
    logo_col, _ = st.columns([2, 5])
    with logo_col:
        st.image("logo.png", use_column_width=True)
    st.caption(
        "Find underserved B2B niches using quantitative signals — "
        "pain intensity, solutionless rate, willingness to pay, and market momentum."
    )

    scorer = get_scorer()
    tab_m, tab_h = st.tabs(["✏️ Manual Analysis", "🌐 Harvest Niches"])

    # ── Manual tab ────────────────────────────────────────────────────────
    with tab_m:
        with st.form("manual_form"):
            raw = st.text_area(
                "Niches to analyze — one per line",
                placeholder="fleet maintenance scheduling\nHVAC contractor invoicing\nB2B AP automation",
                height=110,
            )
            run = st.form_submit_button("▶ Run Analysis", use_container_width=True, type="primary")

        if run:
            niches = [n.strip() for n in raw.splitlines() if n.strip()]
            if not niches:
                st.warning("Enter at least one niche.")
            else:
                prog = st.progress(0.0, text="Starting…")
                new: list[NicheReport] = []
                for i, niche in enumerate(niches):
                    prog.progress(i / len(niches), text=f"Analyzing: {niche}")
                    with st.spinner(f"Analyzing '{niche}'…"):
                        r = _run_single(niche, scorer)
                    if r:
                        new.append(r)
                        _merge(r)
                prog.progress(1.0, text="Done!")
                if new:
                    save_to_csv(st.session_state.reports)
                    st.session_state.selected = new[-1].niche
                    st.rerun()

    # ── Harvest tab ───────────────────────────────────────────────────────
    with tab_h:
        with st.form("harvest_form"):
            seeds_raw = st.text_input(
                "Seed industries (comma-separated)",
                placeholder="construction, HR, logistics, finance",
            )
            max_ideas = st.slider("Max ideas to harvest", 5, 20, 10)
            run_h = st.form_submit_button(
                "🌐 Harvest + Analyze", use_container_width=True, type="primary"
            )

        if run_h:
            seeds = [s.strip() for s in seeds_raw.split(",") if s.strip()]
            buf   = io.StringIO()
            with st.spinner("Mining Reddit, G2, HN, and job boards for pain signals…"):
                with contextlib.redirect_stdout(buf):
                    harvester = IdeaHarvester(scorer.scraper)
                    ideas = harvester.harvest(seed_industries=seeds or None, max_ideas=max_ideas)
            if not ideas:
                st.warning("Harvester found no ideas. Try adding seed industries.")
            else:
                niches = [idea.niche for idea in ideas]
                st.info(f"Harvested {len(niches)} candidates — now scoring…")
                prog = st.progress(0.0, text="Starting…")
                new = []
                for i, niche in enumerate(niches):
                    prog.progress(i / len(niches), text=f"Analyzing: {niche}")
                    with st.spinner(f"Analyzing '{niche}'…"):
                        r = _run_single(niche, scorer)
                    if r:
                        new.append(r)
                        _merge(r)
                prog.progress(1.0, text="Done!")
                if new:
                    save_to_csv(st.session_state.reports)
                    st.session_state.selected = new[-1].niche
                    st.rerun()

    st.divider()

    # ── Detail view or landing page ────────────────────────────────────────
    if st.session_state.selected:
        report = _get_report(st.session_state.selected)
        if report:
            render_detail(report)
        else:
            st.info("Select a result from the sidebar.")
    elif st.session_state.reports:
        st.info("← Select a result from the sidebar to view its breakdown.")
    else:
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        features = [
            ("🔍", "Pain Scoring",
             "GPT-4o-mini rates financial impact with calibrated dollar-band anchors — "
             "no vanity sentiment, just cost-of-problem estimates."),
            ("🏔️", "Gap Detection",
             "Solutionless Rate measures how many pain posts have no existing tool. "
             "High rate = underserved market."),
            ("📈", "Real Trend Data",
             "Google Trends weekly data (5-year history) replaces crude Serper result "
             "counts for accurate momentum scoring."),
            ("💡", "Product Ideas",
             "GPT-4o generates validated, anti-pattern-free product ideas grounded in "
             "real pain evidence and competitive gaps."),
        ]
        cols = st.columns(4)
        for col, (icon, title, desc) in zip(cols, features):
            col.markdown(
                f'<div class="lcard">'
                f'<div class="lcard-icon">{icon}</div>'
                f'<div class="lcard-title">{title}</div>'
                f'<div class="lcard-desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    if st.session_state.log:
        with st.expander("📋 Analysis log", expanded=False):
            st.code(st.session_state.log, language=None)


# ─── render ───────────────────────────────────────────────────────────────────

render_sidebar()
render_main()
