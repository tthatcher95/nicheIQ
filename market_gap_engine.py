"""
Quantitative Market-Gap Engine
Calculates an Opportunity Score for any keyword/industry.

Requirements:
    pip install requests python-dotenv openai whois python-dateutil

Environment variables (.env):
    SERPER_API_KEY=<your key>
    OPENAI_API_KEY=<your key>   # used for sentiment scoring; swap for any LLM
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests
import whois
from dateutil.parser import parse as date_parse
from dotenv import load_dotenv
from openai import OpenAI

try:
    from pytrends.request import TrendReq
    _PYTRENDS_AVAILABLE = True
except ImportError:
    _PYTRENDS_AVAILABLE = False

try:
    import praw
    _PRAW_AVAILABLE = True
except ImportError:
    _PRAW_AVAILABLE = False

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "MarketGapEngine/1.0 (research)")

SERPER_ENDPOINT = "https://google.serper.dev/search"
OUTPUT_FILE = "market_analysis.csv"

# AI-native startup penalty: if >= this many AI-native competitors found -> x0.5
AI_STARTUP_THRESHOLD = 3

# Year range considered "AI-native" (founded during this period)
AI_NATIVE_START = 2024
AI_NATIVE_END = 2026

# How many competitors to consider for the "Competitive Age" metric
TOP_N_FOR_AGE = 5

CSV_COLUMNS = [
    "Niche",
    "Pain_Intensity",
    "Solutionless_Score",
    "Willingness_To_Pay",
    "Momentum_Score",
    "Competitive_Density",
    "AI_Native_Count",
    "Reddit_Subscribers",
    "HN_Ask_Count",
    "Demand_Score",
    "Final_OS_Score",
]

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

# Words in pain snippets that indicate real, costly frustration
_HIGH_SIGNAL_PAIN_WORDS: frozenset[str] = frozenset({
    "frustrated", "nightmare", "painful", "hate", "broken", "terrible",
    "manually", "spreadsheet", "waste of time", "no solution", "i wish",
    "switching", "canceling", "unusable", "expensive", "overpriced",
    "can't find", "stuck with", "forced to", "workaround", "hair on fire",
    "kills productivity", "costing us", "losing money",
})

# Domains treated as content/media, NOT as dedicated SaaS competitors
_CONTENT_DOMAINS: frozenset[str] = frozenset({
    "reddit.com", "youtube.com", "medium.com", "linkedin.com",
    "forbes.com", "techcrunch.com", "wikipedia.org", "quora.com",
    "twitter.com", "x.com", "news.ycombinator.com", "substack.com",
    "hubspot.com", "semrush.com", "g2.com", "capterra.com", "getapp.com",
    "towardsai.net", "towardsdatascience.com", "hackernoon.com",
    "dev.to", "stackoverflow.com", "producthunt.com",
})

# Snippet/URL keywords that signal a real SaaS product page
_SAAS_SNIPPET_SIGNALS: frozenset[str] = frozenset({
    "pricing", "free trial", "sign up", "get started", "per month",
    "per user", "annual plan", "enterprise plan", "schedule a demo",
    "start for free", "try for free",
})

# Forum domains where real users complain — boosts demand quality signal
_FORUM_DOMAINS: frozenset[str] = frozenset({
    "reddit.com", "news.ycombinator.com", "stackexchange.com",
    "quora.com",
})

# Regex patterns that indicate existing spend / budget awareness in snippets
_PRICE_PATTERNS: list[re.Pattern] = [
    re.compile(r'\$[\d,]+(?:\s*(?:/mo|/month|per month|/yr|/year|per year))?', re.I),
    re.compile(r'pay(?:ing)?\s+\$[\d,]+', re.I),
    re.compile(r'costs?\s+(?:us\s+)?\$[\d,]+', re.I),
    re.compile(r'spend(?:ing)?\s+\$[\d,]+', re.I),
    re.compile(r'budget(?:\s+of)?\s+\$', re.I),
    re.compile(r'invoic(?:e|ing)\s+for\s+\$', re.I),
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str
    position: int


@dataclass
class CompetitorInfo:
    domain: str
    creation_date: Optional[datetime] = None
    age_years: float = 0.0
    is_ai_native: bool = False
    # DetectionLayer fields (populated after enrichment)
    founded_year: Optional[int] = None
    founding_source: str = "unknown"   # whois | crunchbase | linkedin | web | unknown
    ai_signals: list[str] = field(default_factory=list)
    detection_confidence: float = 0.0  # 0.0–1.0


@dataclass
class DetectionResult:
    domain: str
    founded_year: Optional[int] = None
    founding_source: str = "unknown"
    ai_signals: list[str] = field(default_factory=list)
    confidence: float = 0.0
    is_ai_native: bool = False


@dataclass
class ProductIdea:
    name: str
    one_liner: str
    target_user: str
    core_feature: str
    differentiation: str  # vs existing competitors
    pricing_model: str
    mvp_scope: str         # smallest shippable version in 4-6 weeks


@dataclass
class HarvestedIdea:
    niche: str
    pain: str
    mention_count: int = 1
    sources: list[str] = field(default_factory=list)   # reddit | g2 | linkedin | producthunt
    evidence: list[str] = field(default_factory=list)  # raw snippets that led to this idea


@dataclass
class NicheReport:
    niche: str
    pain_snippets: list[str] = field(default_factory=list)
    competitors: list[CompetitorInfo] = field(default_factory=list)

    # Computed scores (0-10 each)
    demand_score: float = 0.0         # Quality-weighted pain discussion volume
    pain_intensity: float = 0.0       # LLM: financial/time cost of the problem
    competitive_density: float = 0.0  # SaaS saturation + competitor age blend
    ai_native_count: int = 0          # Dedicated AI-native tools found on page 1

    # Underserved-market signals
    solutionless_score: float = 0.0   # % of pain posts with no tool mentioned — the core "underserved" signal
    willingness_to_pay: float = 0.0   # Job posting proxy + price anchors — budget already exists
    momentum_score: float = 0.0       # Google Trends / Serper momentum — growing vs shrinking problem

    # External community signals (populated when Reddit/HN enabled)
    reddit_subscribers: int = 0       # Sum of subscribers in top 3 relevant subreddits (TAM proxy)
    hn_ask_count: int = 0             # "Ask HN: is there a tool?" posts — direct solutionless evidence

    @property
    def final_os_score(self) -> float:
        """
        Reweighted for underserved-market detection:
          30% Pain Intensity       — does it hurt enough to pay?
          25% Solutionless Score   — are people stranded without a tool?
          20% Willingness to Pay   — is there existing budget?
          15% Momentum Score       — is the problem growing?
          10% Comp. Density        — is there market space? (inverted)

        Demand Score is kept for reference but not in the formula —
        Solutionless + Momentum capture the same signal more precisely.

        Graduated AI-native penalty:
          1 → ×0.90  |  2 → ×0.75  |  3 → ×0.55  |  4+ → ×0.40
        """
        raw = (
            0.30 * self.pain_intensity
            + 0.25 * getattr(self, "solutionless_score", 0.0)
            + 0.20 * getattr(self, "willingness_to_pay", 0.0)
            + 0.15 * getattr(self, "momentum_score", 0.0)
            + 0.10 * (10.0 - self.competitive_density)
        )
        if self.ai_native_count >= 4:
            raw *= 0.40
        elif self.ai_native_count == 3:
            raw *= 0.55
        elif self.ai_native_count == 2:
            raw *= 0.75
        elif self.ai_native_count == 1:
            raw *= 0.90
        return round(min(raw, 10.0), 2)

    def to_csv_row(self) -> dict:
        return {
            "Niche": self.niche,
            "Pain_Intensity": round(self.pain_intensity, 2),
            "Solutionless_Score": round(getattr(self, "solutionless_score", 0.0), 2),
            "Willingness_To_Pay": round(getattr(self, "willingness_to_pay", 0.0), 2),
            "Momentum_Score": round(getattr(self, "momentum_score", 0.0), 2),
            "Competitive_Density": round(self.competitive_density, 2),
            "AI_Native_Count": self.ai_native_count,
            "Reddit_Subscribers": getattr(self, "reddit_subscribers", 0),
            "HN_Ask_Count": getattr(self, "hn_ask_count", 0),
            "Demand_Score": round(self.demand_score, 2),
            "Final_OS_Score": self.final_os_score,
        }


# ---------------------------------------------------------------------------
# MarketScraper - API hooks & raw data collection
# ---------------------------------------------------------------------------


class MarketScraper:
    """
    Handles all outbound requests:
      - Serper.dev searches (problem discussions + organic competitors)
      - WHOIS domain age lookups
    """

    def __init__(self, serper_key: str = SERPER_API_KEY) -> None:
        if not serper_key:
            raise ValueError("SERPER_API_KEY is not set. Add it to your .env file.")
        self._key = serper_key
        self._session = requests.Session()
        self._session.headers.update({
            "X-API-KEY": self._key,
            "Content-Type": "application/json",
        })

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_pain_discussions(self, keyword: str, num: int = 10) -> list[SearchResult]:
        """
        (A) Search for recent "How to" / "Problem with" discussions.
        Returns up to `num` SearchResult objects.
        """
        queries = [
            f'"how to" {keyword} problem',
            f'"problem with" {keyword}',
            f'{keyword} "not working" OR "struggling with" OR "frustrated"',
        ]
        results: list[SearchResult] = []
        for query in queries:
            raw = self._serper_search(query, num=num)
            results.extend(self._parse_organic(raw))
            if len(results) >= num * 2:
                break
            time.sleep(0.3)  # gentle rate-limit
        return results[:num * 2]

    def fetch_competitors(self, keyword: str, num: int = 10) -> list[SearchResult]:
        """
        (B) Top-N organic competitors for the keyword.
        """
        raw = self._serper_search(keyword, num=num)
        return self._parse_organic(raw)[:num]

    def fetch_domain_age(self, domain: str) -> Optional[datetime]:
        """
        WHOIS lookup -> returns the domain creation date (UTC) or None on failure.
        """
        try:
            w = whois.whois(domain)
            created = w.creation_date
            if isinstance(created, list):
                created = created[0]
            if isinstance(created, datetime):
                return created.replace(tzinfo=timezone.utc) if created.tzinfo is None else created
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def fetch_site_snippets(self, domain: str, num: int = 5) -> list[SearchResult]:
        """Homepage content search — no time filter, used for AI-signal detection."""
        raw = self._serper_search(f"site:{domain}", num=num, tbs=None)
        return self._parse_organic(raw)[:num]

    def _serper_search(self, query: str, num: int = 10, tbs: Optional[str] = "qdr:y") -> dict:
        payload: dict = {
            "q": query,
            "num": num,
            "gl": "us",
            "hl": "en",
        }
        if tbs:
            payload["tbs"] = tbs
        response = self._session.post(SERPER_ENDPOINT, json=payload, timeout=15)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _parse_organic(raw: dict) -> list[SearchResult]:
        results = []
        for item in raw.get("organic", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                link=item.get("link", ""),
                snippet=item.get("snippet", ""),
                position=item.get("position", 0),
            ))
        return results


# ---------------------------------------------------------------------------
# DetectionLayer - multi-signal AI-native startup detector
# ---------------------------------------------------------------------------


class DetectionLayer:
    """
    Determines whether a competitor is a true AI-native startup using a
    waterfall of evidence sources, rather than relying solely on WHOIS date.

    Waterfall (stops at first confident hit):
      1. Hint snippet  — the search result snippet already in hand (free)
      2. Crunchbase    — site:crunchbase.com search via Serper (confidence 0.90)
      3. LinkedIn      — site:linkedin.com/company search     (confidence 0.80)
      4. General web   — broad "founded in" query             (confidence 0.60)
      5. WHOIS fallback — domain registration year            (confidence 0.40)

    AI-signal detection uses the accumulated snippets + a site: homepage search
    and scans for AI-related keywords.  A competitor is flagged `is_ai_native`
    only when BOTH conditions hold:
      - founded_year in [AI_NATIVE_START, AI_NATIVE_END]
      - at least one AI signal found
    """

    # ---- Founding-year regex patterns (ordered most→least specific) --------
    _YEAR_PATTERNS: list[re.Pattern] = [
        re.compile(r"founded\s+(?:in\s+)?(\d{4})", re.I),
        re.compile(r"launched\s+(?:in\s+)?(\d{4})", re.I),
        re.compile(r"established\s+(?:in\s+)?(\d{4})", re.I),
        re.compile(r"started\s+(?:in\s+)?(\d{4})", re.I),
        re.compile(r"incorporated\s+(?:in\s+)?(\d{4})", re.I),
        re.compile(r"·\s*(\d{4})\s*·"),           # Crunchbase: "· 2024 ·"
        re.compile(r"(\d{4})\s*[–-]\s*present", re.I),   # LinkedIn: "2024 – present"
        re.compile(r"since\s+(\d{4})", re.I),
        re.compile(r"from\s+(\d{4})", re.I),
    ]

    # ---- AI keyword patterns -----------------------------------------------
    _AI_KEYWORDS: list[re.Pattern] = [
        re.compile(r"\bai\b", re.I),
        re.compile(r"\bllm\b", re.I),
        re.compile(r"\bgpt\b", re.I),
        re.compile(r"\bnlp\b", re.I),
        re.compile(r"artificial intelligence", re.I),
        re.compile(r"machine learning", re.I),
        re.compile(r"large language model", re.I),
        re.compile(r"generative ai", re.I),
        re.compile(r"gen[- ]ai", re.I),
        re.compile(r"deep learning", re.I),
        re.compile(r"neural network", re.I),
        re.compile(r"\bcopilot\b", re.I),
        re.compile(r"\bchatgpt\b", re.I),
        re.compile(r"\bopenai\b", re.I),
        re.compile(r"\banthropic\b", re.I),
        re.compile(r"ai[- ]native", re.I),
        re.compile(r"ai[- ]first", re.I),
        re.compile(r"ai[- ]powered", re.I),
        re.compile(r"powered by ai", re.I),
        re.compile(r"built (?:on|with) ai", re.I),
    ]

    # Reasonable year bounds — reject obvious noise
    _YEAR_MIN = 1990
    _YEAR_MAX = datetime.now(tz=timezone.utc).year

    def __init__(self, scraper: "MarketScraper") -> None:
        self._scraper = scraper
        self._cache: dict[str, DetectionResult] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def enrich(self, info: CompetitorInfo, hint_snippet: str = "") -> CompetitorInfo:
        """
        Enrich a CompetitorInfo in-place with founding year + AI signals.
        `hint_snippet` is whatever text we already have from the SERP result
        (free — no extra API call needed).
        """
        result = self._detect(info.domain, hint_snippet, whois_year=_year_from_date(info.creation_date))
        info.founded_year = result.founded_year
        info.founding_source = result.founding_source
        info.ai_signals = result.ai_signals
        info.detection_confidence = result.confidence
        # Override is_ai_native with the richer signal
        info.is_ai_native = result.is_ai_native
        return info

    # ------------------------------------------------------------------
    # Detection waterfall
    # ------------------------------------------------------------------

    def _detect(
        self,
        domain: str,
        hint_snippet: str = "",
        whois_year: Optional[int] = None,
    ) -> DetectionResult:
        if domain in self._cache:
            return self._cache[domain]

        company = domain.split(".")[0]
        founded_year: Optional[int] = None
        source = "unknown"
        confidence = 0.0
        all_snippets: list[str] = []

        if hint_snippet:
            all_snippets.append(hint_snippet)

        # --- Strategy 1: hint snippet (free) --------------------------------
        year = self._extract_year(hint_snippet)
        if year:
            founded_year, source, confidence = year, "hint_snippet", 0.55

        # --- Strategy 2: Crunchbase -----------------------------------------
        if not founded_year:
            snippets = self._search_snippets(
                f'site:crunchbase.com/organization "{company}" founded', num=3, tbs=None
            )
            all_snippets.extend(snippets)
            year = self._extract_year(" ".join(snippets))
            if year:
                founded_year, source, confidence = year, "crunchbase", 0.90
            time.sleep(0.25)

        # --- Strategy 3: LinkedIn -------------------------------------------
        if not founded_year:
            snippets = self._search_snippets(
                f'site:linkedin.com/company "{company}" founded', num=3, tbs=None
            )
            all_snippets.extend(snippets)
            year = self._extract_year(" ".join(snippets))
            if year:
                founded_year, source, confidence = year, "linkedin", 0.80
            time.sleep(0.25)

        # --- Strategy 4: General web ----------------------------------------
        if not founded_year:
            snippets = self._search_snippets(
                f'"{domain}" OR "{company}" "founded in" OR "founded:"', num=5, tbs=None
            )
            all_snippets.extend(snippets)
            year = self._extract_year(" ".join(snippets))
            if year:
                founded_year, source, confidence = year, "web", 0.60
            time.sleep(0.25)

        # --- Strategy 5: WHOIS fallback -------------------------------------
        if not founded_year and whois_year:
            founded_year, source, confidence = whois_year, "whois", 0.40

        # --- AI signal scan -------------------------------------------------
        # Pull site: snippets for the homepage (already-fetched snippets reused too)
        site_results = self._scraper.fetch_site_snippets(domain, num=5)
        all_snippets.extend(r.snippet for r in site_results if r.snippet)
        combined_text = " ".join(all_snippets)
        ai_signals = self._extract_ai_signals(combined_text)

        # Boost confidence slightly if AI signals corroborate the year
        if ai_signals and founded_year and AI_NATIVE_START <= founded_year <= AI_NATIVE_END:
            confidence = min(1.0, confidence + 0.05 * len(ai_signals))

        is_ai_native = bool(
            founded_year
            and AI_NATIVE_START <= founded_year <= AI_NATIVE_END
            and ai_signals
        )

        result = DetectionResult(
            domain=domain,
            founded_year=founded_year,
            founding_source=source,
            ai_signals=ai_signals,
            confidence=round(confidence, 2),
            is_ai_native=is_ai_native,
        )
        self._cache[domain] = result
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _search_snippets(self, query: str, num: int = 5, tbs: Optional[str] = None) -> list[str]:
        try:
            raw = self._scraper._serper_search(query, num=num, tbs=tbs)
            return [
                item.get("snippet", "")
                for item in raw.get("organic", [])
                if item.get("snippet")
            ]
        except Exception:
            return []

    def _extract_year(self, text: str) -> Optional[int]:
        """Return the first plausible founding year found in `text`, or None."""
        for pattern in self._YEAR_PATTERNS:
            m = pattern.search(text)
            if m:
                year = int(m.group(1))
                if self._YEAR_MIN <= year <= self._YEAR_MAX:
                    return year
        return None

    def _extract_ai_signals(self, text: str) -> list[str]:
        """Return deduplicated list of AI keyword matches found in `text`."""
        found: list[str] = []
        seen: set[str] = set()
        for pattern in self._AI_KEYWORDS:
            m = pattern.search(text)
            if m:
                label = m.group(0).lower()
                if label not in seen:
                    seen.add(label)
                    found.append(m.group(0))
        return found


# ---------------------------------------------------------------------------
# SentimentAnalyzer - LLM-based pain scoring
# ---------------------------------------------------------------------------


class SentimentAnalyzer:
    """
    Uses an OpenAI-compatible LLM to rate how financially painful a problem is.
    Swap `client` for any other provider (Groq, Ollama, etc.) as needed.
    """

    SYSTEM_PROMPT = (
        "You are a market research analyst scoring business problems by financial impact.\n"
        "Rate how much this problem costs the average affected user in time, money, or lost revenue.\n\n"
        "Calibration anchors — use these as reference points:\n"
        "  2  Minor annoyance, no real cost ('I wish my app had dark mode')\n"
        "  4  Adds 1-2 hrs/week of manual work, low dollar impact\n"
        "  5  Costs ~$200-500/month in wasted time or missed revenue\n"
        "  6  Costs $500-2000/month or causes regular customer complaints\n"
        "  8  Costs $5000+/month, creates compliance risk, or loses enterprise deals\n"
        " 10  Existential: regulatory shutdown, client churn, or business failure risk\n\n"
        "Most real B2B problems land between 4-7. Reserve 8-10 for severe, costly pain.\n"
        "Respond with ONLY a single integer between 1 and 10."
    )

    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = "gpt-4o-mini") -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def score_pain(self, niche: str, snippets: list[str]) -> float:
        """Returns a 1-10 pain score for the given niche + evidence snippets."""
        evidence = "\n".join(f"- {s}" for s in snippets[:10])
        user_msg = (
            f"Niche: {niche}\n\n"
            f"Evidence snippets from online discussions:\n{evidence}\n\n"
            "Rate the pain level (1-10):"
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=5,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            score = float(re.search(r"\d+", raw).group())
            return max(1.0, min(10.0, score))
        except Exception as e:
            print(f"  [warn] LLM scoring failed ({e}); defaulting to 5.0")
            return 5.0


# ---------------------------------------------------------------------------
# TrendsSignal — Google Trends momentum via pytrends
# ---------------------------------------------------------------------------


class TrendsSignal:
    """
    Wraps pytrends to provide real Google search-interest data.
    Replaces the Serper-count-based momentum proxy with actual weekly trend data.

    Requires: pip install pytrends  (no API key needed)

    score_momentum() compares last-12-week avg vs trailing-52-week avg.
    rising_queries() surfaces adjacent trending topics — useful for IdeaHarvester.
    """

    def __init__(self) -> None:
        if not _PYTRENDS_AVAILABLE:
            raise RuntimeError("pytrends not installed. Run: pip install pytrends")
        # backoff_factor=0.5 retries on 429s; timeout=(connect, read)
        self._pt = TrendReq(hl="en-US", tz=360, timeout=(10, 25), retries=2, backoff_factor=0.5)

    def score_momentum(self, keyword: str) -> Optional[float]:
        """
        Compare last-12-week average vs trailing-52-week average.
        Returns 0–10 (5.0 = flat, >5 = growing, <5 = declining).
        Returns None if Trends has no data or is rate-limited — caller falls back to Serper.
        """
        try:
            kw = keyword[:100]
            self._pt.build_payload([kw], timeframe="today 5-y", geo="US")
            df = self._pt.interest_over_time()
            if df.empty or kw not in df.columns:
                return None
            series = df[kw].astype(float).values
            if len(series) < 26:
                return None
            recent_avg = float(series[-12:].mean())   # last ~3 months (weekly data)
            annual_avg = float(series[-52:].mean())   # last 1 year
            if annual_avg < 1.0:
                return None
            ratio = recent_avg / annual_avg
            return round(min(10.0, max(0.0, ratio * 5.0)), 2)
        except Exception as e:
            print(f"  [warn] Google Trends score_momentum failed: {e}")
            return None

    def rising_queries(self, keyword: str) -> list[str]:
        """
        Rising related queries over the last 12 months.
        Surfaces adjacent trending problems — used by IdeaHarvester to discover new niches.
        Returns up to 8 query strings, or [] on failure.
        """
        try:
            kw = keyword[:100]
            self._pt.build_payload([kw], timeframe="today 12-m", geo="US")
            related = self._pt.related_queries()
            rising_df = related.get(kw, {}).get("rising")
            if rising_df is not None and not rising_df.empty:
                return rising_df["query"].tolist()[:8]
        except Exception:
            pass
        return []


# ---------------------------------------------------------------------------
# HNSignal — Hacker News Algolia API (free, no key required)
# ---------------------------------------------------------------------------


class HNSignal:
    """
    Queries the HN Algolia search API — free, no key required.

    Why HN is high-signal for B2B niches:
      • "Ask HN: is there a tool for X?" posts are a direct solutionless signal —
        someone publicly asking for a product they can't find.
      • Audience skews toward technical founders and B2B decision-makers.
      • Upvote counts validate community interest.

    Endpoints used:
      https://hn.algolia.com/api/v1/search       — relevance-ranked
      https://hn.algolia.com/api/v1/search_by_date — recency-ranked
    """

    _SEARCH_URL = "https://hn.algolia.com/api/v1/search"

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "MarketGapEngine/1.0 (research)"})

    def ask_hn_count(self, keyword: str) -> int:
        """
        Count of 'Ask HN' posts that mention this keyword.
        Each post = someone publicly asking for a solution they couldn't find.
        """
        try:
            resp = self._session.get(
                self._SEARCH_URL,
                params={"query": keyword, "tags": "ask_hn", "hitsPerPage": 1},
                timeout=8,
            )
            resp.raise_for_status()
            return int(resp.json().get("nbHits", 0))
        except Exception:
            return 0

    def get_pain_snippets(self, keyword: str, n: int = 10) -> list[str]:
        """
        High-quality pain snippets from HN stories, sorted by upvote score.
        Prefixed with "[HN Xpts]" so LLM scoring can weight them appropriately.
        Only includes posts with ≥5 points (community-validated signal).
        """
        try:
            resp = self._session.get(
                self._SEARCH_URL,
                params={"query": keyword, "tags": "story", "hitsPerPage": n * 2},
                timeout=8,
            )
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
            snippets: list[str] = []
            for hit in sorted(hits, key=lambda h: h.get("points") or 0, reverse=True)[:n]:
                pts = hit.get("points") or 0
                title = hit.get("title", "")
                if title and pts >= 5:
                    snippets.append(f"[HN {pts}pts] {title}")
                text = (hit.get("story_text") or "").strip()
                if text and len(text) > 30:
                    snippets.append(text[:250])
            return snippets[:n]
        except Exception as e:
            print(f"  [warn] HN pain snippets failed: {e}")
            return []

    def get_idea_signals(self, seed: str, n: int = 5) -> list[str]:
        """
        Fetch 'Ask HN: is there a tool / looking for software' posts for a seed.
        Used by IdeaHarvester to surface explicit unsatisfied demand.
        """
        queries = [
            f"is there a tool {seed}",
            f"looking for software {seed}",
            f"does anyone know {seed}",
        ]
        snippets: list[str] = []
        for q in queries:
            try:
                resp = self._session.get(
                    self._SEARCH_URL,
                    params={"query": q, "tags": "ask_hn", "hitsPerPage": n},
                    timeout=8,
                )
                resp.raise_for_status()
                for hit in resp.json().get("hits", []):
                    title = hit.get("title", "")
                    pts = hit.get("points") or 0
                    if title:
                        snippets.append(f"[Ask HN, {pts}pts] {title}")
                time.sleep(0.15)
            except Exception:
                pass
        return snippets[:n * 2]


# ---------------------------------------------------------------------------
# RedditSignal — PRAW (Reddit API, read-only)
# ---------------------------------------------------------------------------


class RedditSignal:
    """
    Uses the Reddit API (via PRAW, read-only) for two signals:

      1. Upvote-weighted pain snippets
         More representative than Serper site:reddit queries because we get
         actual upvote counts (community validation) and can filter by top/year.

      2. Subreddit subscriber counts — TAM proxy
         Summing subscribers of the top 3 most relevant subreddits gives a
         rough community-size signal: r/smallbusiness (1M) vs r/hvac (50k)
         tells you something real about the addressable market.

    Setup:
      1. reddit.com/prefs/apps → Create App → choose "script"
      2. Add to .env:
           REDDIT_CLIENT_ID=<client_id shown under app name>
           REDDIT_CLIENT_SECRET=<secret>
    """

    def __init__(
        self,
        client_id: str = REDDIT_CLIENT_ID,
        client_secret: str = REDDIT_CLIENT_SECRET,
        user_agent: str = REDDIT_USER_AGENT,
    ) -> None:
        if not _PRAW_AVAILABLE:
            raise RuntimeError("praw not installed. Run: pip install praw")
        if not client_id or not client_secret:
            raise ValueError(
                "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set in .env"
            )
        self._reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
            read_only=True,
        )

    def community_size(self, keyword: str) -> int:
        """
        Sum of subscribers from the top 3 most relevant subreddits.
        A community of 500k around a pain = real addressable market.
        Returns 0 on failure.
        """
        total = 0
        try:
            for sub in self._reddit.subreddits.search(keyword, limit=3):
                total += sub.subscribers or 0
        except Exception as e:
            print(f"  [warn] Reddit community_size failed: {e}")
        return total

    def get_pain_snippets(self, keyword: str, n: int = 15) -> list[str]:
        """
        Search r/all for posts matching the keyword, sorted by upvote score.
        Returns snippets prefixed with "[r/subreddit, Xup]" for LLM context.
        Highest-validated pain (most upvotes) comes first.
        """
        queries = [
            f"{keyword} problem OR frustrated OR nightmare OR manually",
            f"{keyword} software tool alternative",
        ]
        posts: list[dict] = []
        seen: set[str] = set()
        for query in queries:
            try:
                for post in self._reddit.subreddit("all").search(
                    query, sort="top", time_filter="year", limit=n
                ):
                    if post.id not in seen:
                        seen.add(post.id)
                        posts.append({
                            "title": post.title,
                            "text": (post.selftext or "")[:300],
                            "score": post.score,
                            "subreddit": post.subreddit.display_name,
                        })
                time.sleep(0.3)
            except Exception as e:
                print(f"  [warn] Reddit search failed: {e}")
        posts.sort(key=lambda x: x["score"], reverse=True)
        snippets: list[str] = []
        for p in posts[:n]:
            snippets.append(f"[r/{p['subreddit']}, {p['score']}↑] {p['title']}")
            if p["text"] and len(p["text"].strip()) > 30:
                snippets.append(p["text"].strip()[:200])
        return snippets[:n]


# ---------------------------------------------------------------------------
# IdeaHarvester - mines pain signals and generates niche candidates
# ---------------------------------------------------------------------------


class IdeaHarvester:
    """
    Turns the engine into a true end-to-end pipeline by auto-generating niche
    ideas from internet pain signals before they hit OpportunityScorer.

    Pipeline:
      1. Collect  — Serper queries across Reddit, G2, job boards, ProductHunt
      2. Extract  — LLM reads batched snippets and pulls out niche candidates
      3. Consolidate — LLM deduplicates/normalizes and returns a ranked list

    Usage:
        harvester = IdeaHarvester(scraper)
        ideas = harvester.harvest(seed_industries=["HR", "finance", "operations"])
        niches = [idea.niche for idea in ideas]   # feed into OpportunityScorer
    """

    # ---- Per-source Serper query templates ({seed} is replaced at runtime) --
    _SOURCE_QUERIES: dict[str, list[str]] = {
        "reddit": [
            'site:reddit.com "I wish there was a tool" {seed}',
            'site:reddit.com "why isn\'t there software" {seed}',
            'site:reddit.com "still doing this manually" {seed}',
            'site:reddit.com "does anyone know of a tool" {seed}',
            'site:reddit.com {seed} "nightmare" OR "painful" OR "waste of time" software',
        ],
        "g2": [
            'site:g2.com/reviews {seed} "cons" "manual" OR "workaround" OR "missing"',
            'site:g2.com/reviews {seed} "wish it could" OR "doesn\'t integrate"',
        ],
        "job_boards": [
            'site:linkedin.com/jobs {seed} "manage spreadsheets" OR "manual data entry" OR "track manually"',
            'site:indeed.com {seed} coordinator specialist "organize" OR "track" OR "manage"',
        ],
        "producthunt": [
            'site:producthunt.com {seed} "looking for" OR "alternative" OR "wish it had"',
        ],
        "hn": [
            'site:news.ycombinator.com "Ask HN" "is there a tool" {seed}',
            'site:news.ycombinator.com "Ask HN" {seed} software OR SaaS',
        ],
    }

    # ---- Broad fallback queries when no seeds are given ---------------------
    _BROAD_QUERIES: list[tuple[str, str]] = [
        ("reddit",      'site:reddit.com "I wish there was a tool" software OR SaaS'),
        ("reddit",      'site:reddit.com "why isn\'t there software" B2B'),
        ("reddit",      'site:reddit.com "we still do this manually" small business'),
        ("g2",          'site:g2.com/reviews "biggest con" "still have to manually"'),
        ("reddit",      'site:reddit.com "can\'t find a good solution" operations OR finance OR HR'),
        ("producthunt", 'site:producthunt.com "looking for an alternative" tool OR app'),
        ("job_boards",  'site:linkedin.com/jobs "manage spreadsheets" OR "no software exists" specialist'),
        ("hn",          'site:news.ycombinator.com "Ask HN" "is there a tool" software'),
        ("hn",          'site:news.ycombinator.com "Ask HN" "looking for" software B2B'),
    ]

    _EXTRACTION_SYSTEM_PROMPT = (
        "You are a market research analyst specializing in software opportunities. "
        "Given snippets from online discussions, extract specific software/tool niches "
        "where people clearly have pain or are explicitly asking for a solution.\n\n"
        "Return a JSON object with key \"niches\" containing an array where each item has:\n"
        "  - \"niche\": 2-6 word specific software category "
        "(e.g. 'construction site daily reporting', NOT 'project management')\n"
        "  - \"pain\": one sentence describing the core problem\n"
        "  - \"mention_count\": integer — how many snippets relate to this niche\n\n"
        "Rules: only include niches with CLEAR pain or explicit demand. "
        "Normalize near-duplicates. If nothing qualifies, return {\"niches\": []}. "
        "Return ONLY valid JSON."
    )

    def __init__(self, scraper: "MarketScraper", api_key: str = OPENAI_API_KEY) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
        self._scraper = scraper
        self._client = OpenAI(api_key=api_key)
        self._hn = HNSignal()
        self._trends: Optional[TrendsSignal] = None
        if _PYTRENDS_AVAILABLE:
            try:
                self._trends = TrendsSignal()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def harvest(
        self,
        seed_industries: Optional[list[str]] = None,
        max_ideas: int = 10,
    ) -> list[HarvestedIdea]:
        """
        Full pipeline: collect → extract → consolidate → rank.
        Returns up to `max_ideas` HarvestedIdeas sorted by mention_count desc.
        """
        print("\n[IdeaHarvester] Starting harvest ...")

        # Phase 1: Collect snippets
        all_snippets: list[tuple[str, str]] = []  # (source, snippet_text)

        if seed_industries:
            for seed in seed_industries:
                print(f"  -> Mining seed: '{seed}' ...")
                for source, templates in self._SOURCE_QUERIES.items():
                    for template in templates:
                        query = template.replace("{seed}", seed)
                        snips = self._fetch_snippets(query)
                        all_snippets.extend((source, s) for s in snips)
                        time.sleep(0.3)
        else:
            print("  -> No seeds provided — running broad queries ...")
            for source, query in self._BROAD_QUERIES:
                snips = self._fetch_snippets(query, num=8)
                all_snippets.extend((source, s) for s in snips)
                time.sleep(0.3)

        print(f"  -> Collected {len(all_snippets)} raw snippets from Serper")

        # Phase 1b: HN direct API — explicit "is there a tool?" demand signals
        if seed_industries:
            for seed in seed_industries[:4]:
                hn_snips = self._hn.get_idea_signals(seed, n=5)
                all_snippets.extend(("hn_ask", s) for s in hn_snips)
        else:
            hn_snips = self._hn.get_idea_signals("software tool B2B", n=8)
            all_snippets.extend(("hn_ask", s) for s in hn_snips)

        # Phase 1c: Google Trends rising queries — surface adjacent trending problems
        if self._trends and seed_industries:
            print("  -> Mining Google Trends rising queries ...")
            for seed in seed_industries[:3]:
                rising = self._trends.rising_queries(seed)
                if rising:
                    print(f"    Rising for '{seed}': {rising[:3]}")
                    for q in rising[:4]:
                        snips = self._fetch_snippets(
                            f'{q} "problem" OR "no tool" OR "manually" OR "wish"', num=4
                        )
                        all_snippets.extend(("trends_rising", s) for s in snips)
                        time.sleep(0.2)

        print(f"  -> {len(all_snippets)} total snippets after enrichment")

        if not all_snippets:
            print("  [warn] No snippets collected. Try providing --seeds.")
            return []

        # Phase 2: Extract niches from batches
        raw_ideas: list[dict] = []
        batch_size = 25
        batches = [
            all_snippets[i: i + batch_size]
            for i in range(0, len(all_snippets), batch_size)
        ]
        for i, batch in enumerate(batches, 1):
            print(f"  -> Extracting niches from batch {i}/{len(batches)} ...")
            extracted = self._extract_niches(batch)
            raw_ideas.extend(extracted)
            time.sleep(0.4)

        # Phase 3: Consolidate and rank
        print("  -> Consolidating duplicates ...")
        final_ideas = self._consolidate(raw_ideas, max_ideas=max_ideas)

        # Attach evidence snippets back to each idea
        for idea in final_ideas:
            sources_seen: set[str] = set()
            for source, snippet in all_snippets:
                if any(w.lower() in snippet.lower() for w in idea.niche.split()):
                    if source not in sources_seen:
                        sources_seen.add(source)
                        idea.sources.append(source)
                    if snippet not in idea.evidence:
                        idea.evidence.append(snippet)
            idea.evidence = idea.evidence[:5]  # keep top 5 per idea

        print(f"  [done] Harvested {len(final_ideas)} candidate niches")
        return final_ideas

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_snippets(self, query: str, num: int = 5) -> list[str]:
        try:
            raw = self._scraper._serper_search(query, num=num, tbs=None)
            return [
                item.get("snippet", "")
                for item in raw.get("organic", [])
                if item.get("snippet")
            ]
        except Exception as e:
            print(f"    [warn] Query failed ('{query[:60]}...'): {e}")
            return []

    def _extract_niches(self, snippets: list[tuple[str, str]]) -> list[dict]:
        text = "\n---\n".join(f"[{src}] {s}" for src, s in snippets)
        try:
            resp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self._EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Snippets:\n{text}\n\nExtract niches:"},
                ],
                max_tokens=900,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content)
            # Normalise: accept {"niches": [...]} or top-level list
            if isinstance(parsed, list):
                return parsed
            for key in ("niches", "ideas", "results"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
        except Exception as e:
            print(f"    [warn] LLM extraction failed: {e}")
        return []

    def _consolidate(self, raw_ideas: list[dict], max_ideas: int) -> list[HarvestedIdea]:
        """Merge near-duplicates and return top `max_ideas` via a second LLM pass."""
        if not raw_ideas:
            return []

        # Aggregate raw counts first
        counts: dict[str, int] = {}
        pains: dict[str, str] = {}
        for item in raw_ideas:
            niche = item.get("niche", "").strip()
            if not niche:
                continue
            counts[niche] = counts.get(niche, 0) + item.get("mention_count", 1)
            if niche not in pains:
                pains[niche] = item.get("pain", "")

        if not counts:
            return []

        niche_list = "\n".join(
            f"- {n} (mentions:{c}): {pains.get(n, '')}"
            for n, c in sorted(counts.items(), key=lambda x: x[1], reverse=True)
        )
        prompt = (
            f"Consolidate the following software niche ideas into the top {max_ideas} "
            "most distinct, specific opportunities. Merge near-duplicates by summing their "
            "mention counts.\n\n"
            f"{niche_list}\n\n"
            "Return a JSON object with key \"niches\", each item having: "
            "\"niche\" (2-6 words), \"pain\" (one sentence), \"mention_count\" (int). "
            "Sort by mention_count descending. Return ONLY valid JSON."
        )
        try:
            resp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content)
            items = parsed.get("niches", parsed.get("ideas", []))
            return [
                HarvestedIdea(
                    niche=item["niche"],
                    pain=item.get("pain", ""),
                    mention_count=item.get("mention_count", 1),
                )
                for item in items
                if item.get("niche")
            ]
        except Exception as e:
            print(f"  [warn] Consolidation LLM call failed ({e}) — using raw ranking")
            return [
                HarvestedIdea(niche=n, pain=pains.get(n, ""), mention_count=c)
                for n, c in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:max_ideas]
            ]


# ---------------------------------------------------------------------------
# ProductIdeaGenerator - turns a NicheReport into actionable product ideas
# ---------------------------------------------------------------------------


class ProductIdeaGenerator:
    """
    Takes a scored NicheReport and generates specific, buildable product ideas
    grounded in the real pain evidence and competitive gaps already collected.

    Uses gpt-4o for richer ideation (creative task benefits from the full model).
    """

    _PROMPT_TEMPLATE = """\
You are a product strategist and startup advisor specialising in finding underserved B2B niches.

Market analysis for: "{niche}"
  Opportunity Score : {os_score}/10
  Demand            : {demand}/10
  Pain Intensity    : {pain}/10
  Competitive Density: {density}/10  (10 = very crowded)
  AI-Native rivals  : {ai_count} found on page 1

Existing competitors detected:
{competitor_list}

Real user pain evidence (verbatim from the web):
{pain_snippets}

────────────────────────────────────────────
Generate {n} specific, buildable product ideas that exploit the gaps above.
Prioritise ideas that:
  • Target a user segment the current tools ignore
  • Are simpler/cheaper/more focused than the incumbents
  • Could realistically be validated in 6 weeks

Return a JSON object with key "ideas". Each item must have:
  "name"            — product name, 2-4 words
  "one_liner"       — what it does in one sentence
  "target_user"     — be hyper-specific (e.g. "solo HVAC contractors with <5 vans", NOT "small businesses")
  "core_feature"    — the single capability that makes or breaks this product
  "differentiation" — concrete reason it beats {competitor_names}
  "pricing_model"   — e.g. "$49/mo flat", "$5/user/mo", "usage-based", "freemium + $99/mo pro"
  "mvp_scope"       — 2-3 sentence description of the smallest shippable version in 4-6 weeks

Return ONLY valid JSON.\
"""

    def __init__(self, api_key: str = OPENAI_API_KEY, model: str = "gpt-4o") -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, report: "NicheReport", n_ideas: int = 3) -> list[ProductIdea]:
        """Return `n_ideas` ProductIdea objects grounded in the report's data."""
        competitor_domains = [
            c.domain for c in report.competitors
            if c.domain not in _CONTENT_DOMAINS
            and not any(cd in c.domain for cd in _CONTENT_DOMAINS)
        ][:6]

        competitor_list = (
            "\n".join(f"  • {d}" for d in competitor_domains)
            if competitor_domains else "  • None found"
        )
        competitor_names = ", ".join(competitor_domains[:3]) or "existing tools"

        pain_snippets = (
            "\n".join(f'  "{s}"' for s in report.pain_snippets[:8])
            if report.pain_snippets else "  - No snippets collected."
        )

        prompt = self._PROMPT_TEMPLATE.format(
            niche=report.niche,
            os_score=report.final_os_score,
            demand=report.demand_score,
            pain=report.pain_intensity,
            density=report.competitive_density,
            ai_count=report.ai_native_count,
            competitor_list=competitor_list,
            pain_snippets=pain_snippets,
            n=n_ideas,
            competitor_names=competitor_names,
        )

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,
                temperature=0.75,
                response_format={"type": "json_object"},
            )
            items = json.loads(resp.choices[0].message.content).get("ideas", [])
            return [
                ProductIdea(
                    name=item.get("name", ""),
                    one_liner=item.get("one_liner", ""),
                    target_user=item.get("target_user", ""),
                    core_feature=item.get("core_feature", ""),
                    differentiation=item.get("differentiation", ""),
                    pricing_model=item.get("pricing_model", ""),
                    mvp_scope=item.get("mvp_scope", ""),
                )
                for item in items
                if item.get("name")
            ]
        except Exception as e:
            print(f"[warn] ProductIdeaGenerator failed: {e}")
            return []


# ---------------------------------------------------------------------------
# OpportunityScorer - orchestrates everything
# ---------------------------------------------------------------------------


class OpportunityScorer:
    """
    Ties together MarketScraper + SentimentAnalyzer to produce a NicheReport.
    """

    def __init__(self) -> None:
        self.scraper = MarketScraper()
        self.analyzer = SentimentAnalyzer()
        self.detector = DetectionLayer(self.scraper)

        # Optional enrichment signals — active when deps installed + keys present
        self._hn = HNSignal()

        self._trends: Optional[TrendsSignal] = None
        if _PYTRENDS_AVAILABLE:
            try:
                self._trends = TrendsSignal()
                print("[+] Google Trends: active")
            except Exception as e:
                print(f"[warn] Google Trends unavailable: {e}")

        self._reddit: Optional[RedditSignal] = None
        if _PRAW_AVAILABLE and REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET:
            try:
                self._reddit = RedditSignal()
                print("[+] Reddit signal: active")
            except Exception as e:
                print(f"[warn] Reddit unavailable: {e}")

    def analyze(self, niche: str) -> NicheReport:
        report = NicheReport(niche=niche)
        print(f"\n[+] Analyzing niche: '{niche}'")

        # Step 1: Collect pain signals — Serper + Reddit (if active) + HN
        print("  -> Fetching pain discussions ...")
        pain_results = self.scraper.fetch_pain_discussions(niche)
        serper_snippets = [r.snippet for r in pain_results if r.snippet]

        # Reddit — upvote-weighted snippets + community size (TAM proxy)
        reddit_snippets: list[str] = []
        if self._reddit:
            print("  -> Fetching Reddit signals ...")
            report.reddit_subscribers = self._reddit.community_size(niche)
            reddit_snippets = self._reddit.get_pain_snippets(niche, n=15)

        # HN — Ask HN count + high-signal snippets from B2B decision-makers
        print("  -> Fetching HN signals ...")
        report.hn_ask_count = self._hn.ask_hn_count(niche)
        hn_snippets = self._hn.get_pain_snippets(niche, n=10)

        # Merge all snippets (Serper first for diversity, then Reddit, then HN)
        all_snippets = serper_snippets + reddit_snippets + hn_snippets
        report.pain_snippets = all_snippets[:30]

        # Demand score — quality-weighted across all sources
        count = len(pain_results)
        high_signal = sum(
            1 for s in all_snippets
            if any(w in s.lower() for w in _HIGH_SIGNAL_PAIN_WORDS)
        )
        forum_hits = sum(
            1 for r in pain_results
            if any(f in _extract_domain(r.link) for f in _FORUM_DOMAINS)
        )
        base_demand = math.log1p(count) / math.log1p(30) * 5    # Serper volume: max 5
        quality_bonus = min(2.0, high_signal * 0.35)             # pain language: max 2.0
        forum_bonus = min(0.8, forum_hits * 0.2)                 # forum sources: max 0.8
        # HN Ask count: explicit "is there a tool" posts = validated unsatisfied demand
        hn_bonus = min(0.8, math.log1p(report.hn_ask_count) / math.log1p(20) * 0.8)
        # Reddit community: 100k subs → ~+0.5; 1M → ~+0.9; 5M+ → capped 1.2
        reddit_bonus = (
            min(1.2, math.log10(max(1, report.reddit_subscribers)) / 6.0 * 1.2)
            if report.reddit_subscribers > 0 else 0.0
        )
        report.demand_score = round(
            min(10.0, base_demand + quality_bonus + forum_bonus + hn_bonus + reddit_bonus), 2
        )

        # Step 2: Score pain intensity via LLM
        print("  -> Scoring pain intensity ...")
        report.pain_intensity = self.analyzer.score_pain(niche, report.pain_snippets)

        # Step 3: Fetch competitors & enrich with WHOIS
        print("  -> Fetching competitors ...")
        comp_results = self.scraper.fetch_competitors(niche)
        now = datetime.now(tz=timezone.utc)

        ai_native_count = 0
        for sr in comp_results[:10]:
            domain = _extract_domain(sr.link)
            info = CompetitorInfo(domain=domain)

            # WHOIS — domain age for Competitive_Density calculation
            created = self.scraper.fetch_domain_age(domain)
            if created:
                info.creation_date = created
                info.age_years = (now - created).days / 365.25
            time.sleep(0.2)

            # DetectionLayer — richer founding year + AI-signal check
            print(f"    [detect] {domain} …")
            info = self.detector.enrich(info, hint_snippet=sr.snippet)

            if info.is_ai_native:
                ai_native_count += 1
                print(
                    f"    [AI-native] {domain} | "
                    f"founded={info.founded_year} via {info.founding_source} "
                    f"(conf={info.detection_confidence}) | "
                    f"signals={info.ai_signals}"
                )

            report.competitors.append(info)

        # Competitive density — two signals blended:
        #   1. SaaS saturation: how many dedicated product tools are on page 1?
        #   2. Incumbent age: young avg age = crowded; old avg age = stale incumbents
        saas_count = 0
        for sr in comp_results[:10]:
            d = _extract_domain(sr.link)
            is_content = d in _CONTENT_DOMAINS or any(cd in d for cd in _CONTENT_DOMAINS)
            url_lower = sr.link.lower()
            snippet_lower = sr.snippet.lower()
            is_saas = not is_content and (
                any(url_lower.endswith(f".{ext}") or f".{ext}/" in url_lower
                    for ext in ("io", "ai", "app", "co")) or
                any(sig in url_lower for sig in ("/pricing", "/features", "/product", "free-trial")) or
                any(sig in snippet_lower for sig in _SAAS_SNIPPET_SIGNALS)
            )
            if is_saas:
                saas_count += 1

        # SaaS density: each dedicated tool = 1.4 points (7 tools fills the scale)
        saas_density = min(10.0, saas_count * 1.4)

        # Age density: unknown WHOIS treated as 2 yrs (new entrant = crowded signal)
        ages = [
            c.age_years if c.age_years > 0 else 2.0
            for c in report.competitors[:TOP_N_FOR_AGE]
        ]
        if ages:
            avg_age = sum(ages) / len(ages)
            # 0-2 yrs → ~9.0 density; 10 yrs → ~6.2; 20 yrs → ~2.7; 25+ → floor 1.0
            age_density = round(max(1.0, 9.0 - max(0.0, avg_age - 2.0) * 0.35), 2)
        else:
            age_density = 7.0  # no data → assume moderately crowded

        # Blend: SaaS saturation carries more weight than age
        report.competitive_density = round(0.65 * saas_density + 0.35 * age_density, 2)

        # Step 4: store AI-native count (graduated penalty lives in final_os_score)
        report.ai_native_count = ai_native_count
        if ai_native_count > 0:
            print(f"  [!] {ai_native_count} AI-native startup(s) detected on page 1")

        # Step 5: Underserved-market signals
        competitor_domains = [
            c.domain for c in report.competitors
            if c.domain not in _CONTENT_DOMAINS
            and not any(cd in c.domain for cd in _CONTENT_DOMAINS)
        ]

        print("  -> Scoring solutionless rate ...")
        raw_solutionless = self._score_solutionless(
            niche, report.pain_snippets, competitor_domains
        )
        # HN Ask posts are explicit solutionless evidence — boost proportionally
        hn_boost = min(1.5, report.hn_ask_count * 0.15)
        report.solutionless_score = round(min(10.0, raw_solutionless + hn_boost), 2)

        print("  -> Scoring willingness to pay ...")
        report.willingness_to_pay = self._score_willingness_to_pay(niche, report.pain_snippets)

        print("  -> Scoring market momentum ...")
        report.momentum_score = self._score_momentum(niche)

        print(f"  [done] Final OS Score: {report.final_os_score}")
        return report

    # ------------------------------------------------------------------
    # Underserved-market signal helpers
    # ------------------------------------------------------------------

    def _score_solutionless(
        self, niche: str, snippets: list[str], competitor_domains: list[str]
    ) -> float:
        """
        Asks the LLM: of these pain discussions, how many show no awareness of
        an existing tool?  High rate = people are stranded = underserved market.
        Returns 0-10.
        """
        if not snippets:
            return 5.0
        competitor_names = ", ".join(d.split(".")[0] for d in competitor_domains[:5]) or "none"
        evidence = "\n".join(f"- {s}" for s in snippets[:12])
        prompt = (
            f"Niche: {niche}\n"
            f"Known tools in this space: {competitor_names}\n\n"
            f"Pain snippets:\n{evidence}\n\n"
            "Count how many snippets express pain WITHOUT referencing a specific existing "
            "product or tool as a solution. A snippet is 'solutionless' if the person seems "
            "unaware of or unable to find an adequate tool.\n"
            'Return JSON only: {"solutionless": int, "total": int}'
        )
        try:
            resp = self.analyzer._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            total = max(data.get("total", len(snippets)), 1)
            rate = data.get("solutionless", 0) / total
            return round(rate * 10, 2)
        except Exception as e:
            print(f"  [warn] Solutionless scoring failed: {e}")
            return 5.0

    def _score_willingness_to_pay(self, niche: str, pain_snippets: list[str]) -> float:
        """
        Two signals blended:
          1. Price anchors in snippets — people mention dollar amounts = budget exists
          2. Job posting count — manual-process roles signal salary already being spent
        Returns 0-10.
        """
        # Signal 1: price mentions in pain snippets
        price_hits = sum(
            1 for s in pain_snippets
            if any(p.search(s) for p in _PRICE_PATTERNS)
        )
        price_score = min(5.0, price_hits * 1.5)

        # Signal 2: job postings for manual roles (proxy for existing manual spend)
        try:
            jobs = self.scraper._serper_search(
                f'site:linkedin.com/jobs OR site:indeed.com "{niche}" coordinator specialist manager',
                num=10, tbs=None,
            )
            job_count = len(jobs.get("organic", []))
            job_score = min(5.0, job_count * 0.6)
            time.sleep(0.2)
        except Exception:
            job_score = 2.5

        return round(min(10.0, price_score + job_score), 2)

    def _score_momentum(self, niche: str) -> float:
        """
        How fast is interest in this problem growing?

        Primary:  Google Trends — real weekly search-interest data over 5 years.
                  Compares last-12-week avg to trailing-52-week avg.
        Fallback: Serper 30-day vs annual result-count comparison.

        flat=5.0 | doubling=~7.5 | halving=~2.5
        """
        if self._trends:
            score = self._trends.score_momentum(niche)
            if score is not None:
                print(f"  [trends] momentum={score} (Google Trends)")
                return score
        return self._score_momentum_serper(niche)

    def _score_momentum_serper(self, niche: str) -> float:
        """Fallback momentum: compare Serper 30-day vs annual result counts."""
        try:
            recent = self.scraper._serper_search(
                f'"{niche}" problem OR solution', num=10, tbs="qdr:m"
            )
            annual = self.scraper._serper_search(
                f'"{niche}" problem OR solution', num=10, tbs="qdr:y"
            )
            recent_count = len(recent.get("organic", []))
            annual_count = len(annual.get("organic", []))
            time.sleep(0.2)
            if annual_count == 0:
                return 5.0
            monthly_avg = annual_count / 12.0
            ratio = recent_count / monthly_avg if monthly_avg > 0 else 1.0
            return round(min(10.0, max(0.0, ratio * 5.0)), 2)
        except Exception as e:
            print(f"  [warn] Momentum scoring failed: {e}")
            return 5.0


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------


def save_to_csv(reports: list[NicheReport], path: str = OUTPUT_FILE) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in reports:
            writer.writerow(r.to_csv_row())
    print(f"\n[done] Results written to {path}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _year_from_date(dt: Optional[datetime]) -> Optional[int]:
    """Return the year from a datetime, or None."""
    return dt.year if dt else None


def _extract_domain(url: str) -> str:
    """Strip scheme + path, return bare domain (e.g. 'example.com')."""
    url = re.sub(r"^https?://", "", url)
    return url.split("/")[0].lstrip("www.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Market Gap Engine — score niches or auto-harvest ideas first."
    )
    parser.add_argument(
        "niches",
        nargs="*",
        help="Niches to analyze directly (skip harvesting).",
    )
    parser.add_argument(
        "--harvest",
        action="store_true",
        help="Auto-generate niche ideas from online pain signals, then score them.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        metavar="INDUSTRY",
        default=None,
        help="Industry seed terms for the harvester (e.g. --seeds HR finance operations).",
    )
    parser.add_argument(
        "--max-ideas",
        type=int,
        default=10,
        metavar="N",
        help="Max number of ideas to harvest before scoring (default: 10).",
    )
    args = parser.parse_args()

    scorer = OpportunityScorer()
    niches: list[str] = []

    if args.harvest:
        # Full auto pipeline: mine → score
        harvester = IdeaHarvester(scorer.scraper)
        ideas = harvester.harvest(seed_industries=args.seeds, max_ideas=args.max_ideas)
        if not ideas:
            print("[!] Harvester returned no ideas. Try providing --seeds.")
            return
        print("\nHarvested candidates:")
        for i, idea in enumerate(ideas, 1):
            print(f"  {i:2}. {idea.niche:<45} mentions={idea.mention_count}")
            print(f"      Pain: {idea.pain}")
        print()
        niches = [idea.niche for idea in ideas]
    elif args.niches:
        niches = args.niches
    else:
        # Demo mode
        niches = [
            "AI meeting notes tool",
            "fleet management software",
            "B2B cold email automation",
        ]

    reports: list[NicheReport] = []
    for niche in niches:
        try:
            reports.append(scorer.analyze(niche))
        except Exception as e:
            print(f"  [error] Failed to analyze '{niche}': {e}")

    if reports:
        save_to_csv(reports)
    else:
        print("[!] No reports generated.")


if __name__ == "__main__":
    main()
