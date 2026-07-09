"""WorldMonitor context brief — a STRICTLY ADVISORY, non-signal situational feed.

Hard invariants (the whole point of walling WorldMonitor off):
  * `ContextBrief.signal` is always False. The constitution gate ignores the
    brief; it can never flip a BLOCK or be cited as an edge.
  * IBKR-first: the brief carries WORDS, not numbers used for sizing. Any
    price/greek that drives a trade comes from IBKR, never from here.
  * Fail-soft: WorldMonitor is a manually-launched dev server. If it's down,
    the brief degrades (degraded=True) and NEVER raises.

WorldMonitor is reachable as a plain HTTP API (proto routes /api/{svc}/v1/{rpc})
on WORLDMONITOR_BASE_URL (default http://127.0.0.1:3777) — no MCP/LLM needed.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:3777"

# Allow-list of CONTEXT endpoints (verified live). Deliberately excludes the
# alpha-masquerade feeds (options_flow, social_sentiment-as-ranker, congress) —
# those are dead signals per the research and must not enter the trade path.
EP_MACRO_SIGNALS = "/api/economic/v1/get-macro-signals"
EP_FEED_DIGEST = "/api/news/v1/list-feed-digest"
EP_SECTOR_SUMMARY = "/api/market/v1/get-sector-summary"

# IBKR news injected as a callable so this module never imports ib_async.
# signature: (symbols: list[str]) -> list[dict] with at least {"headline": str}.
IbkrNewsProvider = Callable[[list[str]], list[dict]]


@dataclass(frozen=True)
class ContextBrief:
    as_of: float
    signal: bool = False  # INVARIANT — advisory only, never a signal
    degraded: bool = False  # WorldMonitor unreachable / partial
    headline_regime: str = ""
    relevant_to_positions: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    headlines: list[str] = field(default_factory=list)
    narrative: str = ""
    sources: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.signal is not False:
            raise ValueError(
                "ContextBrief.signal must be False — context is advisory, never a trade signal."
            )

    def as_text(self) -> str:
        head = "MARKET CONTEXT (advisory, non-signal)"
        if self.degraded:
            head += "  [degraded: WorldMonitor partial/unreachable]"
        lines = [head]
        if self.headline_regime:
            lines.append(f"  regime: {self.headline_regime}")
        if self.risk_flags:
            lines.append("  risk flags:")
            lines.extend(f"    - {f}" for f in self.risk_flags)
        if self.headlines:
            lines.append("  top headlines:")
            lines.extend(f"    - {h}" for h in self.headlines[:6])
        return "\n".join(lines)


class WorldMonitorClient:
    """Fail-soft HTTP client for WorldMonitor proto routes. Every method returns
    None on any failure (connection refused, non-JSON source, HTTP error)."""

    def __init__(self, base_url: str | None = None, *, timeout: float = 6.0,
                 client: httpx.Client | None = None) -> None:
        self.base_url = (base_url or os.getenv("WORLDMONITOR_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self._timeout = timeout
        self._client = client or httpx.Client(timeout=timeout)

    def get(self, path: str, params: dict | None = None) -> dict | None:
        try:
            r = self._client.get(self.base_url + path, params=params, timeout=self._timeout)
            if r.status_code != 200:
                return None
            text = r.text.lstrip()
            # vite serves raw JS source for legacy routes — reject it like the MCP client does.
            if text[:1] in {"<"} or text.startswith(("import ", "export ", "module.exports")):
                return None
            data = r.json()
            return data if isinstance(data, dict) else {"data": data}
        except Exception:
            return None

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass


def _macro_regime(client: WorldMonitorClient, sources: list[str]) -> str:
    d = client.get(EP_MACRO_SIGNALS)
    if not d:
        return ""
    sources.append(EP_MACRO_SIGNALS)
    verdict = d.get("verdict")
    bull = d.get("bullishCount")
    bear = d.get("bearishCount")
    parts = []
    if verdict:
        parts.append(str(verdict))
    if bull is not None and bear is not None:
        parts.append(f"{bull} bullish / {bear} bearish signals")
    return "macro: " + ", ".join(parts) if parts else ""


def _digest_headlines(client: WorldMonitorClient, sources: list[str], limit: int = 12) -> list[str]:
    d = client.get(EP_FEED_DIGEST)
    if not d:
        return []
    sources.append(EP_FEED_DIGEST)
    out: list[str] = []
    cats = d.get("categories") or {}
    if isinstance(cats, dict):
        for cat in cats.values():
            for item in (cat or {}).get("items", []) if isinstance(cat, dict) else []:
                src = item.get("source") or ""
                title = (item.get("title") or "").strip()
                if title:
                    out.append(f"{title} ({src})" if src else title)
                if len(out) >= limit:
                    return out
    return out


def _flag_relevant(headlines: list[str], symbols: list[str]) -> tuple[list[str], list[str]]:
    """Heuristic: a headline is relevant to a position if the symbol appears as a
    standalone uppercase token in the title. Conservative on purpose."""
    relevant: list[str] = []
    flags: list[str] = []
    syms = [s.upper() for s in symbols]
    for h in headlines:
        title = h.rsplit(" (", 1)[0]  # strip only the appended "(source)" suffix
        for s in syms:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(s)}(?![A-Za-z0-9])", title):
                if s not in relevant:
                    relevant.append(s)
                flags.append(f"{s}: {title}")
    return relevant, flags


def build_context_brief(
    symbols: list[str],
    *,
    client: WorldMonitorClient | None = None,
    depth: str = "fast",  # "fast" | "deep"
    ibkr_news: IbkrNewsProvider | None = None,
    now: float | None = None,
) -> ContextBrief:
    as_of = now if now is not None else time.time()
    own_client = client is None
    client = client or WorldMonitorClient()
    sources: list[str] = []
    try:
        regime = _macro_regime(client, sources)
        headlines = _digest_headlines(client, sources, limit=12 if depth == "fast" else 24)

        # IBKR-first news layer (injected; never imports ib_async here).
        if ibkr_news is not None and symbols:
            try:
                for n in ibkr_news(symbols) or []:
                    hl = (n.get("headline") or "").strip()
                    if hl:
                        headlines.append(f"{hl} (IBKR)")
            except Exception:
                pass

        relevant, flags = _flag_relevant(headlines, symbols)

        if depth == "deep":
            sec = client.get(EP_SECTOR_SUMMARY)
            if sec is not None:
                sources.append(EP_SECTOR_SUMMARY)

        degraded = not sources  # nothing fetched at all
        narrative_bits = []
        if regime:
            narrative_bits.append(regime)
        if flags:
            narrative_bits.append(f"{len(flags)} headline(s) touch your positions")
        narrative = "; ".join(narrative_bits)

        return ContextBrief(
            as_of=as_of, signal=False, degraded=degraded, headline_regime=regime,
            relevant_to_positions=relevant, risk_flags=flags, headlines=headlines,
            narrative=narrative, sources=sources,
        )
    finally:
        if own_client:
            client.close()
