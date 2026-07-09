"""Tests for the WorldMonitor context brief (advisory, non-signal, fail-soft)."""

from __future__ import annotations

import httpx
import pytest

from trading_algo.context_brief import (
    EP_FEED_DIGEST,
    EP_MACRO_SIGNALS,
    ContextBrief,
    WorldMonitorClient,
    build_context_brief,
)


class FakeClient:
    def __init__(self, responses: dict):
        self.responses = responses

    def get(self, path: str, params=None):
        return self.responses.get(path)

    def close(self):
        pass


_RESP = {
    EP_MACRO_SIGNALS: {"verdict": "CASH", "bullishCount": 3, "bearishCount": 5},
    EP_FEED_DIGEST: {"categories": {"politics": {"items": [
        {"source": "BBC", "title": "MP Materials wins rare-earth supply contract"},
        {"source": "Reuters", "title": "Fed holds rates steady"},
    ]}}},
}


# ---------------------------------------------------------------- invariant

def test_signal_must_be_false():
    ContextBrief(as_of=0.0)  # ok
    with pytest.raises(ValueError, match="signal must be False"):
        ContextBrief(as_of=0.0, signal=True)


# ---------------------------------------------------------------- build

def test_build_brief_regime_and_relevance():
    brief = build_context_brief(["MP", "IAU"], client=FakeClient(_RESP))
    assert brief.signal is False and brief.degraded is False
    assert "CASH" in brief.headline_regime
    assert any("MP Materials" in h for h in brief.headlines)
    assert "MP" in brief.relevant_to_positions
    assert any(f.startswith("MP:") for f in brief.risk_flags)


def test_build_brief_degraded_when_unreachable():
    brief = build_context_brief(["MP"], client=FakeClient({}))  # all None
    assert brief.degraded is True
    assert brief.signal is False  # never raises, never a signal
    assert brief.headlines == []


def test_ibkr_news_injection_appended():
    brief = build_context_brief(
        ["MP"], client=FakeClient(_RESP),
        ibkr_news=lambda syms: [{"headline": "MP trading halt lifted"}],
    )
    assert any("MP trading halt lifted (IBKR)" in h for h in brief.headlines)


def test_relevance_survives_parenthetical_in_title():
    # F5: only the trailing "(source)" is stripped — a parenthetical inside the
    # title must not truncate the match window.
    resp = {EP_FEED_DIGEST: {"categories": {"x": {"items": [
        {"source": "R", "title": "S&P 500 (SPX) rallies to record"},
    ]}}}}
    brief = build_context_brief(["SPX"], client=FakeClient(resp))
    assert "SPX" in brief.relevant_to_positions


def test_relevance_token_boundary_no_false_positive():
    resp = {EP_FEED_DIGEST: {"categories": {"x": {"items": [
        {"source": "X", "title": "IMPACT of inflation on companies"},  # contains 'MP' substring
    ]}}}}
    brief = build_context_brief(["MP"], client=FakeClient(resp))
    assert "MP" not in brief.relevant_to_positions  # not a standalone token


def test_as_text_marks_advisory_and_degraded():
    brief = build_context_brief(["MP"], client=FakeClient({}))
    txt = brief.as_text()
    assert "advisory, non-signal" in txt and "degraded" in txt


# ---------------------------------------------------------------- real client guards

def test_client_rejects_source_and_http_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.path
        if p.endswith("get-macro-signals"):
            return httpx.Response(200, text='{"verdict":"RISK_ON"}')
        if "legacy" in p:
            return httpx.Response(200, text="export const config = { runtime: 'edge' }")
        return httpx.Response(404, text='{"error":"Not found"}')

    c = WorldMonitorClient(client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert c.get(EP_MACRO_SIGNALS)["verdict"] == "RISK_ON"
    assert c.get("/api/legacy/v1/x") is None   # vite source rejected, not parsed
    assert c.get("/api/missing/v1/y") is None   # 404 -> None
    c.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
