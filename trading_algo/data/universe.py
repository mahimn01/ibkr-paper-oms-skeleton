"""Survivorship-bias-free universe resolution.

`UniverseResolver.get_universe('SP500', date(2008, 9, 15))` returns the
exact constituents of the S&P 500 on that date — including securities
that have since delisted (e.g. Lehman Brothers).

This replaces every hardcoded `["AAPL", "MSFT", "SPY", ...]` list in
strategy code. Strategies receive the as-of-date universe from the
backtest engine; they never decide universe membership themselves.

Source priority for membership data (PLAN.md §2.1):
    1. PIT store's `index_membership` table (populated from Norgate / WRDS / manual).
    2. Hardcoded "constants" universes for development (clearly marked).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Iterable

from trading_algo.data.pit_store import PITStore

log = logging.getLogger(__name__)


# Tiny development-only universes. NOT for production backtests — they leak
# survivorship bias by construction. Use index names + populated
# index_membership table instead.
_DEV_UNIVERSES: dict[str, frozenset[str]] = {
    "DEV_MEGACAP": frozenset({"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"}),
    "DEV_ETFS":    frozenset({"SPY", "QQQ", "IWM", "DIA", "TLT", "GLD"}),
    "DEV_FAANG":   frozenset({"META", "AAPL", "AMZN", "NFLX", "GOOGL"}),
}


class SurvivorshipBiasError(Exception):
    """Raised when caller asks for a universe that requires PIT data we don't have."""


class UniverseResolver:
    """Resolve index membership at a point in time.

    Construction:
        UniverseResolver(store=PITStore(...))
        UniverseResolver()                       # store-less, dev mode only

    Usage:
        resolver.get_universe('SP500', date(2008, 9, 15))
        resolver.get_universe(['AAPL', 'MSFT'])  # literal list, no PIT
    """

    def __init__(self, store: PITStore | None = None) -> None:
        self.store = store

    # ---------------------------------------------------------------- public

    def get_universe(
        self,
        spec: str | Iterable[str],
        as_of: date | None = None,
        *,
        allow_dev: bool = False,
    ) -> list[str]:
        """Resolve `spec` at `as_of` to a list of tickers.

        `spec` is one of:
            * an index name (e.g. 'SP500', 'R1000', 'R2000') -> PIT lookup
            * a development universe name (e.g. 'DEV_MEGACAP') -> requires allow_dev=True
            * an iterable of literal tickers -> returned as-is (no PIT, caller's risk)

        For index names, `as_of` is required. For literal lists, `as_of` is ignored.
        """
        # Literal iterable.
        if not isinstance(spec, str):
            return sorted({t.upper() for t in spec})

        # Dev universes.
        if spec in _DEV_UNIVERSES:
            if not allow_dev:
                raise SurvivorshipBiasError(
                    f"Universe '{spec}' is a development convenience and is "
                    "survivorship-biased. Pass allow_dev=True to use it for "
                    "exploration only. For production backtests use an index "
                    "name with a populated index_membership table."
                )
            return sorted(_DEV_UNIVERSES[spec])

        # Index name -> PIT lookup.
        if as_of is None:
            raise ValueError(
                f"Index universe '{spec}' requires `as_of` date for "
                "point-in-time membership resolution."
            )
        if self.store is None:
            raise SurvivorshipBiasError(
                f"Cannot resolve '{spec}' without a PITStore. "
                "Construct UniverseResolver(store=...) and populate "
                "index_membership before backtesting an index universe."
            )
        return self._lookup_index_members(spec, as_of)

    def get_universe_timeline(
        self,
        index_name: str,
        start: date,
        end: date,
    ) -> list[tuple[date, date | None, str]]:
        """Return (added_date, removed_date, ticker) triples for `index_name`
        whose membership window overlaps [start, end]."""
        if self.store is None:
            raise SurvivorshipBiasError(
                f"Cannot get timeline for '{index_name}' without a PITStore."
            )
        with self.store._conn() as conn:    # noqa: SLF001
            rows = conn.execute(
                """
                SELECT im.added_date, im.removed_date, s.primary_ticker
                FROM index_membership im
                JOIN securities s ON s.internal_id = im.internal_id
                WHERE im.index_name = ?
                  AND im.added_date <= ?
                  AND (im.removed_date IS NULL OR im.removed_date >= ?)
                ORDER BY im.added_date
                """,
                (index_name, end.isoformat(), start.isoformat()),
            ).fetchall()
        return [
            (
                date.fromisoformat(r["added_date"]),
                date.fromisoformat(r["removed_date"]) if r["removed_date"] else None,
                r["primary_ticker"],
            )
            for r in rows
        ]

    # ---------------------------------------------------------------- internal

    def _lookup_index_members(self, index_name: str, as_of: date) -> list[str]:
        d = as_of.isoformat()
        with self.store._conn() as conn:    # noqa: SLF001
            rows = conn.execute(
                """
                SELECT s.primary_ticker, im.internal_id
                FROM index_membership im
                JOIN securities s ON s.internal_id = im.internal_id
                WHERE im.index_name = ?
                  AND im.added_date <= ?
                  AND (im.removed_date IS NULL OR im.removed_date > ?)
                  AND (s.delist_date IS NULL OR s.delist_date > ?)
                """,
                (index_name, d, d, d),
            ).fetchall()
        if not rows:
            log.warning(
                "UniverseResolver: zero members for index=%s as_of=%s. "
                "Did you populate index_membership?",
                index_name, as_of,
            )
        # Resolve each internal_id's ticker as of `as_of` (handles ticker changes).
        out: list[str] = []
        for r in rows:
            ticker_at_date = self._ticker_as_of(int(r["internal_id"]), as_of)
            out.append(ticker_at_date or r["primary_ticker"])
        return sorted(set(out))

    def _ticker_as_of(self, internal_id: int, as_of: date) -> str | None:
        d = as_of.isoformat()
        with self.store._conn() as conn:    # noqa: SLF001
            row = conn.execute(
                """
                SELECT ticker FROM ticker_history
                WHERE internal_id = ? AND valid_from <= ? AND valid_to > ?
                LIMIT 1
                """,
                (internal_id, d, d),
            ).fetchone()
            return row["ticker"] if row else None
