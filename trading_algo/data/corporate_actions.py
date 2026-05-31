"""Corporate-action adjustment factors at query time.

Philosophy (PLAN.md §2.1, §2.5):
    Bars are stored unadjusted. When a strategy needs an adjusted price
    series, AdjustmentEngine computes the cumulative split / dividend
    factor for events that occur strictly AFTER the simulated time T,
    and applies that to the historical series to produce a
    forward-adjusted price *as of* time T.

    This preserves point-in-time correctness:
        * Simulated time T = 2019-01-01: AAPL 4:1 split (2020-08-31) is
          in the future, so factor = 1.0; you see the unadjusted ~$155.
        * Simulated time T = 2021-01-01: split is in the past, so factor
          for any pre-split price = 4.0; pre-split prices divide by 4.

Forward-adjusting at storage time (the Yahoo default) is permanently wrong
for backtesting because reloading after a future split silently rewrites
the past.

Reference: López de Prado, "Advances in Financial Machine Learning" (2018), Ch 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

from trading_algo.data.pit_store import Bar, PITStore


@dataclass(frozen=True)
class AdjustedBar:
    """A bar with the adjusted-close field populated."""
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: float
    adjustment_factor: float


class AdjustmentEngine:
    """Compute adjustment factors and adjust bar series at query time.

    Adjustment factor convention:
        factor(t, T) = product of (1 / split_ratio_e) for splits with
                       ex_date e where t < e <= T,
                       times product of (1 - div_e / close_e_minus_1) for
                       dividends if `total_return=True`.

        For prices BEFORE a 4:1 split that has already happened relative to
        simulation time T, the adjustment factor is 0.25 (divide pre-split
        price by 4 to make it comparable to post-split prices).
    """

    def __init__(self, store: PITStore) -> None:
        self.store = store

    # ---------------------------------------------------------------- factor

    def factor(
        self,
        internal_id: int,
        bar_date: date,
        as_of: date,
        *,
        include_dividends: bool = False,
    ) -> float:
        """Cumulative adjustment factor to convert an unadjusted price on
        `bar_date` into an adjusted price as visible at `as_of`.

        Walks splits between `bar_date` (exclusive) and `as_of` (inclusive).
        Dividends are off by default — most strategies use price returns;
        total-return adjustment is opt-in.
        """
        if as_of < bar_date:
            return 1.0
        f = 1.0
        for ex_date, ratio in self.store.get_splits(internal_id):
            if ex_date <= bar_date:
                continue
            if ex_date > as_of:
                continue
            # Pre-split price is "ratio" times bigger than post-split.
            # To bring it into the post-split frame, divide by ratio.
            f /= ratio
        if include_dividends:
            # Approximate: subtract dividend amount from price post-ex.
            # This is the simple "subtract dividend" total-return adjustment.
            # A more rigorous formulation rescales by (1 - div / close_pre_ex).
            divs = self.store.get_dividends(internal_id, types=("regular", "special"))
            for ex_date, amount, _ in divs:
                if ex_date <= bar_date or ex_date > as_of:
                    continue
                # Approximate factor for amount-style adjustment: applied as
                # multiplicative scaling once we know close_{ex-1}. That requires
                # the bar series, so this branch is left to `adjust_series`.
                pass
        return f

    # ---------------------------------------------------------------- series

    def adjust_series(
        self,
        internal_id: int,
        bars: Sequence[Bar],
        as_of: date,
        *,
        include_dividends: bool = False,
    ) -> list[AdjustedBar]:
        """Apply cumulative adjustments forward to `as_of` for each bar.

        Returns AdjustedBar with `adj_close` and `adjustment_factor`. Strategies
        that need adjusted OHLC can multiply each price field by `adjustment_factor`.
        """
        if not bars:
            return []
        sorted_bars = sorted(bars, key=lambda b: b.date)

        # Pre-load events once.
        splits = self.store.get_splits(internal_id)
        divs = (
            self.store.get_dividends(internal_id, types=("regular", "special"))
            if include_dividends else []
        )

        # Build a map from bar_date -> close at bar_date for dividend factor.
        close_by_date = {b.date: b.close for b in sorted_bars}

        out: list[AdjustedBar] = []
        for b in sorted_bars:
            f = 1.0
            for ex_date, ratio in splits:
                if ex_date <= b.date or ex_date > as_of:
                    continue
                f /= ratio
            if include_dividends:
                for ex_date, amount, _ in divs:
                    if ex_date <= b.date or ex_date > as_of:
                        continue
                    # Find the close on the day before ex_date to compute
                    # the multiplicative adjustment.
                    prior_dates = [d for d in close_by_date if d < ex_date]
                    if not prior_dates:
                        continue
                    prior_close = close_by_date[max(prior_dates)]
                    if prior_close <= 0:
                        continue
                    f *= max(0.0, 1.0 - amount / prior_close)
            out.append(AdjustedBar(
                symbol=b.symbol,
                date=b.date,
                open=b.open,
                high=b.high,
                low=b.low,
                close=b.close,
                volume=b.volume,
                adj_close=b.close * f,
                adjustment_factor=f,
            ))
        return out
