"""Realistic transaction cost model for backtests.

Layered estimators (PLAN.md §2.3):
    1. Effective spread from OHLC bars   -> Corwin & Schultz (2012),
                                            Abdi & Ranaldo (2017) fallback.
    2. Square-root market impact         -> Toth, Bouchaud et al. (2011).
    3. Borrow cost for shorts            -> daily ACT/360 accrual, recall risk.
    4. IBKR Tiered commission stack      -> per-share + SEC + FINRA TAF.

Why each estimator:
    The retail "half-spread + 5 bps" baseline systematically overestimates
    edge by 50-150 bps/year on liquid equity strategies and 200-500 bps/year
    on shorts (because borrow is free in naive backtests). These four
    estimators together close that gap without requiring tick data.

References:
    - Corwin, S. & Schultz, P. (2012), "A Simple Way to Estimate Bid-Ask
      Spreads from Daily High and Low Prices," J. Finance 67(2), 719-760.
    - Abdi, F. & Ranaldo, A. (2017), "A Simple Estimation of Bid-Ask
      Spreads from Daily Close, High, and Low Prices," RFS 30(12), 4437-4480.
    - Toth, B., Lemperiere, Y., Deremble, C., de Lataillade, J., Kockelkoren,
      J., & Bouchaud, J.-P. (2011), "Anomalous Price Impact and the Critical
      Nature of Liquidity in Financial Markets," PRX 1, 021006.
    - Almgren, R. & Chriss, N. (2000), "Optimal Execution of Portfolio
      Transactions," J. Risk 3(2), 5-39.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Sequence

# --------------------------------------------------------------------------
# Effective spread estimators
# --------------------------------------------------------------------------


def corwin_schultz_spread(
    h_t: float,  l_t: float,
    h_t1: float, l_t1: float,
) -> float:
    """Corwin-Schultz (2012) effective spread estimate from two consecutive
    daily high-low ranges.

    Returns spread as a fraction of price (multiply by 1e4 for bps).
    Returns 0.0 if the estimator is negative (~10% of days for liquid US
    equities) — caller should fall back to Abdi-Ranaldo.

        beta  = [ln(H_t / L_t)]^2 + [ln(H_{t+1} / L_{t+1})]^2
        gamma = [ln(max(H_t, H_{t+1}) / min(L_t, L_{t+1}))]^2
        alpha = (sqrt(2*beta) - sqrt(beta)) / (3 - 2*sqrt(2))
              - sqrt(gamma / (3 - 2*sqrt(2)))
        S     = 2 * (exp(alpha) - 1) / (1 + exp(alpha))
    """
    if min(h_t, l_t, h_t1, l_t1) <= 0:
        return 0.0
    if h_t < l_t or h_t1 < l_t1:
        return 0.0

    beta = (math.log(h_t / l_t) ** 2) + (math.log(h_t1 / l_t1) ** 2)
    gamma_h = max(h_t, h_t1)
    gamma_l = min(l_t, l_t1)
    gamma = math.log(gamma_h / gamma_l) ** 2

    denom = 3.0 - 2.0 * math.sqrt(2.0)
    if denom <= 0 or beta < 0 or gamma < 0:
        return 0.0
    alpha = (math.sqrt(2.0 * beta) - math.sqrt(beta)) / denom \
            - math.sqrt(gamma / denom)
    spread = 2.0 * (math.exp(alpha) - 1.0) / (1.0 + math.exp(alpha))
    return max(0.0, spread)


def abdi_ranaldo_spread(
    closes: Sequence[float],
    highs:  Sequence[float],
    lows:   Sequence[float],
) -> float:
    """Abdi-Ranaldo (2017) effective spread estimator from CHL triplets.

        eta_t  = (H_t + L_t) / 2
        S^2    = 4 * E[(C_t - eta_t) * (C_t - eta_{t+1})]

    Falls back gracefully (returns 0.0) when the estimator is negative
    or there are <3 days of data.
    """
    n = len(closes)
    if n < 3 or len(highs) != n or len(lows) != n:
        return 0.0

    products: list[float] = []
    for t in range(n - 1):
        c = math.log(max(closes[t], 1e-12))
        eta_t  = math.log(max((highs[t]  + lows[t])  / 2.0, 1e-12))
        eta_t1 = math.log(max((highs[t + 1] + lows[t + 1]) / 2.0, 1e-12))
        products.append(4.0 * (c - eta_t) * (c - eta_t1))
    s2 = sum(products) / len(products)
    if s2 <= 0:
        return 0.0
    return math.sqrt(s2)


def daily_effective_spread(
    bars: Sequence[tuple[float, float, float, float]],
) -> float:
    """Compute effective spread for the *last* bar of `bars`.

    Each bar is (open, high, low, close). Uses Corwin-Schultz with the
    previous bar; falls back to Abdi-Ranaldo over the trailing window if
    CS produces a non-positive estimate.

    Returns spread as a fraction of price.
    """
    if len(bars) < 2:
        return 0.0
    _, h_prev, l_prev, _ = bars[-2]
    _, h_t, l_t, _       = bars[-1]
    cs = corwin_schultz_spread(h_prev, l_prev, h_t, l_t)
    if cs > 0:
        return cs
    # Fallback: AR over the trailing 21 bars when available.
    window = bars[-21:] if len(bars) >= 21 else bars
    closes = [b[3] for b in window]
    highs  = [b[1] for b in window]
    lows   = [b[2] for b in window]
    return abdi_ranaldo_spread(closes, highs, lows)


# --------------------------------------------------------------------------
# Square-root impact (Toth-Bouchaud)
# --------------------------------------------------------------------------


def sqrt_impact_bps(
    quantity: float,
    adv: float,
    daily_vol_bps: float,
    *,
    Y: float = 0.5,
    min_participation: float = 0.001,
    max_participation: float = 0.10,
) -> float:
    """Square-root price impact in basis points.

        impact_bps = Y * sigma_daily_bps * sqrt(participation)

    Below min_participation (default 0.1% ADV) the model returns 0 — at
    that size, spread dominates. Above max_participation the model itself
    breaks (linear regime kicks in, you should be using Almgren-Chriss
    scheduling instead of a one-shot estimate).
    """
    if quantity <= 0 or adv <= 0 or daily_vol_bps <= 0:
        return 0.0
    participation = quantity / adv
    if participation < min_participation:
        return 0.0
    participation = min(participation, max_participation)
    return Y * daily_vol_bps * math.sqrt(participation)


# --------------------------------------------------------------------------
# Borrow cost for shorts
# --------------------------------------------------------------------------


class BorrowTier(str, Enum):
    """Default tier rates (bps annualized) used when a vendor rate is unknown."""
    SP500_MEGACAP = "SP500_MEGACAP"      # 30 bps
    R1000          = "R1000"              # 50 bps
    R2000          = "R2000"              # 200 bps
    HTB            = "HTB"                # NaN — must skip


_TIER_RATES_BPS: dict[BorrowTier, float] = {
    BorrowTier.SP500_MEGACAP: 30.0,
    BorrowTier.R1000:         50.0,
    BorrowTier.R2000:         200.0,
}


@dataclass
class BorrowChargeResult:
    daily_charge: float
    rate_bps: float
    recall_triggered: bool


def borrow_charge(
    notional: float,
    rate_bps: float,
    *,
    days: int = 1,
    day_count: int = 360,
) -> float:
    """Daily borrow accrual using ACT/360.

        charge = notional * rate_bps / 1e4 * days / day_count
    """
    if notional <= 0 or rate_bps <= 0 or days <= 0:
        return 0.0
    return notional * (rate_bps / 1e4) * days / day_count


def recall_probability(rate_bps: float) -> float:
    """Daily recall probability scaled by the borrow rate.

    A 50 bps stock has ~0.25% daily recall risk; a 5000 bps (50%) stock
    has 5% — the empirical cap. Above that, the borrow has effectively
    been recalled by the time you got the rate quote.
    """
    if rate_bps <= 0:
        return 0.0
    return min(0.05, rate_bps / 200.0 / 1e2)


# --------------------------------------------------------------------------
# IBKR Tiered commission
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IBKRTieredFees:
    """IBKR Tiered fee schedule for US stocks (2026 published rates).

    Subject to occasional IBKR adjustments — recheck quarterly. The structure
    here is intentionally explicit so each component is auditable.
    """
    per_share: float = 0.0035        # base IBKR commission
    min_per_order: float = 0.35
    max_pct_trade_value: float = 0.01
    sec_fee_pct: float = 0.0000278   # SEC fee on sells, 27.80 / $1M (2024-25 rate)
    finra_taf_per_share: float = 0.000166
    finra_taf_max: float = 9.27
    nyse_pass_through_pct: float = 0.000175  # rough — depends on add/remove


def ibkr_tiered_commission(
    side: str,
    quantity: float,
    fill_price: float,
    *,
    fees: IBKRTieredFees = IBKRTieredFees(),
) -> float:
    """Total commission + regulatory fees for one US-equity fill."""
    if quantity <= 0 or fill_price <= 0:
        return 0.0
    side_u = side.upper()
    if side_u not in {"BUY", "SELL"}:
        raise ValueError(f"unknown side {side!r}")

    trade_value = quantity * fill_price

    # Base IBKR Tiered commission.
    base = max(fees.min_per_order, quantity * fees.per_share)
    base = min(base, trade_value * fees.max_pct_trade_value)

    # SEC fee (sells only).
    sec_fee = trade_value * fees.sec_fee_pct if side_u == "SELL" else 0.0

    # FINRA TAF (sells only, both equity and options).
    taf = 0.0
    if side_u == "SELL":
        taf = min(fees.finra_taf_max, quantity * fees.finra_taf_per_share)

    # Pass-throughs (small).
    pass_through = trade_value * fees.nyse_pass_through_pct

    return base + sec_fee + taf + pass_through


# --------------------------------------------------------------------------
# Composite cost
# --------------------------------------------------------------------------


@dataclass
class FillCostResult:
    """Decomposed cost on a single fill."""
    spread_cost:    float = 0.0
    impact_cost:    float = 0.0
    commission:     float = 0.0
    @property
    def total(self) -> float:
        return self.spread_cost + self.impact_cost + self.commission


@dataclass
class CostModelConfig:
    """Configuration knobs for the realistic cost model.

    Toggle off any layer for debugging / sensitivity analysis.
    """
    enable_spread:     bool = True
    enable_impact:     bool = True
    enable_commission: bool = True
    Y_impact:          float = 0.5
    fees:              IBKRTieredFees = field(default_factory=IBKRTieredFees)
    fallback_spread_bps: float = 5.0     # used when CS+AR both fail
    spread_cap_bps:      float = 200.0   # protect against bad bars


def compute_fill_cost(
    *,
    side: str,
    quantity: float,
    fill_price: float,
    spread_fraction: float,
    adv: float,
    daily_vol_bps: float,
    config: CostModelConfig = CostModelConfig(),
) -> FillCostResult:
    """Decomposed transaction cost in dollars for one fill.

    Inputs:
        side             "BUY" or "SELL"
        quantity         shares (positive)
        fill_price       midpoint or VWAP — the *paper* fill before slippage
        spread_fraction  e.g. 0.0010 = 10 bps (from daily_effective_spread)
        adv              20-day average daily volume in shares
        daily_vol_bps    daily realized volatility in bps (e.g. 200 for 2%)
    """
    if quantity <= 0 or fill_price <= 0:
        return FillCostResult()

    notional = quantity * fill_price
    out = FillCostResult()

    if config.enable_spread:
        bps = max(0.0, spread_fraction) * 1e4
        if bps == 0.0:
            bps = config.fallback_spread_bps
        bps = min(bps, config.spread_cap_bps)
        out.spread_cost = notional * (bps / 2.0) / 1e4   # half-spread

    if config.enable_impact:
        impact_bps = sqrt_impact_bps(
            quantity, adv, daily_vol_bps, Y=config.Y_impact,
        )
        out.impact_cost = notional * impact_bps / 1e4

    if config.enable_commission:
        out.commission = ibkr_tiered_commission(
            side, quantity, fill_price, fees=config.fees,
        )

    return out


def adjust_fill_price(
    side: str,
    paper_price: float,
    cost: FillCostResult,
    quantity: float,
) -> float:
    """Apply spread + impact slippage to the paper fill price.

    Commission is *not* baked into fill_price — it's a separate cash deduction
    at trade time. Only spread + impact move the per-share fill.

    BUY pays slippage (fill > paper); SELL gets less (fill < paper).
    """
    if quantity <= 0:
        return paper_price
    per_share_slippage = (cost.spread_cost + cost.impact_cost) / quantity
    direction = 1.0 if side.upper() == "BUY" else -1.0
    return paper_price + direction * per_share_slippage
