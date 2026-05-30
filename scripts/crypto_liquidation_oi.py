"""
Liquidation-cascade / OI-imbalance directional perp backtest, hardened gate.

Hypothesis (from crypto_alpha/edges/liquidation_cascade.py):
    Rapid OI buildup + price move against the crowded side fuels a forced
    liquidation cascade, then a reversal. OI built in an uptrend => longs
    overextended => cascade DOWN (short). OI built in a downtrend => shorts
    overextended => cascade UP (long).

We build a clean PER-ASSET time-series signal from OI change + price slope,
trade the PERP directionally next bar, charge realistic taker + slippage,
and run BOTH the reversal hypothesis and its continuation negation through
build_report_card. n_trials is kept honest and small.

DATA REALITY CHECK (printed at runtime): the OI cache only spans the most
recent ~500 hourly bars (~21 days) for every asset, because the exchange OI
endpoint serves limited history. The swap/funding caches span 5.4y, but OI
gates the joinable window. So the OI-driven signal is testable only on
~0.05 asset-years per asset. We report the ACTUAL gate output regardless.
"""

from __future__ import annotations

import numpy as np

from trading_algo.quant_core.validation.report_card import build_report_card

CACHE = "/Users/mahimnpatel/Documents/Dev/randomThings/crypto_data_cache"
ASSETS = ["BTC", "ETH", "SOL"]
TPL = "{cache}/{a}_USDT_USDT_{kind}_2020-10-01_2026-03-01.npz"

# Realistic perp costs (directional, single perp leg per round trip = 2 sides)
TAKER_BPS = 5.0          # per side
SLIP_BPS = {"BTC": 3.0, "ETH": 3.0, "SOL": 6.0}  # per side
PUNITIVE = 2.0           # punitive stack for cost_adjusted_returns

# Signal params (fixed a-priori; the only searched axis is reversal vs continuation)
LOOKBACK = 48            # OI/price lookback (bars) for buildup estimation
MIN_OI_CHANGE = 0.03     # >=3% OI change over lookback to act
IMBALANCE_THR = 1.5      # |oi_change| / |price_change| coiled-spring ratio
HOLD = 8                 # hold the directional perp 8h (one funding cycle)
PERIODS_PER_YEAR = 24 * 365  # hourly rebalance accounting


def load(a: str, kind: str):
    d = np.load(TPL.format(cache=CACHE, a=a, kind=kind))
    return d


def build_asset_panel(a: str):
    """Return (ts, close, oi) aligned on the OI-gated hourly window."""
    oi = load(a, "oi_1h")
    sw = load(a, "swap_1h")
    oi_ts, oi_v = oi["ts"], oi["oi"]
    sw_ts, sw_c = sw["ts"], sw["ohlcv"][:, 3]
    # join on common hourly timestamps (OI is the limiting set)
    sw_map = {int(t): c for t, c in zip(sw_ts, sw_c)}
    ts, close, oiv = [], [], []
    for t, o in zip(oi_ts, oi_v):
        ti = int(t)
        if ti in sw_map and o > 0:
            ts.append(ti)
            close.append(sw_map[ti])
            oiv.append(o)
    return np.array(ts), np.array(close, float), np.array(oiv, float)


def signal_returns(reversal: bool):
    """
    Generate per-bar strategy returns pooled across assets.

    reversal=True  -> trade OPPOSITE the OI-buildup direction (cascade/reversal)
    reversal=False -> trade WITH the buildup direction (continuation negation)

    Positions are non-overlapping (re-evaluate every HOLD bars) so funding/cost
    accounting is clean and there is no look-ahead: at decision bar i we use
    only close[:i+1] and oi[:i+1], then realise the forward HOLD-bar perp return
    starting next bar.
    """
    gross_legs, cost_legs = [], []
    for a in ASSETS:
        ts, close, oi = build_asset_panel(a)
        n = len(close)
        if n < LOOKBACK + HOLD + 2:
            continue
        logc = np.log(close)
        i = LOOKBACK
        while i + HOLD < n:
            # --- signal at decision bar i, info up to and including i ---
            oi_w = oi[i - LOOKBACK : i + 1]
            p_w = close[i - LOOKBACK : i + 1]
            oi_chg = (oi_w[-1] - oi_w[0]) / oi_w[0] if oi_w[0] > 0 else 0.0
            p_chg = (p_w[-1] - p_w[0]) / p_w[0] if p_w[0] > 0 else 0.0
            x = np.arange(len(p_w))
            slope = np.polyfit(x, p_w, 1)[0]

            act = abs(oi_chg) >= MIN_OI_CHANGE and (
                abs(oi_chg) >= IMBALANCE_THR * max(abs(p_chg), 1e-6)
            )
            if not act or oi_chg <= 0:
                i += 1
                continue

            # OI built up (oi_chg>0). buildup_dir: uptrend=+1 longs, downtrend=-1 shorts
            buildup_up = slope > 0
            # reversal: longs built in uptrend -> cascade DOWN -> short (-1)
            if reversal:
                pos = -1 if buildup_up else +1
            else:
                pos = +1 if buildup_up else -1

            # --- realise forward HOLD-bar perp return, entry NEXT bar ---
            r = pos * (logc[i + HOLD] - logc[i + 1])
            r = float(np.expm1(r))  # to simple return

            rt_cost = 2.0 * (TAKER_BPS + SLIP_BPS[a]) / 1e4  # in+out, both sides
            gross_legs.append(r)
            cost_legs.append(r - rt_cost)
            i += HOLD  # non-overlapping
    return np.array(gross_legs), np.array(cost_legs)


def main():
    print("=== Liquidation-cascade / OI-imbalance perp backtest ===")
    total = 0
    for a in ASSETS:
        ts, close, oi = build_asset_panel(a)
        total += len(close)
        print(f"  {a}: joinable OI+swap hourly bars = {len(close)}")
    print(f"  Pooled hourly bars w/ OI = {total} "
          f"(~{total/8760:.3f} asset-years). OI history is the binding constraint.")

    g_rev, c_rev = signal_returns(reversal=True)
    g_con, c_con = signal_returns(reversal=False)
    print(f"  Reversal trades: {len(g_rev)}  | Continuation trades: {len(g_con)}")

    if len(g_rev) < 10:
        print("  INSUFFICIENT TRADES -> cannot gate.")
        return None, len(g_rev)

    # Trade cadence = HOLD bars -> periods per year for annualisation
    ppy_trade = int(PERIODS_PER_YEAR / HOLD)

    # n_trials = 2 (reversal + continuation). DSR deflates for this.
    # Pick the reversal hypothesis as the headline (the edge's actual claim),
    # but build a trial_grid from both for CSCV PBO.
    # Pad to equal length for the grid.
    m = min(len(c_rev), len(c_con))
    grid = np.column_stack([c_rev[:m], c_con[:m]])

    # 2x punitive cost stack: gross minus twice the realised cost gap.
    punitive_rev = g_rev - PUNITIVE * (g_rev - c_rev)

    card = build_report_card(
        strategy_name="liquidation_oi_reversal",
        returns=g_rev,
        n_trials=2,
        trial_grid=grid,
        cost_adjusted_returns=punitive_rev,
        periods_per_year=ppy_trade,
        seed=42,
    )
    print("\n=== REPORT CARD (reversal hypothesis, headline) ===")
    print(card)
    return card, len(g_rev)


if __name__ == "__main__":
    main()
