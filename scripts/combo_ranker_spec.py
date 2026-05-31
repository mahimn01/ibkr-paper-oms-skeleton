"""Combination-ranker METHODOLOGY + ADVERSARIAL PRE-REGISTRATION + INTERFACE SPEC.

Companion to the RUNNABLE prototype scripts/combo_ranker.py (which already proves
the apparatus is honest: price-only -> BLOCKED, +oracle -> APPROVED). This module
is the DESIGN CONTRACT: it pins down — BEFORE any real signal is fit — the exact
panel-assembly, publication-lag, feature, label, walk-forward, n_trials, ablation,
and adversarial-guard protocol that a HONEST combination must follow to clear the
hardened gate. The other components' signal panels plug into the SignalPanel
adapter contract here. Every data-touching function is a STUB that raises
NotImplementedError stating the exact invariant its body must honour, so the
research plan is frozen (and n_trials is fixed) before fitting begins.

The runnable combo_ranker.py covers the happy path on a survivor universe with
flat-bps costs and grid-already-aligned panels. THIS spec hardens the three
things that prototype leaves open and that an adversary would exploit:
  (i)   per-signal publication-lag / as-of join  (the worst look-ahead),
  (ii)  CostModel + survivorship-haircut instead of flat bps,
  (iii) honest n_trials that counts ablations + label/residual forks, and a
        market-beta check so the "edge" can't just be net beta.

====================================================================================
THESIS  (designed to be FALSIFIED, not confirmed)
====================================================================================
One weak alt-data signal (opportunistic insider Form-4: +3.6%/trade hedged but
underpowered) cannot clear the gate. A SMALL set of weak, ORTHOGONAL,
theory-motivated signals combined by a LOW-CAPACITY cross-sectional LightGBM
ranker into a forward-21d market-residual-return rank MIGHT — iff the combination
adds independent information rather than re-discovering market beta or one dominant
signal. We certify ONLY through the unmodified build_report_card on genuinely OOS
returns at an HONEST n_trials. If the combination's OOS rank-IC does not exceed the
best single signal's, or one ablation recovers most of the edge, or the edge is
beta, or cost/DSR fail — the thesis is REJECTED for that config and we stop.

====================================================================================
THE GATE (trading_algo.quant_core.validation.report_card.build_report_card)
====================================================================================
REQUIRED gates (all four must be present AND pass for APPROVED):
  Lower 95% CI on annualised Sharpe  > 0.3   (stationary-bootstrap CI)
  PBO (CSCV)                         < 0.5   (needs trial_grid of variant OOS returns)
  Deflated Sharpe probability        > 0.95  (deflated by n_trials — the killer)
  Cost-adjusted Sharpe               > 0.3   (separate net-of-friction stream)
periods_per_year=12 (monthly book). We never touch the gate; we feed it honest OOS
returns, an honest trial_grid, an honest cost-stressed stream, and an honest n_trials.

====================================================================================
(1) PANEL ASSEMBLY — common universe + date grid + STRICT PER-SIGNAL PUBLICATION LAG
====================================================================================
Each signal releases on its OWN schedule. The deadliest look-ahead is using a value
on a date the market could not have known it. So each panel stamps, per (date,
name), the value KNOWN-AS-OF the close of that date under ITS OWN lag — there is no
single global lag, because the lags differ by an order of magnitude:

  Signal           knowledge-date field      as-of rule
  ---------------  ------------------------  -----------------------------------------
  insider Form-4   filingDate (NOT txnDate)  usable next trading-day open after
                                             filingDate  (cache already keys filingDate;
                                             see bt_insider_form4 NO-LOOK-AHEAD note)
  13F holdings     SEC filingDate            filed up to 45 cal-days post quarter-end;
                                             usable only from filingDate, STALE-fwd-fill
                                             through the next 13F; NEVER the period it
                                             describes
  short interest   FINRA dissemination date  settlement + ~8 business-day FINRA lag;
                                             usable from dissemination, fwd-fill to next
  price/vol        close                     known at close t; 1-rebalance book lag at
                                             formation, never same-bar

COMMON GRID:
  - Universe: PIT via UniverseResolver.get_universe(spec, as_of) once index_membership
    lands. UNTIL THEN a fixed survivor set (atlas_features_v3 442 names) is used ONLY
    with the survivorship flag raised loudly (§6) — an OPTIMISTIC upper bound, never
    a certification.
  - Dates: monthly rebalance grid (REBAL_TD=21), union of trading days in the price
    panel. Each signal is reindexed by BACKWARD AS-OF JOIN: on rebalance t a name
    carries the most recent value its own lag made known on/before t. No interpolation,
    no forward-peek. Missing -> NaN (LightGBM-native); a name with no insider history
    has NaN, NOT 0 — 0 is the real "no recent buying" value and we must not fabricate it.

====================================================================================
(2) FEATURES + LABEL
====================================================================================
FEATURES (per rebalance date, CROSS-SECTIONAL only — never a time-series stat that
peeks forward):
  - For each raw signal value: BOTH a cross-sectional RANK (percentile [0,1], robust
    to microcap tails) AND a winsorised Z-SCORE (keeps magnitude). The ranker picks.
  - Each date standardised using ONLY that date's cross-section.
  - Set is SMALL and FROZEN before fitting (§4). ~one or two features per signal,
    theory-named (insider_opp_rank, thirteen_f_chg_z, short_int_rank, price_resid_z).
LABEL:
  - Forward 21-td MARKET-RESIDUAL return, then cross-sectionally RANKED.
  - residual modes (PRE-REGISTERED; switching counts as a trial):
      (b) trailing_beta  [PRIMARY]: beta from data UP TO t only; resid = fwd_i -
          beta_i * fwd_mkt. Honest market-neutral.
      (a) xs_demean      [ROBUSTNESS]: resid = fwd_i - mean_j(fwd_j) that date.
          The atlas_ranker/combo_ranker prototype convention; first-order proxy.
  - LightGBM objective frozen (regression-on-residual primary; lambdarank robustness),
    optimising cross-sectional ORDER — what the decile book actually trades.

====================================================================================
(3) PURGED + EMBARGOED WALK-FORWARD  (OOS-genuine returns to the gate)
====================================================================================
  - Axis = the monthly rebalance list. scripts.purged_cv.purged_walk_forward in
    REBALANCE units: a 21-td label on a 21-td grid is 1 rebalance of overlap, so
    label_horizon=1, embargo>=1 — drops train rebalances whose forward-label reaches
    into the test block plus a buffer (the exact leak ATLAS-v7 had).
  - Train LGB on each split's TRAIN rebalances only; predict TEST; concatenate test
    predictions into ONE OOS stream. The gate sees ONLY concatenated OOS test returns.
  - Hyperparameters are NOT tuned on test. They come from §4's frozen grid; the grid's
    per-variant OOS streams BECOME the PBO trial_grid — so PBO measures exactly the
    overfit risk of the search actually run.

====================================================================================
(4) HONEST n_trials — WHY THE SIGNAL SET STAYS SMALL  (the DSR survival argument)
====================================================================================
DSR deflates Sharpe for the NUMBER OF THINGS TRIED. n_trials counts, multiplicatively:
    (#LightGBM grid variants) x (#residual/label modes run)
    x (1 full + #leave-one-out + #single-signal ablation fits) + discarded exploratory runs.
The runnable prototype passed n_trials=len(variants)=9 — that UNDERCOUNTS, because it
ignores the ablation fits and any label fork. The honest count is computed at runtime:
    PRE-REGISTERED budget (4-signal set):
      LightGBM grid                 9   (num_leaves {7,15,31} x min_child {100,200,400})
      residual modes RUN            1   (primary only in a clean run; +1 if robustness run)
      ablation fits               1+4+4 = 9   (full + 4 LOO + 4 single-signal)
    => honest n_trials ~ 9 * 1 * 9 = 81  (report the ACTUAL count produced).
Every extra signal MULTIPLIES the ablation factor and widens the grid, RAISING the
deflation bar. A 5th weak signal contributing ~0 incremental IC strictly LOWERS the
deflated-Sharpe probability. So the ONLY config that clears DSR>0.95 is a SMALL set
where each signal carries non-trivial ORTHOGONAL IC: a signal earns its place only if
it raises OOS IC MORE than it raises the deflation penalty — measured, not asserted.
The set + grid are FROZEN here; expanding after seeing results is an unbudgeted trial
that silently breaks DSR.

====================================================================================
(5) IS IT JUST BETA, OR ONE DOMINANT SIGNAL?  — beta + ablation + orthogonality
====================================================================================
  BETA CHECK: label is market-residual (§2); additionally regress the combined OOS
    book return on the market — require near-zero beta with surviving alpha. Report
    net dollar exposure per rebalance (decile L/S ~0 by construction; verify).
  DOMINANT-SIGNAL (leave-one-out): full ranker vs 4 LOO rankers vs 4 single-signal
    rankers; OOS rank-IC + Sharpe each. The combination is REAL only if the full model
    beats EVERY single-signal model AND no single LOO recovers >80% of the edge. Else
    it's one signal in a ranker costume — and that signal already failed the gate alone.
  ORTHOGONALITY: per rebalance, each raw signal's IC vs the label + pairwise Spearman
    BETWEEN signals. Low signal-signal correlation + individually-weak-but-positive IC
    is the PRECONDITION for combination to help. |pairwise corr|>0.7 -> drop the
    redundant one (lowers n_trials, helps DSR). Report the IC table + correlation matrix.

====================================================================================
(6) ADVERSARIAL FAILURE MODES + THE GUARD FOR EACH
====================================================================================
  A. LOOK-AHEAD via publication lag.  GUARD: each SignalPanel.to_grid stamps values by
     knowledge-as-of-date under ITS OWN lag (filingDate / 13F-45d / FINRA-dissemination),
     backward merge_asof ONLY. Leak test T2 plants a value at filing/dissemination date D
     and HARD-FAILS if it appears on any grid date before the next valid open after D.
     NEVER key any signal on transactionDate/period-end.
  B. SURVIVORSHIP in the long leg.  GUARD: report IWM/SPY-hedged L/S and the long leg
     SEPARATELY; apply scripts.survivorship.apply_long_leg_haircut to the long leg and
     show the gate verdict BOTH raw and haircut; stamp extra_warnings=["SURVIVORSHIP:
     fixed survivor universe; long-leg numbers are an upper bound"]. Lean the
     certification claim on the dollar-neutral hedged spread (far less survivorship-
     sensitive than long-only).
  C. TURNOVER / COST.  GUARD: net stream via scripts.costs.CostModel.per_name_cost_bps +
     cost_adjust_returns on the SAME held weights (NOT flat bps); cost-stressed stream
     to the gate's cost_adjusted_returns. The cost-adjusted Sharpe>0.3 gate is REQUIRED;
     a high-turnover microcap ranker that only works gross is correctly rejected.
  D. OVERFIT via search.  GUARD: n_trials counts every variant/label/ablation (§4) ->
     DSR; the grid's OOS streams ARE the PBO trial_grid; low-capacity model by default
     (small num_leaves, high min_child, l2); set+grid FROZEN here. Any post-hoc tweak is
     a new trial and must increment n_trials.
  E. PROGRAM-WIDE MULTIPLE HYPOTHESES.  GUARD: this is ONE pre-registered experiment;
     do not cherry-pick its config after peeking. The pre-registration below is the contract.

====================================================================================
PRE-REGISTERED TEST PROTOCOL  (run IN ORDER; do not reorder after seeing results)
====================================================================================
  T0 FREEZE signal set = {insider_opportunistic, thirteen_f_change, short_interest,
     price_residual_mom}; feature_rep = rank+z; label = §2(b) primary; LGB grid = §4's 9;
     n_splits/embargo fixed below. Record n_trials NOW, before any fit.
  T1 ASSEMBLE: each SignalPanel.to_grid(grid_dates, universe) -> (T_rebal, N), NaN where
     unknown-as-of. Assert shapes + NaN-not-zero policy + non-empty publication_lag_doc.
  T2 LEAK TEST: synthetic value at a filing/dissemination date must NOT appear on any
     earlier grid date. EVERY adapter. Hard-fail on leak.
  T3 ORTHOGONALITY: per-signal IC table + signal-signal Spearman matrix; drop |corr|>0.7.
  T4 WALK-FORWARD FIT: purged_walk_forward over rebalances; train per split, predict test;
     concat OOS predictions -> OOS gross book returns.
  T5 COST: net via CostModel; cost-stressed stream for the gate.
  T6 ABLATION: full vs 4 LOO vs 4 single-signal; OOS IC + Sharpe each; assert full beats
     every single-signal AND no LOO recovers >80% (else FALSIFIED).
  T7 BETA: regress OOS book return on market; require near-zero beta, surviving alpha.
  T8 GATE: build_report_card(returns=OOS_net, n_trials=<T0 count>, trial_grid=<9 variant
     OOS streams>, cost_adjusted_returns=<cost-stressed>, periods_per_year=12,
     extra_warnings=[survivorship]). PASS == APPROVED. Also record long-leg-haircut verdict.
  FALSIFICATION (any one => REJECT this config; STOP; do NOT tweak-and-rerun without
  incrementing n_trials):
     OOS rank-IC t-stat < ~2, or mean IC <= best single signal's; a single signal recovers
     >80% of edge; book beta materially != 0; cost-adjusted Sharpe <= 0.3; DSR prob <= 0.95.

====================================================================================
INTERFACE SPEC — signatures the other components' panels plug into
====================================================================================
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Real infra this spec binds to (imports verified to resolve from repo root):
from scripts.purged_cv import Split, purged_walk_forward  # noqa: F401
from scripts.costs import CostModel, cost_adjust_returns  # noqa: F401
from scripts.survivorship import apply_long_leg_haircut  # noqa: F401
from trading_algo.quant_core.validation.report_card import (  # noqa: F401
    REQUIRED_GATES,
    ReportCard,
    build_report_card,
)

# ---- FROZEN run constants (the pre-registration; editing these is a NEW trial) ----
REBAL_TD: int = 21                 # monthly rebalance, trading days
LABEL_FWD_TD: int = 21             # forward label horizon, trading days
LABEL_HORIZON_REBAL: int = 1       # label overlap in REBALANCE units (21td label / 21td grid)
EMBARGO_REBAL: int = 1             # extra rebalances purged before each test block
N_SPLITS: int = 6
DECILE: float = 0.10               # top/bottom decile L/S
PERIODS_PER_YEAR: int = 12         # monthly book
COST_BPS_STRESS: float = 25.0      # punitive re-run for the cost-adjusted gate
DOMINANT_RECOVERY_THRESH: float = 0.80   # LOO recovering >80% of edge => falsified
REDUNDANT_CORR_THRESH: float = 0.70      # |signal-signal corr|> this => drop one

LGB_GRID: tuple[dict, ...] = tuple(
    {"objective": "regression", "num_leaves": nl, "min_child_samples": mc,
     "learning_rate": 0.03, "lambda_l2": 5.0, "feature_fraction": 0.8,
     "verbose": -1, "_rounds": 150}
    for nl in (7, 15, 31) for mc in (100, 200, 400)
)  # 9 variants -> the PBO trial_grid columns

ResidualMode = Literal["trailing_beta", "xs_demean"]   # §2(b) primary, §2(a) robustness
FeatureRep = Literal["rank", "z", "rank_z"]            # rank_z = both, frozen default


# --------------------------------------------------------------------------
# SignalPanel: the adapter contract EVERY component's panel MUST satisfy.
# The combiner never inspects raw source files — it only calls these methods,
# so look-ahead/lag correctness is OWNED by the adapter, per-signal.
# --------------------------------------------------------------------------
class SignalPanel(Protocol):
    name: str
    #: human-readable publication-lag rule for the audit log
    #: (e.g. "usable next-open after filingDate"). Asserted non-empty by the combiner.
    publication_lag_doc: str

    def to_grid(
        self,
        grid_dates: NDArray[np.int64],      # (T_rebal,) epoch-sec rebalance dates, sorted
        universe: Sequence[str],            # N column symbols, fixed order
        *,
        as_of_strict: bool = True,          # True -> hard-fail on any future-peek
    ) -> NDArray[np.float64]:               # (T_rebal, N), NaN where unknown-as-of
        """AS-OF value of this signal on each (rebalance date, name).

        CONTRACT (enforced by the combiner + leak test T2):
          - value[t, j] uses ONLY information published on/before grid_dates[t] per
            THIS signal's lag (backward merge_asof). Knowledge-date after grid_dates[t]
            MUST be NaN.
          - NaN = "no signal known as-of t", preserved (LightGBM-native), NOT imputed
            to 0 unless 0 is the true no-event value.
          - Shape exactly (len(grid_dates), len(universe)); column j == universe[j].
        """
        ...


@dataclass
class SignalSpec:
    panel: SignalPanel
    feature_rep: FeatureRep = "rank_z"
    enabled: bool = True            # leave-one-out ablation flips this


# --------------------------------------------------------------------------
# Cross-sectional feature engineering (look-ahead-safe; per-date only)
# --------------------------------------------------------------------------
def xs_rank(row: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cross-sectional percentile rank [0,1] across valid names on ONE date.
    NaN-preserving; ties -> average rank; uses ONLY this date's cross-section."""
    raise NotImplementedError(
        "rank finite entries to [0,1]; NaN stays NaN; average-rank ties"
    )


def xs_zscore(row: NDArray[np.float64], winsor: float = 3.0) -> NDArray[np.float64]:
    """Cross-sectional winsorised z-score across names on ONE date. NaN-preserving."""
    raise NotImplementedError(
        "winsorise finite entries at +/-winsor SD then standardise; NaN stays NaN; "
        "ONLY this date's cross-section (never a time-series stat)"
    )


def build_feature_matrix(
    panels: Sequence[SignalSpec],
    grid_dates: NDArray[np.int64],
    universe: Sequence[str],
) -> tuple[NDArray[np.float64], list[str]]:
    """Assemble (T_rebal, N, F) feature tensor from ENABLED panels + theory names.

    Per enabled spec: X = panel.to_grid(grid_dates, universe); per feature_rep append
    xs_rank(X[t]) and/or xs_zscore(X[t]) along F. ASSERT panel.publication_lag_doc
    non-empty and X.shape == (len(grid_dates), len(universe)). F stays SMALL (§4).
    """
    raise NotImplementedError(
        "stack xs_rank/xs_zscore of each enabled panel.to_grid; return (T,N,F)+names; "
        "assert non-empty publication_lag_doc and matching shapes"
    )


# --------------------------------------------------------------------------
# Label: forward 21d market-residual return, then cross-sectional rank.
# --------------------------------------------------------------------------
def forward_residual_return(
    closes_daily: NDArray[np.float64],  # (T_daily, N) aligned daily closes
    rebal_idx: Sequence[int],           # positions of rebalance dates on the daily axis
    market_daily: NDArray[np.float64],  # (T_daily,) market (IWM/SPY) close series
    mode: ResidualMode = "trailing_beta",
    fwd_td: int = LABEL_FWD_TD,
) -> NDArray[np.float64]:                # (T_rebal, N) residual fwd return; NaN where undefined
    """Forward fwd_td-day return at each rebalance, residualised vs the market.

    mode='trailing_beta'  [PRIMARY]: beta_i from data UP TO t only (no look-ahead);
        resid = fwd_ret_i - beta_i * fwd_mkt_ret.
    mode='xs_demean'      [ROBUSTNESS]: resid = fwd_ret_i - mean_j(fwd_ret_j) that date.
    Switching modes is a TRIAL -> increments n_trials.
    """
    raise NotImplementedError(
        "t->t+fwd_td raw return per name at each rebalance; residualise per mode "
        "(trailing beta uses ONLY pre-t window); return (T_rebal,N). Caller ranks."
    )


def rank_label(resid: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cross-sectional rank of residual fwd return per date -> ranking label."""
    raise NotImplementedError("xs_rank each row of resid; NaN preserved")


# --------------------------------------------------------------------------
# Walk-forward fit + book formation (OOS-genuine), CostModel + beta.
# --------------------------------------------------------------------------
@dataclass
class BookResult:
    oos_gross: NDArray[np.float64]          # (T_test,) concatenated OOS monthly L/S returns
    oos_net: NDArray[np.float64]            # (T_test,) after CostModel
    oos_cost_stressed: NDArray[np.float64]  # (T_test,) punitive-cost stream for the gate
    long_leg: NDArray[np.float64]           # (T_test,) long-only leg (survivorship-exposed)
    rank_ic: NDArray[np.float64]            # (T_test,) per-rebalance OOS rank-IC
    turnover: NDArray[np.float64]           # (T_test,) per-rebalance turnover
    weights: NDArray[np.float64]            # (T_test, N) weights held (cost + beta inputs)
    market_beta: float                      # OOS book return regressed on market


def run_walk_forward_book(
    feature_tensor: NDArray[np.float64],    # (T_rebal, N, F)
    label_rank: NDArray[np.float64],        # (T_rebal, N)
    label_resid_raw: NDArray[np.float64],   # (T_rebal, N) realised P&L of the book
    closes_daily: NDArray[np.float64],
    rebal_idx: Sequence[int],
    market_daily: NDArray[np.float64],
    volumes_daily: NDArray[np.float64],
    *,
    lgb_params: dict,
    cost_model: Optional[CostModel] = None,
    cost_bps_stress: float = COST_BPS_STRESS,
    n_splits: int = N_SPLITS,
    decile: float = DECILE,
) -> BookResult:
    """Purged walk-forward train/predict, decile dollar-neutral L/S, CostModel, beta.

    purged_walk_forward(len(rebal_idx), n_splits, label_horizon=LABEL_HORIZON_REBAL,
    embargo=EMBARGO_REBAL); per Split train LGB on TRAIN rebalances, predict each TEST
    rebalance; rank predictions -> top/bottom decile equal-weight L/S; realise on
    label_resid_raw; accumulate ONLY test blocks. Net via cost_adjust_returns on held
    weights (CostModel.per_name_cost_bps); stressed via cost_bps_stress. Regress
    concatenated OOS on market_daily for market_beta. NEVER include train returns.
    """
    raise NotImplementedError(
        "train per Split on train only; predict test; decile L/S; concat OOS test; "
        "net+stressed via CostModel; per-rebal rank-IC, turnover, beta"
    )


# --------------------------------------------------------------------------
# Diagnostics: orthogonality, ablation (§5)
# --------------------------------------------------------------------------
@dataclass
class SignalDiagnostics:
    per_signal_ic: dict[str, float]         # mean OOS rank-IC of each raw signal
    per_signal_ic_t: dict[str, float]       # t-stat of each
    signal_corr: NDArray[np.float64]        # (S,S) pairwise Spearman between signals
    signal_names: list[str]
    redundant_pairs: list[tuple[str, str]]  # |corr|>REDUNDANT_CORR_THRESH


def signal_orthogonality(
    panels: Sequence[SignalSpec],
    grid_dates: NDArray[np.int64],
    universe: Sequence[str],
    label_rank: NDArray[np.float64],
) -> SignalDiagnostics:
    """Per-signal IC + signal-signal correlation matrix (precondition for combining)."""
    raise NotImplementedError(
        "per rebalance: spearman(each raw signal, label)->mean+t; spearman between "
        "signals->(S,S); flag |corr|>REDUNDANT_CORR_THRESH"
    )


@dataclass
class AblationResult:
    full_ic: float
    full_sharpe: float
    leave_one_out: dict[str, tuple[float, float]]   # signal -> (ic_without, sharpe_without)
    single_signal: dict[str, tuple[float, float]]   # signal -> (ic_alone, sharpe_alone)
    falsified_dominant: bool                        # one LOO recovers >80% OR a single == full
    n_ablation_fits: int                            # contributes to n_trials


def ablation_study(panels: Sequence[SignalSpec], *fit_inputs) -> AblationResult:
    """Full vs leave-one-out vs single-signal rankers (§5 dominant-signal guard).

    falsified_dominant := any LOO recovers > DOMINANT_RECOVERY_THRESH of full OOS Sharpe,
    OR any single-signal model matches full. Each fit here is a TRIAL (-> n_trials).
    """
    raise NotImplementedError(
        "re-run run_walk_forward_book with each signal disabled and each alone; compare "
        "OOS IC/Sharpe; set falsified_dominant per DOMINANT_RECOVERY_THRESH; "
        "n_ablation_fits = 1 + 2*len(panels)"
    )


# --------------------------------------------------------------------------
# Top-level orchestrator: runs the pre-registered protocol T0..T8.
# --------------------------------------------------------------------------
@dataclass
class ComboRunReport:
    report_card: ReportCard
    diagnostics: SignalDiagnostics
    ablation: AblationResult
    book: BookResult
    n_trials: int
    survivorship_haircut_status: str        # gate verdict after long-leg haircut
    falsified: bool
    falsification_reasons: list[str] = field(default_factory=list)


def compute_honest_n_trials(
    lgb_grid: Sequence[dict],
    n_residual_modes_run: int,
    n_signals: int,
) -> int:
    """The §4 multiplicative count, computed (never hardcoded).

    n_trials = len(lgb_grid) * n_residual_modes_run * (1 + 2*n_signals)
    where (1 + 2*n_signals) = full + n LOO + n single-signal ablation fits.
    """
    return len(lgb_grid) * max(1, n_residual_modes_run) * (1 + 2 * n_signals)


def run_combo_ranker(
    panels: Sequence[SignalSpec],
    *,
    load_price_panel: Callable[..., tuple],         # e.g. atlas_ranker.load_feature_panel
    market_loader: Callable[..., NDArray[np.float64]],  # IWM/SPY aligned to the grid
    residual_mode: ResidualMode = "trailing_beta",
    lgb_grid: Sequence[dict] = LGB_GRID,
    extra_warnings: Sequence[str] = (
        "SURVIVORSHIP: fixed survivor universe; long-leg numbers are an upper bound.",
    ),
) -> ComboRunReport:
    """Execute the pre-registered protocol T0..T8 and return the gate verdict.

    n_trials = compute_honest_n_trials(lgb_grid, #residual modes run, len(panels)) —
    passed verbatim to build_report_card. trial_grid = lgb_grid variants' OOS NET
    streams (PBO). cost_adjusted_returns = the cost-stressed stream. falsified is set
    if ANY §-protocol falsification condition trips; a falsified run is reported, never
    silently retried with a tweaked config (that would be an unbudgeted trial).
    """
    raise NotImplementedError(
        "T0 freeze+compute_honest_n_trials; T1 build_feature_matrix; T2 leak asserts; "
        "T3 signal_orthogonality; T4/T5 run_walk_forward_book over lgb_grid (grid->PBO); "
        "T6 ablation_study; T7 beta check; T8 build_report_card(... n_trials, trial_grid, "
        "cost_adjusted_returns, periods_per_year=12, extra_warnings); apply long-leg "
        "haircut, record BOTH verdicts; set falsified per the conditions"
    )


# --------------------------------------------------------------------------
# Self-test of the INTERFACE (no data, no fitting): proves the spec is wired to
# real infra and the frozen pre-registration is internally consistent.
# --------------------------------------------------------------------------
def _interface_selftest() -> int:
    # purged_cv contract holds in REBALANCE units and is non-empty for the monthly grid.
    n_rebal = 110  # ~9y of monthly rebalances
    splits = purged_walk_forward(
        n_rebal, n_splits=N_SPLITS, label_horizon=LABEL_HORIZON_REBAL,
        embargo=EMBARGO_REBAL, min_train=24,
    )
    assert splits, "no purged splits for the monthly grid"
    for s in splits:
        assert s.test[0] - s.train[-1] >= LABEL_HORIZON_REBAL + EMBARGO_REBAL
        assert max(s.train) < min(s.test)

    # PBO grid width is the frozen 9; honest n_trials >> the prototype's 9.
    assert len(LGB_GRID) == 9, "frozen LightGBM grid must be 9 variants"
    n_trials = compute_honest_n_trials(LGB_GRID, n_residual_modes_run=1, n_signals=4)
    assert n_trials == 9 * 1 * (1 + 8) == 81, f"unexpected honest n_trials: {n_trials}"
    # adding a 5th signal raises the deflation bar (more ablation fits)
    assert compute_honest_n_trials(LGB_GRID, 1, 5) > n_trials

    # The gate is the unmodified hardened one (4 required gates).
    assert len(REQUIRED_GATES) == 4, "report card must enforce 4 required gates"

    # Every data-touching entrypoint is still a stub (skeleton invariant).
    import inspect
    for fn in (xs_rank, xs_zscore, build_feature_matrix, forward_residual_return,
               rank_label, run_walk_forward_book, signal_orthogonality,
               ablation_study, run_combo_ranker):
        assert "NotImplementedError" in inspect.getsource(fn), f"{fn.__name__} must stay a stub"

    print(f"OK interface: {len(splits)} purged rebalance-splits "
          f"(sizes {[len(s.test) for s in splits]}); PBO grid={len(LGB_GRID)}; "
          f"honest n_trials(4-signal)={n_trials} (vs prototype's 9); "
          f"gate has {len(REQUIRED_GATES)} required gates; all data entrypoints are stubs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_interface_selftest())
