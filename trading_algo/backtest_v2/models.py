"""
Backtest data models - comprehensive types for enterprise-level backtesting.

These models capture every detail needed for deep analysis of backtest results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
import statistics
import math


class FillType(Enum):
    """How an order was filled."""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()


@dataclass
class Bar:
    """A single price bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int = 0
    vwap: float = 0.0

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open


@dataclass
class BacktestTrade:
    """
    A complete trade record with all analysis data.

    Contains everything needed for post-trade analysis.
    """
    id: str
    symbol: str
    direction: str  # "LONG" or "SHORT"

    # Entry
    entry_time: datetime
    entry_price: float
    entry_bar_index: int
    quantity: int = 1

    # Exit
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_bar_index: Optional[int] = None
    exit_reason: str = ""

    # Risk management
    initial_stop: Optional[float] = None
    initial_target: Optional[float] = None
    final_stop: Optional[float] = None

    # P&L
    gross_pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    net_pnl: float = 0.0
    pnl_percent: float = 0.0

    # Excursions (tracked during trade)
    max_favorable_excursion: float = 0.0  # Best unrealized P&L
    max_adverse_excursion: float = 0.0    # Worst unrealized P&L
    mfe_price: float = 0.0  # Price at MFE
    mae_price: float = 0.0  # Price at MAE

    # Timing
    bars_held: int = 0
    duration_minutes: int = 0

    # Signal info
    signal_confidence: float = 0.0
    signal_reason: str = ""
    edge_votes: Dict[str, str] = field(default_factory=dict)

    # Market context at entry
    market_regime: str = ""
    entry_atr: float = 0.0
    entry_vwap: float = 0.0

    @property
    def is_winner(self) -> bool:
        return self.net_pnl > 0

    @property
    def r_multiple(self) -> Optional[float]:
        """Return in terms of initial risk (R)."""
        if self.initial_stop is None or self.entry_price == 0:
            return None
        risk = abs(self.entry_price - self.initial_stop)
        if risk == 0:
            return None
        return self.net_pnl / (risk * self.quantity)

    @property
    def efficiency(self) -> Optional[float]:
        """How much of MFE was captured (0-1)."""
        if self.max_favorable_excursion <= 0:
            return None
        return self.net_pnl / self.max_favorable_excursion

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "direction": self.direction,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "entry_price": self.entry_price,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "quantity": self.quantity,
            "initial_stop": self.initial_stop,
            "initial_target": self.initial_target,
            "gross_pnl": self.gross_pnl,
            "commission": self.commission,
            "slippage": self.slippage,
            "net_pnl": self.net_pnl,
            "pnl_percent": self.pnl_percent,
            "max_favorable_excursion": self.max_favorable_excursion,
            "max_adverse_excursion": self.max_adverse_excursion,
            "bars_held": self.bars_held,
            "duration_minutes": self.duration_minutes,
            "signal_confidence": self.signal_confidence,
            "signal_reason": self.signal_reason,
            "market_regime": self.market_regime,
            "r_multiple": self.r_multiple,
            "efficiency": self.efficiency,
        }


@dataclass
class DailyResult:
    """P&L and statistics for a single trading day."""
    date: date
    starting_equity: float
    ending_equity: float

    # P&L
    gross_pnl: float = 0.0
    commissions: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0

    # Trades
    trades_taken: int = 0
    trades_won: int = 0
    trades_lost: int = 0

    # Intraday
    high_water_mark: float = 0.0
    low_water_mark: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # Time
    first_trade_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None

    # Market context
    market_regime: str = ""
    spy_return: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.trades_taken == 0:
            return 0.0
        return (self.trades_won / self.trades_taken) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "starting_equity": self.starting_equity,
            "ending_equity": self.ending_equity,
            "gross_pnl": self.gross_pnl,
            "commissions": self.commissions,
            "net_pnl": self.net_pnl,
            "return_pct": self.return_pct,
            "trades_taken": self.trades_taken,
            "trades_won": self.trades_won,
            "trades_lost": self.trades_lost,
            "win_rate": self.win_rate,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_pct": self.max_drawdown_pct,
            "market_regime": self.market_regime,
            "spy_return": self.spy_return,
        }


@dataclass
class DrawdownPeriod:
    """A drawdown period from peak to recovery."""
    start_date: date
    trough_date: date
    end_date: Optional[date]  # None if still in drawdown

    peak_equity: float
    trough_equity: float
    current_equity: float

    drawdown_amount: float
    drawdown_percent: float

    duration_days: int
    recovery_days: Optional[int]

    @property
    def is_recovered(self) -> bool:
        return self.end_date is not None


@dataclass
class BacktestMetrics:
    """
    Comprehensive performance metrics from a backtest.

    All the statistics needed to evaluate a strategy.
    """
    # Period
    start_date: date
    end_date: date
    trading_days: int

    # Capital
    initial_capital: float
    final_capital: float

    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    cagr: float = 0.0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0  # days
    avg_drawdown: float = 0.0
    avg_drawdown_pct: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0

    win_rate: float = 0.0
    loss_rate: float = 0.0

    # P&L
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0

    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_trade: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Risk metrics
    avg_win_loss_ratio: float = 0.0
    expectancy: float = 0.0
    expectancy_ratio: float = 0.0  # Expectancy / avg loss

    # Time metrics
    avg_bars_in_winner: float = 0.0
    avg_bars_in_loser: float = 0.0
    avg_trade_duration_minutes: float = 0.0

    # Daily statistics
    avg_daily_pnl: float = 0.0
    std_daily_pnl: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    profitable_days: int = 0
    losing_days: int = 0
    daily_win_rate: float = 0.0

    # Monthly statistics
    avg_monthly_return: float = 0.0
    best_month: float = 0.0
    worst_month: float = 0.0
    profitable_months: int = 0

    # Streak analysis
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_consecutive_wins: float = 0.0
    avg_consecutive_losses: float = 0.0

    # Exposure
    time_in_market_pct: float = 0.0
    avg_position_size: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "period": {
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "trading_days": self.trading_days,
            },
            "capital": {
                "initial": self.initial_capital,
                "final": self.final_capital,
            },
            "returns": {
                "total_return": self.total_return,
                "total_return_pct": self.total_return_pct,
                "annualized_return": self.annualized_return,
                "cagr": self.cagr,
            },
            "risk_adjusted": {
                "sharpe_ratio": self.sharpe_ratio,
                "sortino_ratio": self.sortino_ratio,
                "calmar_ratio": self.calmar_ratio,
            },
            "drawdown": {
                "max_drawdown": self.max_drawdown,
                "max_drawdown_pct": self.max_drawdown_pct,
                "max_drawdown_duration_days": self.max_drawdown_duration,
                "avg_drawdown_pct": self.avg_drawdown_pct,
            },
            "trades": {
                "total": self.total_trades,
                "winners": self.winning_trades,
                "losers": self.losing_trades,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
            },
            "pnl": {
                "gross_profit": self.gross_profit,
                "gross_loss": self.gross_loss,
                "net_profit": self.net_profit,
                "avg_win": self.avg_win,
                "avg_loss": self.avg_loss,
                "avg_trade": self.avg_trade,
                "largest_win": self.largest_win,
                "largest_loss": self.largest_loss,
                "expectancy": self.expectancy,
            },
            "daily": {
                "avg_daily_pnl": self.avg_daily_pnl,
                "std_daily_pnl": self.std_daily_pnl,
                "best_day": self.best_day,
                "worst_day": self.worst_day,
                "profitable_days": self.profitable_days,
                "losing_days": self.losing_days,
                "daily_win_rate": self.daily_win_rate,
            },
            "streaks": {
                "max_consecutive_wins": self.max_consecutive_wins,
                "max_consecutive_losses": self.max_consecutive_losses,
            },
            "exposure": {
                "time_in_market_pct": self.time_in_market_pct,
            },
        }


@dataclass
class EquityPoint:
    """A point on the equity curve."""
    timestamp: datetime
    equity: float
    cash: float
    position_value: float
    drawdown: float
    drawdown_pct: float
    high_water_mark: float


@dataclass
class CostProfile:
    """Transaction cost profile for backtesting."""
    commission_per_share: float
    min_commission: float
    slippage_pct: float
    name: str = "custom"


# Pre-built cost profiles for common brokers
IBKR_TIERED_COSTS = CostProfile(
    commission_per_share=0.0035,
    min_commission=0.35,
    slippage_pct=0.0002,  # 2bp for liquid large-caps
    name="IBKR Tiered (US Equities)",
)

IBKR_FIXED_COSTS = CostProfile(
    commission_per_share=0.005,
    min_commission=1.0,
    slippage_pct=0.0003,  # 3bp general estimate
    name="IBKR Fixed (US Equities)",
)

CONSERVATIVE_COSTS = CostProfile(
    commission_per_share=0.005,
    min_commission=1.0,
    slippage_pct=0.0005,  # 5bp conservative
    name="Conservative Estimate",
)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    # Strategy
    strategy_name: str
    strategy_version: str = "1.0.0"
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    # Data
    symbols: List[str] = field(default_factory=list)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    bar_size: str = "5 mins"
    data_source: str = "IBKR"

    # Capital
    initial_capital: float = 100_000
    position_size_pct: float = 0.02  # 2% per position
    max_positions: int = 5

    # Costs (defaults match IBKR Fixed pricing)
    commission_per_share: float = 0.005
    min_commission: float = 1.0
    slippage_pct: float = 0.0005  # 0.05%

    # Execution
    fill_on_close: bool = False  # Fill at close vs next open
    allow_shorting: bool = True

    # Realistic execution v2 (PLAN.md §2.3-2.4).
    # When set, the engine uses Corwin-Schultz spread + sqrt impact +
    # IBKR tiered commission instead of the flat slippage_pct above.
    # When None, legacy flat-bps slippage is used (backwards-compatible).
    cost_model_config: Optional[Any] = None  # CostModelConfig | None
    """If supplied, replaces the flat slippage_pct + commission_per_share
    pair with a realistic decomposed fill: half-spread (CS+AR fallback),
    square-root impact, IBKR tiered commission. See cost_model.py."""

    # ExecutionPolicy: bar T signal -> fill where?
    #   "same_bar_close": legacy (look-ahead)
    #   "next_bar_open":  signal queued on T, fill at T+1 open  (recommended)
    #   "next_bar_vwap":  signal queued on T, fill at T+1 VWAP
    execution_policy: str = "same_bar_close"
    """String form so configs serialise cleanly. Convert to ExecutionPolicy
    enum in engine code."""

    # Borrow cost for shorts. None disables; when set, applies daily
    # ACT/360 accrual on short notional.
    default_borrow_rate_bps: Optional[float] = None

    # Risk
    max_daily_loss_pct: float = 0.05  # 5% daily loss limit
    max_drawdown_pct: float = 0.20   # 20% max drawdown

    @classmethod
    def with_cost_profile(
        cls,
        strategy_name: str,
        cost_profile: CostProfile,
        **kwargs,
    ) -> "BacktestConfig":
        """Create a config with a specific cost profile applied."""
        return cls(
            strategy_name=strategy_name,
            commission_per_share=cost_profile.commission_per_share,
            min_commission=cost_profile.min_commission,
            slippage_pct=cost_profile.slippage_pct,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy": {
                "name": self.strategy_name,
                "version": self.strategy_version,
                "params": self.strategy_params,
            },
            "data": {
                "symbols": self.symbols,
                "start_date": self.start_date.isoformat() if self.start_date else None,
                "end_date": self.end_date.isoformat() if self.end_date else None,
                "bar_size": self.bar_size,
                "data_source": self.data_source,
            },
            "capital": {
                "initial": self.initial_capital,
                "position_size_pct": self.position_size_pct,
                "max_positions": self.max_positions,
            },
            "costs": {
                "commission_per_share": self.commission_per_share,
                "min_commission": self.min_commission,
                "slippage_pct": self.slippage_pct,
            },
            "execution": {
                "fill_on_close": self.fill_on_close,
                "allow_shorting": self.allow_shorting,
            },
            "risk": {
                "max_daily_loss_pct": self.max_daily_loss_pct,
                "max_drawdown_pct": self.max_drawdown_pct,
            },
        }


@dataclass
class BacktestResults:
    """
    Complete results from a backtest run.

    Contains everything needed for comprehensive analysis.
    """
    # Identification
    run_id: str
    run_timestamp: datetime

    # Configuration
    config: BacktestConfig

    # Core results
    metrics: BacktestMetrics
    trades: List[BacktestTrade]
    daily_results: List[DailyResult]
    equity_curve: List[EquityPoint]
    drawdown_periods: List[DrawdownPeriod]

    # Signals (all, not just executed)
    signals: List[Dict[str, Any]] = field(default_factory=list)

    # Raw data info
    bars_processed: int = 0
    data_gaps: List[Dict[str, Any]] = field(default_factory=list)

    # Execution info
    execution_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_profitable(self) -> bool:
        return self.metrics.net_profit > 0

    def get_trades_for_date(self, d: date) -> List[BacktestTrade]:
        """Get all trades that exited on a specific date."""
        return [t for t in self.trades if t.exit_time and t.exit_time.date() == d]

    def get_monthly_returns(self) -> Dict[str, float]:
        """Get returns grouped by month."""
        monthly: Dict[str, float] = {}
        for dr in self.daily_results:
            key = dr.date.strftime("%Y-%m")
            monthly[key] = monthly.get(key, 0) + dr.net_pnl
        return monthly

    def get_weekday_analysis(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by day of week."""
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        analysis = {day: {"pnl": 0.0, "trades": 0, "wins": 0} for day in weekdays}

        for dr in self.daily_results:
            day = weekdays[dr.date.weekday()]
            analysis[day]["pnl"] += dr.net_pnl
            analysis[day]["trades"] += dr.trades_taken
            analysis[day]["wins"] += dr.trades_won

        return analysis

    def get_hourly_analysis(self) -> Dict[int, Dict[str, float]]:
        """Analyze performance by hour of entry."""
        hourly = {h: {"pnl": 0.0, "trades": 0, "wins": 0} for h in range(9, 17)}

        for trade in self.trades:
            hour = trade.entry_time.hour
            if 9 <= hour < 17:
                hourly[hour]["pnl"] += trade.net_pnl
                hourly[hour]["trades"] += 1
                if trade.is_winner:
                    hourly[hour]["wins"] += 1

        return hourly
