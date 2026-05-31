"""
Enterprise-level backtesting engine.

Provides accurate simulation of trading strategies with:
- Realistic fill simulation (slippage, commissions)
- Multi-symbol support
- Comprehensive tracking of all metrics
- Support for any strategy via adapter pattern
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
import statistics
import math
import time as time_module

from .models import (
    Bar,
    BacktestTrade,
    DailyResult,
    DrawdownPeriod,
    BacktestMetrics,
    EquityPoint,
    BacktestConfig,
    BacktestResults,
)
from .cost_model import (
    CostModelConfig,
    adjust_fill_price,
    borrow_charge,
    compute_fill_cost,
    daily_effective_spread,
    ibkr_tiered_commission,
)
from .execution_policy import (
    ExecutionPolicy,
    PendingOrder,
    fill_price_for_policy,
    policy_introduces_lookahead,
)


class StrategyProtocol(Protocol):
    """Protocol for strategies compatible with the backtest engine."""

    def update_asset(
        self,
        symbol: str,
        timestamp: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """Update the strategy with a new bar."""
        ...

    def generate_signal(self, symbol: str, timestamp: datetime) -> Any:
        """Generate a trading signal."""
        ...

    def clear_positions(self) -> None:
        """Clear all positions (for warmup)."""
        ...

    @property
    def positions(self) -> Dict[str, Any]:
        """Current positions."""
        ...


class _BlankStrategyRef:
    """Stand-in passed to _open_position from the pending-order drain.

    The caller path doesn't have the live strategy in scope; the engine
    only reads `asset_states` for logging context. Returning empty here
    makes the open path safe under deferred fills."""
    asset_states: Dict[str, Any] = {}


@dataclass
class OpenPosition:
    """An open position during backtesting."""
    symbol: str
    direction: int  # 1 for long, -1 for short
    entry_price: float
    entry_time: datetime
    entry_bar_index: int
    quantity: int
    stop_loss: Optional[float]
    take_profit: Optional[float]
    signal_confidence: float
    signal_reason: str
    edge_votes: Dict[str, str]
    market_regime: str
    atr: float
    vwap: float

    # Tracking
    best_price: float = 0.0
    worst_price: float = 0.0
    trailing_stop: Optional[float] = None
    trailing_active: bool = False


class BacktestEngine:
    """
    Enterprise-level backtesting engine.

    Features:
    - Accurate simulation with slippage and commissions
    - Multi-symbol support
    - Comprehensive metrics calculation
    - Position tracking with MAE/MFE
    - Daily P&L breakdown
    - Drawdown analysis
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

        # State
        self.equity = config.initial_capital
        self.cash = config.initial_capital
        self.positions: Dict[str, OpenPosition] = {}

        # Tracking
        self.trades: List[BacktestTrade] = []
        self.signals: List[Dict[str, Any]] = []
        self.equity_history: List[EquityPoint] = []
        self.daily_results: Dict[date, DailyResult] = {}

        # Metrics tracking
        self.high_water_mark = config.initial_capital
        self.current_drawdown = 0.0
        self.current_drawdown_pct = 0.0
        self.drawdown_start_date: Optional[date] = None
        self.drawdown_periods: List[DrawdownPeriod] = []

        # Bar tracking
        self.current_bar_index = 0
        self.current_date: Optional[date] = None
        self.bars_in_market = 0
        self.total_bars = 0

        # Errors/warnings
        self.errors: List[str] = []
        self.warnings: List[str] = []

        # PLAN.md §2.4 — pending-order queue for non-look-ahead fills.
        # Keyed by symbol; one pending entry per symbol at most.
        self._pending_orders: Dict[str, PendingOrder] = {}

        # Resolve string policy from config to enum once.
        try:
            self._execution_policy = ExecutionPolicy(config.execution_policy)
        except ValueError:
            self.warnings.append(
                f"unknown execution_policy={config.execution_policy!r}; "
                "falling back to SAME_BAR_CLOSE"
            )
            self._execution_policy = ExecutionPolicy.SAME_BAR_CLOSE

        # Per-symbol rolling window of (open, high, low, close) used for
        # Corwin-Schultz / Abdi-Ranaldo spread estimation.
        self._spread_window: Dict[str, List[Tuple[float, float, float, float]]] = (
            defaultdict(list)
        )

        # Per-symbol 20-day ADV cache (rolling).
        self._volume_window: Dict[str, List[int]] = defaultdict(list)

        # Per-symbol daily realized vol in bps (rolling).
        self._return_window: Dict[str, List[float]] = defaultdict(list)

        if policy_introduces_lookahead(self._execution_policy):
            self.warnings.append(
                "execution_policy=SAME_BAR_CLOSE introduces look-ahead bias; "
                "validator will flag results from this run"
            )

    # ------------------------------------------------------------------
    # Cost-model helpers (PLAN.md §2.3).
    #
    # These maintain rolling per-symbol windows of OHLCV/return/volume so
    # we can estimate effective spread (Corwin-Schultz / Abdi-Ranaldo
    # fallback), 20-day ADV, and realised daily vol on the fly without
    # round-tripping to the PIT store. They're updated lazily — only when
    # cost_model_config is set, the rolling state is populated.
    # ------------------------------------------------------------------

    _SPREAD_WIN_LEN: int = 22       # 22 bars ~ 1 month of daily data
    _ADV_WIN_LEN: int = 20          # 20-day ADV
    _RETURN_WIN_LEN: int = 60       # 60-bar window for vol estimate

    def _record_bar_for_costs(self, symbol: str, bar: Bar) -> None:
        """Update the rolling windows used by the cost estimators.

        No-op when cost_model_config is None; that path uses flat slippage.
        Called from _process_bar before signal generation.
        """
        if self.config.cost_model_config is None:
            return
        ohlc = (float(bar.open), float(bar.high), float(bar.low), float(bar.close))
        sw = self._spread_window[symbol]
        sw.append(ohlc)
        if len(sw) > self._SPREAD_WIN_LEN:
            del sw[0]

        vw = self._volume_window[symbol]
        vw.append(int(bar.volume))
        if len(vw) > self._ADV_WIN_LEN:
            del vw[0]

        rw = self._return_window[symbol]
        if len(sw) >= 2 and sw[-2][3] > 0:
            ret = (ohlc[3] - sw[-2][3]) / sw[-2][3]
            rw.append(float(ret))
        if len(rw) > self._RETURN_WIN_LEN:
            del rw[0]

    def _estimate_spread_fraction(self, symbol: str) -> float:
        """Effective spread for `symbol` from the most recent two bars.

        Falls back to cost_model_config.fallback_spread_bps when there
        isn't enough history yet. Returned as a fraction of price (e.g.
        0.0010 = 10 bps).
        """
        cmc = self.config.cost_model_config
        bars = self._spread_window.get(symbol, [])
        if len(bars) < 2:
            return (cmc.fallback_spread_bps if cmc else 5.0) / 1e4
        s = daily_effective_spread(bars)
        if s <= 0:
            return (cmc.fallback_spread_bps if cmc else 5.0) / 1e4
        # Cap to protect against pathological bars.
        cap = (cmc.spread_cap_bps if cmc else 200.0) / 1e4
        return min(s, cap)

    def _estimate_adv(self, symbol: str) -> float:
        """20-day average daily volume for `symbol`. Returns 1.0 (zero
        impact) when insufficient history; the impact model is bounded
        below by min_participation so this is safe."""
        vols = self._volume_window.get(symbol, [])
        if not vols:
            return 1.0
        return float(sum(vols) / len(vols))

    def _estimate_daily_vol_bps(self, symbol: str) -> float:
        """Realised daily vol of bar-to-bar log returns, in basis points.

        With <2 returns of history, returns 200 bps (a conservative typical
        US-equity daily vol).
        """
        rets = self._return_window.get(symbol, [])
        if len(rets) < 2:
            return 200.0
        try:
            sd = statistics.stdev(rets)
        except statistics.StatisticsError:
            return 200.0
        return float(sd) * 1e4

    def run(
        self,
        strategy: StrategyProtocol,
        data: Dict[str, List[Bar]],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> BacktestResults:
        """
        Run a complete backtest.

        Args:
            strategy: The trading strategy to test
            data: Historical data as {symbol: [bars]}
            progress_callback: Optional callback for progress updates

        Returns:
            BacktestResults with all metrics and trade details
        """
        start_time = time_module.time()
        run_id = f"BT-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:6]}"

        if progress_callback:
            progress_callback(0.0, "Starting backtest...")

        # Validate data
        if not data:
            raise ValueError("No data provided")

        # Get all unique timestamps across all symbols
        all_timestamps: List[datetime] = []
        for symbol, bars in data.items():
            all_timestamps.extend(bar.timestamp for bar in bars)
        all_timestamps = sorted(set(all_timestamps))

        if not all_timestamps:
            raise ValueError("No timestamps in data")

        # Create bar lookup for fast access
        bar_lookup: Dict[str, Dict[datetime, Bar]] = {
            symbol: {bar.timestamp: bar for bar in bars}
            for symbol, bars in data.items()
        }

        # Warmup period (first 20% or 50 bars minimum)
        warmup_count = max(50, int(len(all_timestamps) * 0.05))
        warmup_timestamps = all_timestamps[:warmup_count]
        trading_timestamps = all_timestamps[warmup_count:]

        if progress_callback:
            progress_callback(0.05, f"Warming up with {warmup_count} bars...")

        # Warmup phase
        for ts in warmup_timestamps:
            self._process_bar(strategy, bar_lookup, ts, is_warmup=True)

        strategy.clear_positions()
        self.current_bar_index = 0

        if progress_callback:
            progress_callback(0.1, "Starting trading simulation...")

        # Trading phase
        total_trading_bars = len(trading_timestamps)
        for i, ts in enumerate(trading_timestamps):
            self._process_bar(strategy, bar_lookup, ts, is_warmup=False)

            # Progress update every 1%
            if progress_callback and i % max(1, total_trading_bars // 100) == 0:
                pct = 0.1 + (i / total_trading_bars) * 0.8
                progress_callback(pct, f"Processing {ts.strftime('%Y-%m-%d')}...")

        # Close any remaining positions at end
        self._close_all_positions(trading_timestamps[-1] if trading_timestamps else datetime.now())

        if progress_callback:
            progress_callback(0.9, "Calculating metrics...")

        # Calculate final metrics
        metrics = self._calculate_metrics()

        # Build daily results list
        daily_list = sorted(self.daily_results.values(), key=lambda x: x.date)

        if progress_callback:
            progress_callback(0.95, "Finalizing results...")

        # Create results
        results = BacktestResults(
            run_id=run_id,
            run_timestamp=datetime.now(),
            config=self.config,
            metrics=metrics,
            trades=self.trades,
            daily_results=daily_list,
            equity_curve=self.equity_history,
            drawdown_periods=self.drawdown_periods,
            signals=self.signals,
            bars_processed=self.total_bars,
            execution_time_seconds=time_module.time() - start_time,
            errors=self.errors,
            warnings=self.warnings,
        )

        if progress_callback:
            progress_callback(1.0, "Backtest complete!")

        return results

    def _process_bar(
        self,
        strategy: StrategyProtocol,
        bar_lookup: Dict[str, Dict[datetime, Bar]],
        timestamp: datetime,
        is_warmup: bool,
    ) -> None:
        """Process a single bar across all symbols."""
        self.current_bar_index += 1
        self.total_bars += 1

        # Track date changes
        bar_date = timestamp.date()
        if self.current_date != bar_date:
            if self.current_date is not None and not is_warmup:
                self._finalize_day(self.current_date)
            self.current_date = bar_date
            if not is_warmup:
                self._start_day(bar_date)

        # Update strategy with ALL available bars (including reference assets like SPY, QQQ)
        # This is crucial for strategies that need market context (regime detection, etc.)
        for symbol in bar_lookup.keys():
            if timestamp in bar_lookup[symbol]:
                bar = bar_lookup[symbol][timestamp]
                strategy.update_asset(
                    symbol=symbol,
                    timestamp=timestamp,
                    open_price=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )

        # Update rolling cost-model state for tracked symbols.
        # Done during warmup too so the windows are populated by the time
        # trading starts.
        for symbol in self.config.symbols:
            if symbol in bar_lookup and timestamp in bar_lookup[symbol]:
                self._record_bar_for_costs(symbol, bar_lookup[symbol][timestamp])

        if is_warmup:
            return

        # Drain any pending orders queued on previous bar (NEXT_BAR_OPEN /
        # NEXT_BAR_VWAP policies). Fills happen at THIS bar's open/vwap
        # before strategy generates new signals on this bar.
        self._drain_pending_orders(bar_lookup, timestamp)

        # Apply borrow accrual for any short positions held overnight.
        if self.config.default_borrow_rate_bps is not None:
            self._accrue_borrow_costs(bar_lookup, timestamp)

        # Update existing positions
        for symbol in list(self.positions.keys()):
            if symbol in bar_lookup and timestamp in bar_lookup[symbol]:
                bar = bar_lookup[symbol][timestamp]
                self._update_position(symbol, bar, timestamp)

        # Track time in market
        if self.positions:
            self.bars_in_market += 1

        # Generate signals for each symbol
        for symbol in self.config.symbols:
            if symbol not in bar_lookup or timestamp not in bar_lookup[symbol]:
                continue

            bar = bar_lookup[symbol][timestamp]

            # Check if we can take new positions
            if len(self.positions) >= self.config.max_positions:
                continue

            if symbol in self.positions:
                continue  # Already have position

            try:
                signal = strategy.generate_signal(symbol, timestamp)
                self._process_signal(signal, symbol, bar, timestamp, strategy)
            except Exception as e:
                self.errors.append(f"{timestamp}: Error generating signal for {symbol}: {e}")

        # Record equity point
        self._record_equity_point(timestamp, bar_lookup)

    def _process_signal(
        self,
        signal: Any,
        symbol: str,
        bar: Bar,
        timestamp: datetime,
        strategy: StrategyProtocol,
    ) -> None:
        """Process a trading signal."""
        if signal is None:
            return

        action = getattr(signal, 'action', None)
        if action is None or action == 'hold':
            return

        # Record signal
        self.signals.append({
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "action": action,
            "price": bar.close,
            "confidence": getattr(signal, 'confidence', 0),
            "reason": getattr(signal, 'reason', ''),
        })

        # Check daily loss limit
        if self.current_date and self.current_date in self.daily_results:
            daily = self.daily_results[self.current_date]
            if daily.net_pnl < -self.config.initial_capital * self.config.max_daily_loss_pct:
                self.warnings.append(f"{timestamp}: Daily loss limit hit, skipping trade")
                return

        # Lookahead-safe routing: under NEXT_BAR_* policies the signal is
        # *queued* for fill on the next bar, not executed against this bar.
        defer = self._execution_policy is not ExecutionPolicy.SAME_BAR_CLOSE

        if action in ('buy', 'short'):
            if defer:
                self._enqueue_pending(signal, symbol, timestamp, action)
            else:
                self._open_position(signal, symbol, bar, timestamp, strategy)
        elif action in ('sell', 'cover'):
            if symbol in self.positions:
                if defer:
                    self._enqueue_pending(signal, symbol, timestamp, action)
                else:
                    self._close_position(
                        symbol,
                        bar.close,
                        timestamp,
                        getattr(signal, 'reason', 'Signal exit'),
                    )

    def _open_position(
        self,
        signal: Any,
        symbol: str,
        bar: Bar,
        timestamp: datetime,
        strategy: Any,
    ) -> None:
        """Open a new position.

        Two paths:
          * config.cost_model_config is None -> legacy flat slippage_pct
          * config.cost_model_config is set  -> realistic decomposition
            (Corwin-Schultz spread, sqrt impact, IBKR tiered commission).

        See PLAN.md §2.3 for the cost model. The legacy path is preserved
        for the existing 84-test surface; new backtests should set
        cost_model_config=CostModelConfig() to opt into realistic fills.
        """
        action = signal.action
        direction = 1 if action == 'buy' else -1
        side = "BUY" if direction > 0 else "SELL"

        # Paper price = current bar's close (legacy), or next-bar open if
        # NEXT_BAR_OPEN policy fired the deferred path. Either way, by
        # the time _open_position is called, `bar` is the bar to fill on.
        paper_price = bar.close if self._execution_policy is ExecutionPolicy.SAME_BAR_CLOSE \
            else bar.open
        position_value = self.equity * self.config.position_size_pct
        quantity = int(position_value / paper_price) if paper_price > 0 else 0
        if quantity <= 0:
            return

        # Decide cost path.
        cmc: Optional[CostModelConfig] = self.config.cost_model_config
        if cmc is not None:
            spread_frac = self._estimate_spread_fraction(symbol)
            adv = self._estimate_adv(symbol)
            vol_bps = self._estimate_daily_vol_bps(symbol)
            cost = compute_fill_cost(
                side=side,
                quantity=quantity,
                fill_price=paper_price,
                spread_fraction=spread_frac,
                adv=adv,
                daily_vol_bps=vol_bps,
                config=cmc,
            )
            fill_price = adjust_fill_price(side, paper_price, cost, quantity)
            commission = cost.commission
        else:
            # Legacy path: flat slippage + simple commission.
            slippage = paper_price * self.config.slippage_pct * direction
            fill_price = paper_price + slippage
            commission = max(
                self.config.min_commission,
                quantity * self.config.commission_per_share,
            )

        # Cash check: do we have enough to open?
        cost_total = fill_price * quantity + commission
        if cost_total > self.cash and direction > 0:
            quantity = int((self.cash - commission) / fill_price) if fill_price > 0 else 0
            if quantity <= 0:
                return
            # Recompute commission at the reduced size (legacy path only;
            # realistic path scales linearly so it's already proportional).
            if cmc is None:
                commission = max(
                    self.config.min_commission,
                    quantity * self.config.commission_per_share,
                )

        # Deduct cost.
        self.cash -= fill_price * quantity * direction + commission

        # Strategy state for context.
        state = strategy.asset_states.get(symbol) if hasattr(strategy, 'asset_states') else None

        position = OpenPosition(
            symbol=symbol,
            direction=direction,
            entry_price=fill_price,
            entry_time=timestamp,
            entry_bar_index=self.current_bar_index,
            quantity=quantity,
            stop_loss=getattr(signal, 'stop_loss', None),
            take_profit=getattr(signal, 'take_profit', None),
            signal_confidence=getattr(signal, 'confidence', 0),
            signal_reason=getattr(signal, 'reason', ''),
            edge_votes={k: v.name if hasattr(v, 'name') else str(v)
                        for k, v in getattr(signal, 'edge_votes', {}).items()},
            market_regime=getattr(signal, 'market_regime', 'UNKNOWN').name
                          if hasattr(getattr(signal, 'market_regime', None), 'name') else 'UNKNOWN',
            atr=state.atr if state else 0,
            vwap=state.vwap if state else 0,
            best_price=fill_price,
            worst_price=fill_price,
        )

        self.positions[symbol] = position

    def _update_position(self, symbol: str, bar: Bar, timestamp: datetime) -> None:
        """Update position with new bar, check stops/targets."""
        pos = self.positions[symbol]

        # Track best/worst prices
        if pos.direction > 0:
            pos.best_price = max(pos.best_price, bar.high)
            pos.worst_price = min(pos.worst_price, bar.low)
        else:
            pos.best_price = min(pos.best_price, bar.low)
            pos.worst_price = max(pos.worst_price, bar.high)

        # Check stop loss
        if pos.stop_loss:
            if pos.direction > 0 and bar.low <= pos.stop_loss:
                self._close_position(symbol, pos.stop_loss, timestamp, "Stop loss hit")
                return
            elif pos.direction < 0 and bar.high >= pos.stop_loss:
                self._close_position(symbol, pos.stop_loss, timestamp, "Stop loss hit")
                return

        # Check take profit
        if pos.take_profit:
            if pos.direction > 0 and bar.high >= pos.take_profit:
                self._close_position(symbol, pos.take_profit, timestamp, "Take profit hit")
                return
            elif pos.direction < 0 and bar.low <= pos.take_profit:
                self._close_position(symbol, pos.take_profit, timestamp, "Take profit hit")
                return

        # Check trailing stop activation and update
        if pos.stop_loss and not pos.trailing_active:
            # Activate trailing at 2x initial risk
            initial_risk = abs(pos.entry_price - pos.stop_loss)
            current_profit = (bar.close - pos.entry_price) * pos.direction
            if current_profit >= initial_risk * 2:
                pos.trailing_active = True
                if pos.direction > 0:
                    pos.trailing_stop = pos.entry_price + initial_risk
                else:
                    pos.trailing_stop = pos.entry_price - initial_risk

        if pos.trailing_active and pos.trailing_stop:
            if pos.direction > 0:
                new_stop = bar.close - abs(pos.entry_price - pos.stop_loss) * 1.5
                if new_stop > pos.trailing_stop:
                    pos.trailing_stop = new_stop
                if bar.low <= pos.trailing_stop:
                    self._close_position(symbol, pos.trailing_stop, timestamp, "Trailing stop hit")
                    return
            else:
                new_stop = bar.close + abs(pos.entry_price - pos.stop_loss) * 1.5
                if new_stop < pos.trailing_stop:
                    pos.trailing_stop = new_stop
                if bar.high >= pos.trailing_stop:
                    self._close_position(symbol, pos.trailing_stop, timestamp, "Trailing stop hit")
                    return

        # End of day close
        if timestamp.time() >= time(15, 55):
            self._close_position(symbol, bar.close, timestamp, "End of day")

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        timestamp: datetime,
        reason: str,
    ) -> None:
        """Close a position and record the trade."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        # On exit, you sell what you're long (BUY -> SELL) and buy what you're
        # short (SELL -> BUY). For cost purposes, side is the *closing* side.
        side = "SELL" if pos.direction > 0 else "BUY"

        cmc: Optional[CostModelConfig] = self.config.cost_model_config
        if cmc is not None:
            spread_frac = self._estimate_spread_fraction(symbol)
            adv = self._estimate_adv(symbol)
            vol_bps = self._estimate_daily_vol_bps(symbol)
            cost = compute_fill_cost(
                side=side,
                quantity=pos.quantity,
                fill_price=exit_price,
                spread_fraction=spread_frac,
                adv=adv,
                daily_vol_bps=vol_bps,
                config=cmc,
            )
            fill_price = adjust_fill_price(side, exit_price, cost, pos.quantity)
            commission = cost.commission
            total_slippage = cost.spread_cost + cost.impact_cost
        else:
            # Legacy path.
            slippage = exit_price * self.config.slippage_pct * (-pos.direction)
            fill_price = exit_price + slippage
            commission = max(
                self.config.min_commission,
                pos.quantity * self.config.commission_per_share,
            )
            total_slippage = abs(slippage) * pos.quantity

        # Calculate P&L
        gross_pnl = (fill_price - pos.entry_price) * pos.direction * pos.quantity
        net_pnl = gross_pnl - commission

        # Calculate MFE/MAE
        if pos.direction > 0:
            mfe = (pos.best_price - pos.entry_price) * pos.quantity
            mae = (pos.entry_price - pos.worst_price) * pos.quantity
        else:
            mfe = (pos.entry_price - pos.best_price) * pos.quantity
            mae = (pos.worst_price - pos.entry_price) * pos.quantity

        # Update cash
        self.cash += fill_price * pos.quantity * pos.direction - commission

        # Create trade record
        trade = BacktestTrade(
            id=f"T-{len(self.trades)+1:05d}",
            symbol=symbol,
            direction="LONG" if pos.direction > 0 else "SHORT",
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            entry_bar_index=pos.entry_bar_index,
            quantity=pos.quantity,
            exit_time=timestamp,
            exit_price=fill_price,
            exit_bar_index=self.current_bar_index,
            exit_reason=reason,
            initial_stop=pos.stop_loss,
            initial_target=pos.take_profit,
            final_stop=pos.trailing_stop or pos.stop_loss,
            gross_pnl=gross_pnl,
            commission=commission,
            slippage=total_slippage,
            net_pnl=net_pnl,
            pnl_percent=(net_pnl / (pos.entry_price * pos.quantity)) * 100,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae,
            mfe_price=pos.best_price,
            mae_price=pos.worst_price,
            bars_held=self.current_bar_index - pos.entry_bar_index,
            duration_minutes=int((timestamp - pos.entry_time).total_seconds() / 60),
            signal_confidence=pos.signal_confidence,
            signal_reason=pos.signal_reason,
            edge_votes=pos.edge_votes,
            market_regime=pos.market_regime,
            entry_atr=pos.atr,
            entry_vwap=pos.vwap,
        )

        self.trades.append(trade)

        # Update daily result
        if self.current_date in self.daily_results:
            daily = self.daily_results[self.current_date]
            daily.trades_taken += 1
            daily.gross_pnl += gross_pnl
            daily.commissions += commission
            daily.net_pnl += net_pnl
            if net_pnl > 0:
                daily.trades_won += 1
            elif net_pnl < 0:
                daily.trades_lost += 1

        # Update equity
        self.equity += net_pnl

        # Remove position
        del self.positions[symbol]

    def _close_all_positions(self, timestamp: datetime) -> None:
        """Close all remaining positions."""
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            self._close_position(symbol, pos.entry_price, timestamp, "Backtest end")

    # ------------------------------------------------------------------
    # Pending-order queue (PLAN.md §2.4): NEXT_BAR_OPEN / NEXT_BAR_VWAP.
    # ------------------------------------------------------------------

    def _enqueue_pending(
        self,
        signal: Any,
        symbol: str,
        timestamp: datetime,
        action: str,
    ) -> None:
        """Queue an entry/exit signal for fill on the next bar."""
        # Determine direction; for closes it's the inverse of the held position.
        if action in ('buy',):
            direction = +1
            side = "BUY"
        elif action in ('short',):
            direction = -1
            side = "SELL"
        elif action in ('sell',):
            direction = -1
            side = "SELL"
        elif action in ('cover',):
            direction = +1
            side = "BUY"
        else:
            return

        # Determine quantity: for entries we resize at fill time; for closes
        # we use the current position size.
        qty = 0.0
        if action in ('buy', 'short'):
            qty = -1.0   # sentinel; engine sizes from equity at fill time
        elif symbol in self.positions:
            qty = float(self.positions[symbol].quantity)

        self._pending_orders[symbol] = PendingOrder(
            symbol=symbol,
            side=side,
            quantity=qty,
            queued_at=timestamp,
            direction=direction,
            metadata={
                "action": action,
                "signal": signal,
                "reason": getattr(signal, 'reason', ''),
            },
            stop_loss=getattr(signal, 'stop_loss', None),
            take_profit=getattr(signal, 'take_profit', None),
        )

    def _drain_pending_orders(
        self,
        bar_lookup: Dict[str, Dict[datetime, Bar]],
        timestamp: datetime,
    ) -> None:
        """Fill any pending orders queued on the previous bar.

        Fill price is determined by self._execution_policy and this
        bar's OHLC. After draining the queue is reset for the bar that's
        about to generate new signals.
        """
        if not self._pending_orders:
            return
        drained: List[str] = []
        for symbol, pending in self._pending_orders.items():
            if symbol not in bar_lookup or timestamp not in bar_lookup[symbol]:
                # No bar for this symbol on this timestamp — keep pending.
                continue
            bar = bar_lookup[symbol][timestamp]
            paper_price = fill_price_for_policy(
                self._execution_policy,
                bar_open=bar.open,
                bar_close=bar.close,
                bar_vwap=bar.vwap or None,
            )
            action = pending.metadata.get("action")
            signal = pending.metadata.get("signal")
            if action in ('buy', 'short'):
                # Fabricate a bar at the fill price for the open.
                synthetic = Bar(
                    timestamp=timestamp,
                    open=paper_price, high=paper_price, low=paper_price,
                    close=paper_price,
                    volume=bar.volume,
                    vwap=bar.vwap,
                )
                # Re-route via _open_position with SAME_BAR_CLOSE semantics
                # locally so paper_price is used. Switch policy for the
                # call, then restore.
                prev_policy = self._execution_policy
                self._execution_policy = ExecutionPolicy.SAME_BAR_CLOSE
                try:
                    # Strategy ref isn't available here; pass a stub that
                    # exposes asset_states fallback via getattr.
                    self._open_position(signal, symbol, synthetic, timestamp,
                                         _BlankStrategyRef())
                finally:
                    self._execution_policy = prev_policy
            elif action in ('sell', 'cover'):
                if symbol in self.positions:
                    self._close_position(
                        symbol, paper_price, timestamp,
                        pending.metadata.get("reason") or "Signal exit",
                    )
            drained.append(symbol)
        for sym in drained:
            self._pending_orders.pop(sym, None)

    # ------------------------------------------------------------------
    # Borrow accrual.
    # ------------------------------------------------------------------

    def _accrue_borrow_costs(
        self,
        bar_lookup: Dict[str, Dict[datetime, Bar]],
        timestamp: datetime,
    ) -> None:
        """Charge daily ACT/360 borrow on any open short positions.

        We call this once per bar for simplicity; for an intraday-bar
        backtest the per-bar charge is divided across bars-per-day so the
        annualised total ties out. For a daily-bar backtest, one bar = one
        day's accrual.
        """
        rate_bps = self.config.default_borrow_rate_bps
        if rate_bps is None or rate_bps <= 0:
            return
        # Approximate bars-per-day from bar_size string. Daily = 1.
        bars_per_day = 1.0
        bs = (self.config.bar_size or "").lower()
        if "min" in bs:
            try:
                m = float(bs.split()[0])
                bars_per_day = max(1.0, 390.0 / m)
            except (ValueError, IndexError):
                bars_per_day = 78.0   # 5-min default
        elif "hour" in bs:
            bars_per_day = 7.0
        for symbol, pos in self.positions.items():
            if pos.direction >= 0:
                continue
            if symbol not in bar_lookup or timestamp not in bar_lookup[symbol]:
                continue
            bar = bar_lookup[symbol][timestamp]
            notional = abs(pos.quantity) * bar.close
            charge = borrow_charge(
                notional=notional, rate_bps=rate_bps, days=1,
            ) / bars_per_day
            self.cash -= charge
            self.equity -= charge

    def _start_day(self, d: date) -> None:
        """Initialize tracking for a new day."""
        self.daily_results[d] = DailyResult(
            date=d,
            starting_equity=self.equity,
            ending_equity=self.equity,
            high_water_mark=self.equity,
            low_water_mark=self.equity,
        )

    def _finalize_day(self, d: date) -> None:
        """Finalize daily statistics."""
        if d not in self.daily_results:
            return

        daily = self.daily_results[d]
        daily.ending_equity = self.equity

        if daily.starting_equity > 0:
            daily.return_pct = ((daily.ending_equity - daily.starting_equity) /
                                daily.starting_equity) * 100

    def _record_equity_point(
        self,
        timestamp: datetime,
        bar_lookup: Dict[str, Dict[datetime, Bar]],
    ) -> None:
        """Record current equity point."""
        # Calculate position value
        position_value = 0.0
        for symbol, pos in self.positions.items():
            if symbol in bar_lookup and timestamp in bar_lookup[symbol]:
                bar = bar_lookup[symbol][timestamp]
                pnl = (bar.close - pos.entry_price) * pos.direction * pos.quantity
                position_value += pos.entry_price * pos.quantity + pnl

        current_equity = self.cash + position_value
        self.equity = current_equity

        # Update high water mark and drawdown
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity
            # End current drawdown period if any
            if self.drawdown_start_date and self.drawdown_periods:
                self.drawdown_periods[-1].end_date = timestamp.date()
                self.drawdown_periods[-1].recovery_days = (
                    timestamp.date() - self.drawdown_periods[-1].start_date
                ).days
            self.drawdown_start_date = None

        self.current_drawdown = self.high_water_mark - current_equity
        self.current_drawdown_pct = (self.current_drawdown / self.high_water_mark * 100
                                     if self.high_water_mark > 0 else 0)

        # Start new drawdown period if needed
        if self.current_drawdown > 0 and self.drawdown_start_date is None:
            self.drawdown_start_date = timestamp.date()
            self.drawdown_periods.append(DrawdownPeriod(
                start_date=timestamp.date(),
                trough_date=timestamp.date(),
                end_date=None,
                peak_equity=self.high_water_mark,
                trough_equity=current_equity,
                current_equity=current_equity,
                drawdown_amount=self.current_drawdown,
                drawdown_percent=self.current_drawdown_pct,
                duration_days=0,
                recovery_days=None,
            ))
        elif self.drawdown_periods and self.drawdown_periods[-1].end_date is None:
            # Update current drawdown period
            dd = self.drawdown_periods[-1]
            if current_equity < dd.trough_equity:
                dd.trough_equity = current_equity
                dd.trough_date = timestamp.date()
            dd.current_equity = current_equity
            dd.drawdown_amount = self.current_drawdown
            dd.drawdown_percent = self.current_drawdown_pct
            dd.duration_days = (timestamp.date() - dd.start_date).days

        # Record equity point (sample every few bars to avoid huge lists)
        if len(self.equity_history) == 0 or \
           (timestamp - self.equity_history[-1].timestamp).seconds >= 300:  # 5 min
            self.equity_history.append(EquityPoint(
                timestamp=timestamp,
                equity=current_equity,
                cash=self.cash,
                position_value=position_value,
                drawdown=self.current_drawdown,
                drawdown_pct=self.current_drawdown_pct,
                high_water_mark=self.high_water_mark,
            ))

    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            # Return empty metrics
            return BacktestMetrics(
                start_date=self.config.start_date or date.today(),
                end_date=self.config.end_date or date.today(),
                trading_days=len(self.daily_results),
                initial_capital=self.config.initial_capital,
                final_capital=self.equity,
            )

        # Basic info
        daily_list = sorted(self.daily_results.values(), key=lambda x: x.date)
        start_date = daily_list[0].date if daily_list else date.today()
        end_date = daily_list[-1].date if daily_list else date.today()
        trading_days = len(daily_list)

        # Returns
        total_return = self.equity - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital) * 100

        # Annualized
        years = trading_days / 252 if trading_days > 0 else 1
        if years > 0 and self.equity > 0 and self.config.initial_capital > 0:
            cagr = ((self.equity / self.config.initial_capital) ** (1 / years) - 1) * 100
        else:
            cagr = 0

        # Trade statistics
        winners = [t for t in self.trades if t.net_pnl > 0]
        losers = [t for t in self.trades if t.net_pnl < 0]
        break_evens = [t for t in self.trades if t.net_pnl == 0]

        win_rate = (len(winners) / len(self.trades) * 100) if self.trades else 0
        loss_rate = (len(losers) / len(self.trades) * 100) if self.trades else 0

        gross_profit = sum(t.net_pnl for t in winners)
        gross_loss = sum(t.net_pnl for t in losers)
        net_profit = gross_profit + gross_loss

        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else float('inf')

        avg_win = (gross_profit / len(winners)) if winners else 0
        avg_loss = (gross_loss / len(losers)) if losers else 0
        avg_trade = (net_profit / len(self.trades)) if self.trades else 0

        largest_win = max((t.net_pnl for t in self.trades), default=0)
        largest_loss = min((t.net_pnl for t in self.trades), default=0)

        avg_win_loss_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else float('inf')

        # Expectancy
        if self.trades:
            win_prob = len(winners) / len(self.trades)
            loss_prob = len(losers) / len(self.trades)
            expectancy = (win_prob * avg_win) + (loss_prob * avg_loss)
        else:
            expectancy = 0

        # Daily stats
        daily_pnls = [d.net_pnl for d in daily_list]
        avg_daily_pnl = statistics.mean(daily_pnls) if daily_pnls else 0
        std_daily_pnl = statistics.stdev(daily_pnls) if len(daily_pnls) > 1 else 0

        profitable_days = sum(1 for d in daily_list if d.net_pnl > 0)
        losing_days = sum(1 for d in daily_list if d.net_pnl < 0)

        # Sharpe (annualized)
        if std_daily_pnl > 0:
            sharpe = (avg_daily_pnl / std_daily_pnl) * math.sqrt(252)
        else:
            sharpe = 0

        # Sortino (only downside deviation)
        negative_daily = [d.net_pnl for d in daily_list if d.net_pnl < 0]
        if negative_daily and len(negative_daily) > 1:
            downside_std = statistics.stdev(negative_daily)
            sortino = (avg_daily_pnl / downside_std) * math.sqrt(252) if downside_std > 0 else 0
        else:
            sortino = 0

        # Max drawdown
        max_dd = max((dd.drawdown_amount for dd in self.drawdown_periods), default=0)
        max_dd_pct = max((dd.drawdown_percent for dd in self.drawdown_periods), default=0)
        max_dd_duration = max((dd.duration_days for dd in self.drawdown_periods), default=0)

        # Calmar
        calmar = (cagr / max_dd_pct) if max_dd_pct > 0 else 0

        # Streaks
        streaks_win = []
        streaks_loss = []
        current_streak = 0
        last_result = None

        for trade in self.trades:
            is_win = trade.net_pnl > 0
            if last_result is None:
                current_streak = 1
                last_result = is_win
            elif is_win == last_result:
                current_streak += 1
            else:
                if last_result:
                    streaks_win.append(current_streak)
                else:
                    streaks_loss.append(current_streak)
                current_streak = 1
                last_result = is_win

        if last_result is not None:
            if last_result:
                streaks_win.append(current_streak)
            else:
                streaks_loss.append(current_streak)

        max_consec_wins = max(streaks_win, default=0)
        max_consec_losses = max(streaks_loss, default=0)
        avg_consec_wins = statistics.mean(streaks_win) if streaks_win else 0
        avg_consec_losses = statistics.mean(streaks_loss) if streaks_loss else 0

        # Time metrics
        avg_bars_winner = statistics.mean([t.bars_held for t in winners]) if winners else 0
        avg_bars_loser = statistics.mean([t.bars_held for t in losers]) if losers else 0
        avg_duration = statistics.mean([t.duration_minutes for t in self.trades]) if self.trades else 0

        # Time in market
        time_in_market = (self.bars_in_market / self.total_bars * 100) if self.total_bars > 0 else 0

        return BacktestMetrics(
            start_date=start_date,
            end_date=end_date,
            trading_days=trading_days,
            initial_capital=self.config.initial_capital,
            final_capital=self.equity,
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=cagr,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_duration=max_dd_duration,
            total_trades=len(self.trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            break_even_trades=len(break_evens),
            win_rate=win_rate,
            loss_rate=loss_rate,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_profit=net_profit,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_win_loss_ratio=avg_win_loss_ratio,
            expectancy=expectancy,
            avg_bars_in_winner=avg_bars_winner,
            avg_bars_in_loser=avg_bars_loser,
            avg_trade_duration_minutes=avg_duration,
            avg_daily_pnl=avg_daily_pnl,
            std_daily_pnl=std_daily_pnl,
            best_day=max(daily_pnls, default=0),
            worst_day=min(daily_pnls, default=0),
            profitable_days=profitable_days,
            losing_days=losing_days,
            daily_win_rate=(profitable_days / trading_days * 100) if trading_days > 0 else 0,
            max_consecutive_wins=max_consec_wins,
            max_consecutive_losses=max_consec_losses,
            avg_consecutive_wins=avg_consec_wins,
            avg_consecutive_losses=avg_consec_losses,
            time_in_market_pct=time_in_market,
        )
