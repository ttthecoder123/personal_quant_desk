"""
P&L Monitor

Real-time P&L calculation, unrealized P&L tracking, daily P&L monitoring,
attribution analysis, drawdown tracking, high water mark tracking,
and Sharpe ratio monitoring.
"""

import time
import threading
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pandas as pd
import numpy as np


@dataclass
class PnLSnapshot:
    """P&L snapshot at a point in time."""
    timestamp: datetime
    realized_pnl: float
    unrealized_pnl: float
    total_pnl: float
    daily_pnl: float
    commission: float = 0.0
    fees: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def net_pnl(self) -> float:
        """Calculate net P&L after costs."""
        return self.total_pnl - self.commission - self.fees


@dataclass
class Trade:
    """Trade information for P&L calculation."""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    commission: float = 0.0
    fees: float = 0.0
    strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def gross_value(self) -> float:
        """Calculate gross trade value."""
        return self.quantity * self.price

    @property
    def net_value(self) -> float:
        """Calculate net trade value."""
        return self.gross_value - self.commission - self.fees


@dataclass
class Attribution:
    """P&L attribution by strategy/symbol."""
    name: str
    attribution_type: str  # 'strategy' or 'symbol'
    pnl: float
    trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    timestamp: datetime


class PnLMonitor:
    """
    Comprehensive P&L monitoring.

    Features:
    - Real-time P&L calculation
    - Unrealized P&L tracking
    - Daily P&L monitoring
    - Attribution analysis (by strategy, symbol, etc.)
    - Drawdown tracking
    - High water mark tracking
    - Sharpe ratio monitoring
    - Commission and fee tracking
    - Win/loss statistics
    """

    def __init__(self, update_interval: int = 1, risk_free_rate: float = 0.02):
        """
        Initialize P&L monitor.

        Args:
            update_interval: Seconds between P&L updates
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.update_interval = update_interval
        self.risk_free_rate = risk_free_rate
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # P&L tracking
        self.pnl_history: deque = deque(maxlen=100000)
        self.daily_pnl: Dict[date, float] = {}
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_commission = 0.0
        self.total_fees = 0.0

        # Trades
        self.trades: deque = deque(maxlen=10000)
        self.open_positions: Dict[str, List[Trade]] = defaultdict(list)

        # Performance metrics
        self.high_water_mark = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.drawdown_history: deque = deque(maxlen=10000)
        self.sharpe_history: deque = deque(maxlen=1000)

        # Attribution
        self.strategy_attribution: Dict[str, Attribution] = {}
        self.symbol_attribution: Dict[str, Attribution] = {}
        self.attribution_history: deque = deque(maxlen=1000)

        # Statistics
        self.win_count = 0
        self.loss_count = 0
        self.total_wins = 0.0
        self.total_losses = 0.0

    def start(self):
        """Start P&L monitoring."""
        if self.running:
            return

        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        """Stop P&L monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def record_trade(self, trade: Trade):
        """
        Record a trade.

        Args:
            trade: Trade object
        """
        with self.lock:
            self.trades.append(trade)
            self.total_commission += trade.commission
            self.total_fees += trade.fees

            # Update open positions
            if trade.side == 'buy':
                self.open_positions[trade.symbol].append(trade)
            elif trade.side == 'sell':
                # Close positions FIFO
                self._close_positions(trade)

    def _close_positions(self, sell_trade: Trade):
        """
        Close positions and calculate realized P&L.

        Args:
            sell_trade: Sell trade
        """
        remaining_quantity = sell_trade.quantity
        realized = 0.0

        while remaining_quantity > 0 and self.open_positions[sell_trade.symbol]:
            buy_trade = self.open_positions[sell_trade.symbol][0]

            if buy_trade.quantity <= remaining_quantity:
                # Close entire position
                pnl = (sell_trade.price - buy_trade.price) * buy_trade.quantity
                realized += pnl
                remaining_quantity -= buy_trade.quantity
                self.open_positions[sell_trade.symbol].pop(0)
            else:
                # Partial close
                pnl = (sell_trade.price - buy_trade.price) * remaining_quantity
                realized += pnl
                buy_trade.quantity -= remaining_quantity
                remaining_quantity = 0

        self.realized_pnl += realized

        # Update statistics
        if realized > 0:
            self.win_count += 1
            self.total_wins += realized
        elif realized < 0:
            self.loss_count += 1
            self.total_losses += abs(realized)

    def update_unrealized_pnl(self, symbol: str, current_price: float):
        """
        Update unrealized P&L for a symbol.

        Args:
            symbol: Symbol
            current_price: Current market price
        """
        with self.lock:
            unrealized = 0.0
            for trade in self.open_positions.get(symbol, []):
                unrealized += (current_price - trade.price) * trade.quantity

            # Update total unrealized
            # Remove old unrealized for this symbol and add new
            old_unrealized = sum(
                (current_price - t.price) * t.quantity
                for t in self.open_positions.get(symbol, [])
            )
            self.unrealized_pnl = self.unrealized_pnl - old_unrealized + unrealized

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                self._calculate_pnl()
                self._calculate_drawdown()
                self._calculate_sharpe()
                self._calculate_attribution()
                self._update_daily_pnl()

                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in P&L monitor loop: {e}")

    def _calculate_pnl(self):
        """Calculate current P&L."""
        with self.lock:
            total_pnl = self.realized_pnl + self.unrealized_pnl
            today = date.today()
            daily_pnl = self.daily_pnl.get(today, 0.0)

            snapshot = PnLSnapshot(
                timestamp=datetime.now(),
                realized_pnl=self.realized_pnl,
                unrealized_pnl=self.unrealized_pnl,
                total_pnl=total_pnl,
                daily_pnl=daily_pnl,
                commission=self.total_commission,
                fees=self.total_fees
            )

            self.pnl_history.append(snapshot)

    def _calculate_drawdown(self):
        """Calculate current drawdown and high water mark."""
        with self.lock:
            if not self.pnl_history:
                return

            current_pnl = self.pnl_history[-1].total_pnl

            # Update high water mark
            if current_pnl > self.high_water_mark:
                self.high_water_mark = current_pnl

            # Calculate drawdown
            if self.high_water_mark > 0:
                self.current_drawdown = (self.high_water_mark - current_pnl) / self.high_water_mark
            else:
                self.current_drawdown = 0.0

            # Update max drawdown
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown

            self.drawdown_history.append({
                'timestamp': datetime.now(),
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown,
                'high_water_mark': self.high_water_mark,
                'current_pnl': current_pnl
            })

    def _calculate_sharpe(self):
        """Calculate Sharpe ratio."""
        with self.lock:
            if len(self.pnl_history) < 30:
                return

            # Get recent returns
            recent_pnl = [snapshot.total_pnl for snapshot in list(self.pnl_history)[-252:]]
            returns = np.diff(recent_pnl) / np.array(recent_pnl[:-1]) if recent_pnl[0] != 0 else np.array([])

            if len(returns) > 0:
                mean_return = np.mean(returns)
                std_return = np.std(returns)

                if std_return > 0:
                    # Annualized Sharpe ratio
                    sharpe = (mean_return * 252 - self.risk_free_rate) / (std_return * np.sqrt(252))
                else:
                    sharpe = 0.0

                self.sharpe_history.append({
                    'timestamp': datetime.now(),
                    'sharpe_ratio': sharpe,
                    'mean_return': mean_return,
                    'std_return': std_return
                })

    def _calculate_attribution(self):
        """Calculate P&L attribution by strategy and symbol."""
        with self.lock:
            # Strategy attribution
            strategy_pnl = defaultdict(float)
            strategy_trades = defaultdict(int)
            strategy_wins = defaultdict(float)
            strategy_losses = defaultdict(float)
            strategy_win_count = defaultdict(int)
            strategy_loss_count = defaultdict(int)

            for trade in self.trades:
                if trade.strategy:
                    strategy_trades[trade.strategy] += 1
                    # This is simplified - in reality need to track position P&L
                    # For now, track trade value
                    value = trade.net_value
                    strategy_pnl[trade.strategy] += value

            # Symbol attribution
            symbol_pnl = defaultdict(float)
            symbol_trades = defaultdict(int)

            for trade in self.trades:
                symbol_trades[trade.symbol] += 1
                symbol_pnl[trade.symbol] += trade.net_value

            # Store attribution
            for strategy, pnl in strategy_pnl.items():
                total_trades = strategy_trades[strategy]
                wins = strategy_wins[strategy]
                losses = strategy_losses[strategy]
                win_count = strategy_win_count[strategy]
                loss_count = strategy_loss_count[strategy]

                win_rate = win_count / total_trades if total_trades > 0 else 0
                avg_win = wins / win_count if win_count > 0 else 0
                avg_loss = losses / loss_count if loss_count > 0 else 0
                profit_factor = wins / losses if losses > 0 else 0

                self.strategy_attribution[strategy] = Attribution(
                    name=strategy,
                    attribution_type='strategy',
                    pnl=pnl,
                    trades=total_trades,
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    profit_factor=profit_factor,
                    timestamp=datetime.now()
                )

    def _update_daily_pnl(self):
        """Update daily P&L tracking."""
        with self.lock:
            if not self.pnl_history:
                return

            today = date.today()
            current_pnl = self.pnl_history[-1].total_pnl

            # Find P&L at start of day
            start_of_day = datetime.combine(today, datetime.min.time())
            day_start_pnl = 0.0

            for snapshot in reversed(list(self.pnl_history)):
                if snapshot.timestamp < start_of_day:
                    day_start_pnl = snapshot.total_pnl
                    break

            daily_pnl = current_pnl - day_start_pnl
            self.daily_pnl[today] = daily_pnl

    def get_current_pnl(self) -> Dict[str, float]:
        """
        Get current P&L metrics.

        Returns:
            Dictionary of P&L metrics
        """
        with self.lock:
            if not self.pnl_history:
                return {
                    'realized_pnl': 0.0,
                    'unrealized_pnl': 0.0,
                    'total_pnl': 0.0,
                    'net_pnl': 0.0,
                    'daily_pnl': 0.0
                }

            snapshot = self.pnl_history[-1]
            return {
                'realized_pnl': snapshot.realized_pnl,
                'unrealized_pnl': snapshot.unrealized_pnl,
                'total_pnl': snapshot.total_pnl,
                'net_pnl': snapshot.net_pnl,
                'daily_pnl': snapshot.daily_pnl,
                'commission': snapshot.commission,
                'fees': snapshot.fees
            }

    def get_drawdown_metrics(self) -> Dict[str, float]:
        """
        Get drawdown metrics.

        Returns:
            Dictionary of drawdown metrics
        """
        with self.lock:
            return {
                'current_drawdown': self.current_drawdown,
                'max_drawdown': self.max_drawdown,
                'high_water_mark': self.high_water_mark
            }

    def get_sharpe_ratio(self) -> float:
        """
        Get current Sharpe ratio.

        Returns:
            Sharpe ratio
        """
        with self.lock:
            if not self.sharpe_history:
                return 0.0
            return self.sharpe_history[-1]['sharpe_ratio']

    def get_win_loss_stats(self) -> Dict[str, Any]:
        """
        Get win/loss statistics.

        Returns:
            Dictionary of statistics
        """
        with self.lock:
            total_trades = self.win_count + self.loss_count
            win_rate = self.win_count / total_trades if total_trades > 0 else 0
            avg_win = self.total_wins / self.win_count if self.win_count > 0 else 0
            avg_loss = self.total_losses / self.loss_count if self.loss_count > 0 else 0
            profit_factor = self.total_wins / self.total_losses if self.total_losses > 0 else 0

            return {
                'total_trades': total_trades,
                'winning_trades': self.win_count,
                'losing_trades': self.loss_count,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_wins': self.total_wins,
                'total_losses': self.total_losses
            }

    def get_strategy_attribution(self) -> Dict[str, Attribution]:
        """
        Get P&L attribution by strategy.

        Returns:
            Dictionary of strategy attributions
        """
        with self.lock:
            return self.strategy_attribution.copy()

    def get_symbol_attribution(self) -> Dict[str, Attribution]:
        """
        Get P&L attribution by symbol.

        Returns:
            Dictionary of symbol attributions
        """
        with self.lock:
            return self.symbol_attribution.copy()

    def get_daily_pnl_history(self, days: int = 30) -> Dict[date, float]:
        """
        Get daily P&L history.

        Args:
            days: Number of days to retrieve

        Returns:
            Dictionary of daily P&L
        """
        with self.lock:
            cutoff_date = date.today() - timedelta(days=days)
            return {
                d: pnl for d, pnl in self.daily_pnl.items()
                if d >= cutoff_date
            }

    def get_pnl_history(self, hours: int = 24) -> List[PnLSnapshot]:
        """
        Get P&L history.

        Args:
            hours: Number of hours to retrieve

        Returns:
            List of P&L snapshots
        """
        with self.lock:
            cutoff = datetime.now() - timedelta(hours=hours)
            return [
                snapshot for snapshot in self.pnl_history
                if snapshot.timestamp >= cutoff
            ]

    def get_summary(self) -> Dict[str, Any]:
        """
        Get P&L summary.

        Returns:
            Summary dictionary
        """
        with self.lock:
            current_pnl = self.get_current_pnl()
            drawdown = self.get_drawdown_metrics()
            stats = self.get_win_loss_stats()
            sharpe = self.get_sharpe_ratio()

            return {
                'realized_pnl': current_pnl['realized_pnl'],
                'unrealized_pnl': current_pnl['unrealized_pnl'],
                'total_pnl': current_pnl['total_pnl'],
                'net_pnl': current_pnl['net_pnl'],
                'daily_pnl': current_pnl['daily_pnl'],
                'current_drawdown': drawdown['current_drawdown'],
                'max_drawdown': drawdown['max_drawdown'],
                'high_water_mark': drawdown['high_water_mark'],
                'sharpe_ratio': sharpe,
                'win_rate': stats['win_rate'],
                'profit_factor': stats['profit_factor'],
                'total_trades': stats['total_trades'],
                'total_commission': self.total_commission,
                'total_fees': self.total_fees,
                'timestamp': datetime.now().isoformat()
            }

    def get_performance_dataframe(self) -> pd.DataFrame:
        """
        Get performance metrics as DataFrame.

        Returns:
            DataFrame of performance metrics
        """
        with self.lock:
            if not self.pnl_history:
                return pd.DataFrame()

            data = []
            for snapshot in self.pnl_history:
                data.append({
                    'timestamp': snapshot.timestamp,
                    'realized_pnl': snapshot.realized_pnl,
                    'unrealized_pnl': snapshot.unrealized_pnl,
                    'total_pnl': snapshot.total_pnl,
                    'net_pnl': snapshot.net_pnl,
                    'daily_pnl': snapshot.daily_pnl,
                    'commission': snapshot.commission,
                    'fees': snapshot.fees
                })

            return pd.DataFrame(data)
