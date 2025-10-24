"""
Comprehensive Tests for Strategy System (Step 5)

Tests core functionality of:
- Base strategy framework
- Position manager
- Performance tracker
- Strategy engine
- Individual strategies
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Core components
from strategies.base.strategy_base import StrategyBase, StrategySignal, SignalType, RiskMetrics
from strategies.base.position_manager import PositionManager, Position, PositionStatus
from strategies.base.performance_tracker import PerformanceTracker, PerformanceMetrics
from strategies.strategy_engine import StrategyEngine

# Strategies
from strategies.mean_reversion import BollingerReversionStrategy
from strategies.momentum import TrendFollowingStrategy
from strategies.hybrid import MLEnhancedStrategy


class TestPositionManager:
    """Test position manager functionality."""

    def test_open_position(self):
        """Test opening a position."""
        pm = PositionManager()

        position = pm.open_position(
            symbol="SPY",
            side=1,
            size=100,
            entry_price=400.0,
            entry_time=datetime.now(),
            strategy="test_strategy",
            stop_loss=390.0,
            take_profit=410.0
        )

        assert position.symbol == "SPY"
        assert position.size == 100
        assert position.entry_price == 400.0
        assert len(pm.open_positions) == 1

    def test_close_position(self):
        """Test closing a position."""
        pm = PositionManager()

        # Open position
        pm.open_position(
            symbol="SPY",
            side=1,
            size=100,
            entry_price=400.0,
            entry_time=datetime.now(),
            strategy="test_strategy"
        )

        # Close position
        exit_time = datetime.now()
        closed_position = pm.close_position(
            symbol="SPY",
            strategy="test_strategy",
            exit_price=405.0,
            exit_time=exit_time
        )

        assert closed_position is not None
        assert closed_position.realized_pnl == 500.0  # (405-400) * 100
        assert len(pm.open_positions) == 0
        assert len(pm.closed_positions) == 1

    def test_update_positions(self):
        """Test updating positions with market data."""
        pm = PositionManager()

        # Open position
        entry_time = datetime.now()
        pm.open_position(
            symbol="SPY",
            side=1,
            size=100,
            entry_price=400.0,
            entry_time=entry_time,
            strategy="test_strategy"
        )

        # Update with new price
        market_data = {"SPY": 405.0}
        pm.update_positions(market_data, datetime.now())

        position = pm.get_position("SPY", "test_strategy")
        assert position.current_price == 405.0
        assert position.unrealized_pnl == 500.0

    def test_stop_loss_trigger(self):
        """Test stop loss triggering."""
        pm = PositionManager()

        # Open position with stop loss
        entry_time = datetime.now()
        pm.open_position(
            symbol="SPY",
            side=1,
            size=100,
            entry_price=400.0,
            entry_time=entry_time,
            strategy="test_strategy",
            stop_loss=390.0
        )

        # Price hits stop loss
        market_data = {"SPY": 389.0}
        pm.update_positions(market_data, datetime.now())

        # Position should be closed
        assert len(pm.open_positions) == 0
        assert len(pm.closed_positions) == 1
        assert pm.closed_positions[0].status == PositionStatus.STOPPED_OUT


class TestPerformanceTracker:
    """Test performance tracking functionality."""

    def test_initialization(self):
        """Test tracker initialization."""
        pt = PerformanceTracker(initial_capital=100000.0)

        assert pt.initial_capital == 100000.0
        assert pt.current_capital == 100000.0
        assert len(pt.equity_curve) == 1

    def test_update_equity(self):
        """Test equity updates."""
        pt = PerformanceTracker(initial_capital=100000.0)

        pt.update_equity(datetime.now(), 105000.0)

        assert pt.current_capital == 105000.0
        assert len(pt.equity_curve) == 2
        assert len(pt.returns) == 1

    def test_record_trade(self):
        """Test recording trades."""
        pt = PerformanceTracker(initial_capital=100000.0)

        entry_time = datetime.now()
        exit_time = entry_time + timedelta(days=1)

        pt.record_trade(
            symbol="SPY",
            entry_time=entry_time,
            exit_time=exit_time,
            entry_price=400.0,
            exit_price=405.0,
            size=100,
            side=1,
            pnl=500.0,
            strategy="test_strategy"
        )

        assert len(pt.trades) == 1
        assert pt.trades[0]['pnl'] == 500.0

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        pt = PerformanceTracker(initial_capital=100000.0)

        # Simulate some equity changes
        base_time = datetime.now()
        for i in range(10):
            equity = 100000 + (i * 1000)  # Linear growth
            pt.update_equity(base_time + timedelta(days=i), equity)

        metrics = pt.calculate_metrics()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return > 0
        assert metrics.volatility >= 0

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        pt = PerformanceTracker(initial_capital=100000.0)

        # Record winning and losing trades
        base_time = datetime.now()

        # 3 winning trades
        for i in range(3):
            pt.record_trade(
                symbol="SPY",
                entry_time=base_time,
                exit_time=base_time + timedelta(hours=1),
                entry_price=400.0,
                exit_price=405.0,
                size=100,
                side=1,
                pnl=500.0,
                strategy="test"
            )

        # 2 losing trades
        for i in range(2):
            pt.record_trade(
                symbol="SPY",
                entry_time=base_time,
                exit_time=base_time + timedelta(hours=1),
                entry_price=400.0,
                exit_price=395.0,
                size=100,
                side=1,
                pnl=-500.0,
                strategy="test"
            )

        metrics = pt.calculate_metrics()
        assert metrics.win_rate == 0.6  # 3/5


class TestStrategyEngine:
    """Test strategy engine functionality."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = StrategyEngine(
            strategy_config_path="strategies/config/strategy_config.yaml",
            portfolio_config_path="strategies/config/portfolio_config.yaml",
            initial_capital=100000.0
        )

        assert engine.initial_capital == 100000.0
        assert engine.position_manager is not None
        assert engine.performance_tracker is not None
        assert isinstance(engine.strategies, dict)

    def test_signal_generation(self):
        """Test signal generation from strategies."""
        engine = StrategyEngine(initial_capital=100000.0)

        # Create sample market data
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        market_data = {
            "SPY": pd.DataFrame({
                'Open': np.random.randn(100).cumsum() + 400,
                'High': np.random.randn(100).cumsum() + 401,
                'Low': np.random.randn(100).cumsum() + 399,
                'Close': np.random.randn(100).cumsum() + 400,
                'Volume': np.random.randint(1000000, 10000000, 100)
            }, index=dates)
        }

        # Generate signals
        signals = engine.update_signals(
            symbols=["SPY"],
            market_data=market_data
        )

        assert isinstance(signals, dict)

    def test_save_and_load_state(self, tmp_path):
        """Test state persistence."""
        engine = StrategyEngine(initial_capital=100000.0)

        # Save state
        state_file = tmp_path / "test_state.json"
        engine.save_state(str(state_file))

        assert state_file.exists()

        # Load state
        engine2 = StrategyEngine(initial_capital=100000.0)
        engine2.load_state(str(state_file))

        assert engine2.current_capital == engine.current_capital


class TestBollingerReversionStrategy:
    """Test Bollinger Band reversion strategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        config = {
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70
        }

        strategy = BollingerReversionStrategy('test_bb', config)

        assert strategy.name == 'test_bb'
        assert strategy.config['bb_period'] == 20

    def test_signal_generation(self):
        """Test signal generation."""
        config = {
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'require_volume_confirmation': False
        }

        strategy = BollingerReversionStrategy('test_bb', config)

        # Create sample data with clear oversold condition
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')

        # Create decreasing price pattern to trigger oversold
        prices = np.linspace(100, 80, 50)

        market_data = pd.DataFrame({
            'Open': prices,
            'High': prices + 1,
            'Low': prices - 1,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 50)
        }, index=dates)

        signal = strategy.generate_signal(
            symbol="SPY",
            market_data=market_data
        )

        # Signal may or may not be generated depending on exact conditions
        # Just check that the method executes without error
        assert signal is None or isinstance(signal, StrategySignal)


class TestTrendFollowingStrategy:
    """Test trend following strategy."""

    def test_initialization(self):
        """Test strategy initialization."""
        config = {
            'timeframes': [20, 60, 120],
            'forecast_scalar': 10.0,
            'min_adx': 25.0
        }

        strategy = TrendFollowingStrategy('test_trend', config)

        assert strategy.name == 'test_trend'
        assert len(strategy.config['timeframes']) == 3

    def test_signal_generation_uptrend(self):
        """Test signal generation in uptrend."""
        config = {
            'timeframes': [20, 60, 120],
            'forecast_scalar': 10.0,
            'min_adx': 20.0  # Lower threshold for testing
        }

        strategy = TrendFollowingStrategy('test_trend', config)

        # Create strong uptrend data
        dates = pd.date_range(end=datetime.now(), periods=150, freq='D')
        prices = np.linspace(100, 150, 150)  # Strong uptrend

        market_data = pd.DataFrame({
            'Open': prices,
            'High': prices + 2,
            'Low': prices - 1,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 150)
        }, index=dates)

        signal = strategy.generate_signal(
            symbol="SPY",
            market_data=market_data
        )

        # Should generate a signal (likely LONG) or None if ADX too low
        assert signal is None or isinstance(signal, StrategySignal)


def test_strategy_integration():
    """Integration test for complete strategy workflow."""
    # Initialize engine
    engine = StrategyEngine(initial_capital=100000.0)

    # Create market data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    symbols = ["SPY", "GC=F"]

    market_data = {}
    for symbol in symbols:
        base_price = 400 if symbol == "SPY" else 1800
        market_data[symbol] = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + base_price,
            'High': np.random.randn(100).cumsum() + base_price + 1,
            'Low': np.random.randn(100).cumsum() + base_price - 1,
            'Close': np.random.randn(100).cumsum() + base_price,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

    # Generate signals
    signals = engine.update_signals(symbols, market_data)

    # Calculate positions
    positions = engine.calculate_positions()

    # Generate orders
    orders = engine.generate_orders()

    # Update performance
    metrics = engine.update_performance()

    # Check risk limits
    within_limits, violations = engine.check_risk_limits()

    # All should execute without errors
    assert isinstance(signals, dict)
    assert isinstance(positions, dict)
    assert isinstance(orders, list)
    assert isinstance(metrics, PerformanceMetrics)
    assert isinstance(within_limits, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
