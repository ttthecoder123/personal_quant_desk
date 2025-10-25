"""
Strategy Engine - Main Orchestration System

Coordinates all strategies, portfolio construction, and execution.
Integrates with Steps 2, 3, and 4 for data, features, and ML signals.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from loguru import logger

# Base components
from strategies.base.strategy_base import StrategyBase, StrategySignal, SignalType
from strategies.base.position_manager import PositionManager, Position, PositionStatus
from strategies.base.performance_tracker import PerformanceTracker, PerformanceMetrics

# Mean reversion strategies
from strategies.mean_reversion import (
    PairsTradingStrategy,
    BollingerReversionStrategy,
    OUProcessStrategy,
    IndexArbitrageStrategy
)

# Momentum strategies
from strategies.momentum import (
    TrendFollowingStrategy,
    BreakoutMomentumStrategy,
    CrossSectionalMomentumStrategy,
    TimeSeriesMomentumStrategy
)

# Volatility strategies
from strategies.volatility import (
    VolatilityTargetingStrategy,
    VolArbitrageStrategy,
    GammaScalpingStrategy,
    DispersionTradingStrategy
)

# Hybrid strategies
from strategies.hybrid import (
    MLEnhancedStrategy,
    RegimeSwitchingStrategy,
    MultiFactorStrategy,
    EnsembleStrategy
)

# Portfolio construction
from strategies.portfolio import (
    PortfolioOptimizer,
    RiskParityAllocator,
    KellySizer,
    CorrelationManager,
    Rebalancer
)

# Execution
from strategies.execution import (
    OrderGenerator,
    ExecutionAlgorithm,
    SlippageModel,
    CostModel
)


class StrategyEngine:
    """
    Main strategy orchestration engine.

    Responsibilities:
    - Load and manage strategies
    - Fetch signals from Step 4 models
    - Execute portfolio construction
    - Generate executable orders
    - Track performance and risk
    - Manage state persistence
    """

    def __init__(
        self,
        strategy_config_path: str = "strategies/config/strategy_config.yaml",
        portfolio_config_path: str = "strategies/config/portfolio_config.yaml",
        initial_capital: float = 100000.0
    ):
        """
        Initialize strategy engine.

        Args:
            strategy_config_path: Path to strategy configuration
            portfolio_config_path: Path to portfolio configuration
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # Load configurations
        self.strategy_config = self._load_config(strategy_config_path)
        self.portfolio_config = self._load_config(portfolio_config_path)

        # Core components
        self.position_manager = PositionManager()
        self.performance_tracker = PerformanceTracker(initial_capital)
        self.portfolio_optimizer = PortfolioOptimizer(self.portfolio_config)
        self.correlation_manager = CorrelationManager(self.portfolio_config.get('correlation', {}))
        self.rebalancer = Rebalancer(self.portfolio_config.get('rebalancing', {}))
        self.order_generator = OrderGenerator()
        self.cost_model = CostModel()
        self.slippage_model = SlippageModel()

        # Strategy registry
        self.strategies: Dict[str, StrategyBase] = {}
        self.strategy_signals: Dict[str, List[StrategySignal]] = {}

        # State
        self.current_positions: Dict[str, Position] = {}
        self.target_positions: Dict[str, float] = {}
        self.last_rebalance: Optional[datetime] = None
        self.portfolio_state: Dict[str, Any] = {}

        # Initialize strategies
        self._initialize_strategies()

        logger.info(f"Strategy Engine initialized with ${initial_capital:,.2f} capital")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}

        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config

    def _initialize_strategies(self):
        """Initialize all enabled strategies from configuration."""
        logger.info("Initializing strategies...")

        # Mean Reversion
        mr_config = self.strategy_config.get('mean_reversion', {})

        if mr_config.get('pairs_trading', {}).get('enabled', False):
            self.strategies['pairs_trading'] = PairsTradingStrategy(
                'pairs_trading',
                mr_config['pairs_trading']
            )

        if mr_config.get('bollinger_reversion', {}).get('enabled', False):
            self.strategies['bollinger_reversion'] = BollingerReversionStrategy(
                'bollinger_reversion',
                mr_config['bollinger_reversion']
            )

        if mr_config.get('ornstein_uhlenbeck', {}).get('enabled', False):
            self.strategies['ornstein_uhlenbeck'] = OUProcessStrategy(
                'ornstein_uhlenbeck',
                mr_config['ornstein_uhlenbeck']
            )

        # Momentum
        mom_config = self.strategy_config.get('momentum', {})

        if mom_config.get('trend_following', {}).get('enabled', False):
            self.strategies['trend_following'] = TrendFollowingStrategy(
                'trend_following',
                mom_config['trend_following']
            )

        if mom_config.get('breakout_momentum', {}).get('enabled', False):
            self.strategies['breakout_momentum'] = BreakoutMomentumStrategy(
                'breakout_momentum',
                mom_config['breakout_momentum']
            )

        if mom_config.get('time_series_momentum', {}).get('enabled', False):
            self.strategies['time_series_momentum'] = TimeSeriesMomentumStrategy(
                'time_series_momentum',
                mom_config['time_series_momentum']
            )

        # Volatility
        vol_config = self.strategy_config.get('volatility', {})

        if vol_config.get('vol_targeting', {}).get('enabled', False):
            self.strategies['vol_targeting'] = VolatilityTargetingStrategy(
                'vol_targeting',
                vol_config['vol_targeting']
            )

        # Hybrid
        hybrid_config = self.strategy_config.get('hybrid', {})

        if hybrid_config.get('ml_enhanced', {}).get('enabled', False):
            self.strategies['ml_enhanced'] = MLEnhancedStrategy(
                'ml_enhanced',
                hybrid_config['ml_enhanced']
            )

        if hybrid_config.get('regime_switching', {}).get('enabled', False):
            self.strategies['regime_switching'] = RegimeSwitchingStrategy(
                'regime_switching',
                hybrid_config['regime_switching']
            )

        if hybrid_config.get('multi_factor', {}).get('enabled', False):
            self.strategies['multi_factor'] = MultiFactorStrategy(
                'multi_factor',
                hybrid_config['multi_factor']
            )

        logger.success(f"Initialized {len(self.strategies)} strategies: {list(self.strategies.keys())}")

    def update_signals(
        self,
        symbols: List[str],
        market_data: Dict[str, pd.DataFrame],
        features: Optional[Dict[str, pd.DataFrame]] = None,
        ml_signals: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, List[StrategySignal]]:
        """
        Generate signals from all strategies for given instruments.

        Args:
            symbols: List of instrument symbols
            market_data: Dictionary of {symbol: OHLCV DataFrame}
            features: Dictionary of {symbol: features DataFrame} from Step 3
            ml_signals: Dictionary of {symbol: ML signals DataFrame} from Step 4

        Returns:
            Dictionary of {strategy_name: [signals]}
        """
        logger.info(f"Generating signals for {len(symbols)} instruments")

        all_signals = {}

        for strategy_name, strategy in self.strategies.items():
            strategy_signals = []

            for symbol in symbols:
                try:
                    # Get data for this symbol
                    data = market_data.get(symbol)
                    if data is None or len(data) == 0:
                        continue

                    symbol_features = features.get(symbol) if features else None
                    symbol_ml_signals = ml_signals.get(symbol) if ml_signals else None

                    # Generate signal
                    signal = strategy.generate_signal(
                        symbol=symbol,
                        market_data=data,
                        features=symbol_features,
                        ml_signals=symbol_ml_signals
                    )

                    if signal is not None:
                        # Validate against risk constraints
                        is_valid, reason = strategy.manage_risk(
                            signal,
                            self._get_portfolio_state()
                        )

                        if is_valid:
                            strategy_signals.append(signal)
                        else:
                            logger.debug(f"Signal rejected for {symbol}: {reason}")

                except Exception as e:
                    logger.error(f"Error generating signal for {symbol} in {strategy_name}: {e}")

            all_signals[strategy_name] = strategy_signals
            logger.info(f"{strategy_name}: Generated {len(strategy_signals)} signals")

        self.strategy_signals = all_signals
        return all_signals

    def calculate_positions(self) -> Dict[str, float]:
        """
        Calculate target positions using portfolio optimization.

        Returns:
            Dictionary of {symbol: target_size}
        """
        logger.info("Calculating target positions")

        # Collect all signals
        all_signals = []
        for signals_list in self.strategy_signals.values():
            all_signals.extend(signals_list)

        if not all_signals:
            logger.warning("No signals to process")
            return {}

        # Group signals by symbol
        signals_by_symbol = {}
        for signal in all_signals:
            if signal.symbol not in signals_by_symbol:
                signals_by_symbol[signal.symbol] = []
            signals_by_symbol[signal.symbol].append(signal)

        # Calculate target positions
        target_positions = {}

        for symbol, signals in signals_by_symbol.items():
            # Combine signals for this symbol (simple average for now)
            # More sophisticated: use confidence-weighted averaging
            net_size = 0.0
            total_confidence = 0.0

            for signal in signals:
                direction = 1.0 if signal.signal_type == SignalType.LONG else -1.0
                net_size += signal.size * direction * signal.confidence
                total_confidence += signal.confidence

            if total_confidence > 0:
                target_positions[symbol] = net_size / len(signals)

        # Apply portfolio optimization
        try:
            optimized_positions = self.portfolio_optimizer.optimize_portfolio(
                target_positions,
                self.correlation_manager,
                self.current_capital
            )
            target_positions = optimized_positions
        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}, using raw positions")

        logger.info(f"Calculated target positions for {len(target_positions)} instruments")
        self.target_positions = target_positions
        return target_positions

    def generate_orders(self) -> List[Dict[str, Any]]:
        """
        Generate executable orders based on target positions.

        Returns:
            List of order dictionaries
        """
        logger.info("Generating orders")

        # Get current positions
        current_positions = {}
        for position in self.position_manager.open_positions.values():
            key = position.symbol
            current_positions[key] = position.size * position.side

        # Calculate required trades
        orders = self.order_generator.generate_orders(
            target_positions=self.target_positions,
            current_positions=current_positions,
            current_prices=self._get_current_prices()
        )

        # Estimate costs
        for order in orders:
            order['estimated_commission'] = self.cost_model.calculate_commission(
                quantity=order['quantity'],
                price=order['price']
            )
            order['estimated_slippage'] = self.slippage_model.estimate_slippage(
                symbol=order['symbol'],
                quantity=order['quantity'],
                side=order['side']
            )

        logger.info(f"Generated {len(orders)} orders")
        return orders

    def update_performance(self):
        """Update performance metrics."""
        # Update equity
        current_equity = self.current_capital + self.position_manager.get_unrealized_pnl()
        self.performance_tracker.update_equity(datetime.now(), current_equity)

        # Calculate metrics
        metrics = self.performance_tracker.calculate_metrics()

        logger.info(
            f"Performance: Return={metrics.total_return:.2%}, "
            f"Sharpe={metrics.sharpe_ratio:.2f}, "
            f"Drawdown={metrics.current_drawdown:.2%}"
        )

        return metrics

    def check_risk_limits(self) -> Tuple[bool, List[str]]:
        """
        Check if portfolio is within risk limits.

        Returns:
            (within_limits, violations)
        """
        violations = []
        risk_config = self.strategy_config.get('risk', {})

        # Check volatility
        metrics = self.performance_tracker.calculate_metrics()
        max_vol = risk_config.get('max_portfolio_volatility', 0.20)
        if metrics.volatility > max_vol:
            violations.append(f"Volatility {metrics.volatility:.2%} exceeds {max_vol:.2%}")

        # Check drawdown
        max_dd = risk_config.get('max_drawdown', 0.20)
        if metrics.current_drawdown > max_dd:
            violations.append(f"Drawdown {metrics.current_drawdown:.2%} exceeds {max_dd:.2%}")

        # Check position concentration
        total_exposure = self.position_manager.get_total_exposure()
        if total_exposure > 0:
            for position in self.position_manager.open_positions.values():
                position_exposure = abs(position.size * position.current_price)
                concentration = position_exposure / total_exposure
                max_concentration = risk_config.get('position_concentration_limit', 0.20)
                if concentration > max_concentration:
                    violations.append(
                        f"{position.symbol} concentration {concentration:.2%} exceeds {max_concentration:.2%}"
                    )

        if violations:
            logger.warning(f"Risk limit violations: {violations}")

        return len(violations) == 0, violations

    def rebalance_portfolio(self) -> bool:
        """
        Check if rebalancing is needed and execute.

        Returns:
            True if rebalanced
        """
        should_rebalance = self.rebalancer.should_rebalance(
            current_positions=self._get_current_positions_dict(),
            target_positions=self.target_positions,
            last_rebalance=self.last_rebalance
        )

        if should_rebalance:
            logger.info("Executing portfolio rebalance")
            self.last_rebalance = datetime.now()
            return True

        return False

    def save_state(self, filepath: str = "strategies/state/engine_state.json"):
        """
        Save engine state for persistence.

        Args:
            filepath: Path to save state
        """
        import json

        state = {
            'timestamp': datetime.now().isoformat(),
            'current_capital': self.current_capital,
            'strategies': {name: s.get_state() for name, s in self.strategies.items()},
            'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
            'target_positions': self.target_positions
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

        # Save position manager state
        self.position_manager.save_state(filepath.replace('engine_state', 'positions'))

        logger.info(f"Saved engine state to {filepath}")

    def load_state(self, filepath: str = "strategies/state/engine_state.json"):
        """
        Load engine state from persistence.

        Args:
            filepath: Path to load state
        """
        import json

        if not Path(filepath).exists():
            logger.warning(f"State file not found: {filepath}")
            return

        with open(filepath, 'r') as f:
            state = json.load(f)

        self.current_capital = state.get('current_capital', self.initial_capital)
        self.target_positions = state.get('target_positions', {})

        if state.get('last_rebalance'):
            self.last_rebalance = datetime.fromisoformat(state['last_rebalance'])

        # Load strategy states
        for name, strategy_state in state.get('strategies', {}).items():
            if name in self.strategies:
                self.strategies[name].load_state(strategy_state)

        logger.info(f"Loaded engine state from {filepath}")

    def _get_portfolio_state(self) -> Dict[str, Any]:
        """Get current portfolio state for risk management."""
        return {
            'total_value': self.current_capital + self.position_manager.get_unrealized_pnl(),
            'positions': self.position_manager.open_positions,
            'net_exposure': self.position_manager.get_net_exposure(),
            'total_exposure': self.position_manager.get_total_exposure()
        }

    def _get_current_positions_dict(self) -> Dict[str, float]:
        """Get current positions as dictionary."""
        positions = {}
        for position in self.position_manager.open_positions.values():
            positions[position.symbol] = position.size * position.side
        return positions

    def _get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all instruments (placeholder)."""
        # In production, this would fetch from data system
        prices = {}
        for position in self.position_manager.open_positions.values():
            prices[position.symbol] = position.current_price
        return prices

    def get_summary(self) -> str:
        """Get formatted summary of engine state."""
        metrics = self.performance_tracker.calculate_metrics()
        position_summary = self.position_manager.get_position_summary()

        summary = f"""
Strategy Engine Summary
=======================
Capital:            ${self.current_capital:,.2f}
Total P&L:          ${self.position_manager.get_total_pnl():,.2f}
Active Strategies:  {len(self.strategies)}
Open Positions:     {len(self.position_manager.open_positions)}

Performance Metrics:
  Sharpe Ratio:     {metrics.sharpe_ratio:.2f}
  Total Return:     {metrics.total_return:.2%}
  Max Drawdown:     {metrics.max_drawdown:.2%}
  Win Rate:         {metrics.win_rate:.2%}
  Total Trades:     {metrics.total_trades}

Risk Metrics:
  Volatility:       {metrics.volatility:.2%}
  Current DD:       {metrics.current_drawdown:.2%}
  VaR (95%):        {metrics.var_95:.2%}

Positions:
{position_summary.to_string() if not position_summary.empty else '  No open positions'}
"""
        return summary

    def __repr__(self) -> str:
        return (
            f"StrategyEngine(capital=${self.current_capital:,.2f}, "
            f"strategies={len(self.strategies)}, "
            f"positions={len(self.position_manager.open_positions)})"
        )
