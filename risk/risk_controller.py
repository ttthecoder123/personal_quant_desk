"""
Risk Controller - Main Interface

Unified interface for all risk management operations:
- Coordinates all risk subsystems
- Provides high-level risk management API
- Integrates with strategy engine
- Manages risk state and reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging
import yaml

from .core.risk_engine import RiskEngine, RiskLimits, RiskState, PreTradeCheckResult
from .core.risk_metrics import RiskMetrics
from .core.var_models import VaRModels
from .core.stress_testing import StressTester

from .position_sizing.volatility_targeting import VolatilityTargeting
from .position_sizing.kelly_optimizer import KellyOptimizer
from .position_sizing.risk_budgeting import RiskBudgeting
from .position_sizing.dynamic_sizing import DynamicSizing

from .portfolio_risk.correlation_risk import CorrelationRisk
from .portfolio_risk.concentration_risk import ConcentrationRisk
from .portfolio_risk.liquidity_risk import LiquidityRisk
from .portfolio_risk.tail_risk import TailRisk

from .drawdown_control.drawdown_manager import DrawdownManager
from .drawdown_control.stop_loss_system import StopLossSystem
from .drawdown_control.circuit_breakers import CircuitBreakers
from .drawdown_control.recovery_rules import RecoveryRules

from .market_risk.regime_detection import RegimeDetector
from .market_risk.volatility_forecasting import VolatilityForecaster
from .market_risk.correlation_dynamics import CorrelationDynamics
from .market_risk.factor_risk import FactorRisk

from .operational_risk.execution_risk import ExecutionRisk
from .operational_risk.model_risk import ModelRisk
from .operational_risk.data_risk import DataRisk
from .operational_risk.system_risk import SystemRisk

from .reporting.risk_dashboard import RiskDashboard
from .reporting.risk_reports import RiskReporter
from .reporting.attribution import RiskAttribution

from .alerts.alert_manager import AlertManager
from .alerts.thresholds import ThresholdManager


class RiskController:
    """
    Main risk management controller

    Provides unified interface to all risk management subsystems
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        initial_capital: float = 100000.0
    ):
        """
        Initialize risk controller

        Args:
            config_path: Path to risk configuration file
            initial_capital: Initial portfolio capital
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        self.config = self._load_config(config_path)

        # Initialize risk limits
        risk_limits = self._create_risk_limits()

        # Core risk subsystems
        self.risk_engine = RiskEngine(risk_limits, initial_capital)
        self.risk_metrics = RiskMetrics()
        self.var_models = VaRModels()
        self.stress_tester = StressTester()

        # Position sizing
        self.vol_targeting = VolatilityTargeting(
            target_volatility=risk_limits.target_volatility
        )
        self.kelly_optimizer = KellyOptimizer()
        self.risk_budgeting = RiskBudgeting()
        self.dynamic_sizing = DynamicSizing()

        # Portfolio risk
        self.correlation_risk = CorrelationRisk()
        self.concentration_risk = ConcentrationRisk()
        self.liquidity_risk = LiquidityRisk()
        self.tail_risk = TailRisk()

        # Drawdown control
        self.drawdown_manager = DrawdownManager(
            initial_capital=initial_capital
        )
        self.stop_loss_system = StopLossSystem()
        self.circuit_breakers = CircuitBreakers()
        self.recovery_rules = RecoveryRules(
            initial_capital=initial_capital
        )

        # Market risk
        self.regime_detector = RegimeDetector()
        self.vol_forecaster = VolatilityForecaster()
        self.correlation_dynamics = CorrelationDynamics()
        self.factor_risk = FactorRisk()

        # Operational risk
        self.execution_risk = ExecutionRisk()
        self.model_risk = ModelRisk()
        self.data_risk = DataRisk()
        self.system_risk = SystemRisk()

        # Reporting and alerts
        self.dashboard = RiskDashboard()
        self.reporter = RiskReporter()
        self.attribution = RiskAttribution()
        self.alert_manager = AlertManager()
        self.threshold_manager = ThresholdManager()

        # State
        self.current_positions: Dict[str, float] = {}
        self.current_prices: Dict[str, float] = {}
        self.portfolio_value: float = initial_capital

        self.logger.info("Risk Controller initialized")

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load risk configuration from YAML"""
        if config_path is None:
            # Default config path
            config_dir = Path(__file__).parent / "config"
            config_path = config_dir / "risk_limits.yaml"

        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}. Using defaults.")
            return {}

    def _create_risk_limits(self) -> RiskLimits:
        """Create RiskLimits object from configuration"""
        portfolio = self.config.get('portfolio', {})
        position = self.config.get('position', {})
        correlation = self.config.get('correlation', {})
        execution = self.config.get('execution', {})

        return RiskLimits(
            max_var_95=portfolio.get('max_var_95', 0.02),
            max_drawdown=portfolio.get('max_drawdown', 0.20),
            target_volatility=portfolio.get('target_volatility', 0.20),
            max_leverage=portfolio.get('max_leverage', 2.0),
            max_position_size=position.get('max_position_size', 0.02),
            max_concentration=position.get('max_concentration', 0.20),
            max_correlation=correlation.get('max_correlation', 0.85),
            min_effective_bets=correlation.get('min_effective_bets', 3),
            max_slippage_bps=execution.get('max_slippage_bps', 10),
            max_daily_turnover=execution.get('max_daily_turnover', 0.50)
        )

    # ====================
    # Main Risk Operations
    # ====================

    def check_pre_trade(
        self,
        symbol: str,
        order_size: float,
        order_price: float
    ) -> PreTradeCheckResult:
        """
        Pre-trade risk check

        Args:
            symbol: Symbol to trade
            order_size: Order size (signed)
            order_price: Expected price

        Returns:
            PreTradeCheckResult
        """
        return self.risk_engine.check_pre_trade_risk(
            symbol=symbol,
            order_size=order_size,
            order_price=order_price,
            current_positions=self.current_positions,
            current_prices=self.current_prices,
            portfolio_value=self.portfolio_value
        )

    def update_risk_state(
        self,
        returns: pd.Series,
        equity_curve: pd.Series
    ) -> RiskState:
        """
        Update current risk state

        Args:
            returns: Return series
            equity_curve: Equity curve

        Returns:
            Updated RiskState
        """
        risk_state = self.risk_engine.update_risk_metrics(
            returns=returns,
            equity_curve=equity_curve,
            current_positions=self.current_positions,
            current_prices=self.current_prices,
            portfolio_value=self.portfolio_value
        )

        # Update drawdown manager
        self.drawdown_manager.update(
            equity=self.portfolio_value,
            timestamp=datetime.now()
        )

        # Check circuit breakers
        self.circuit_breakers.update(
            equity_curve=equity_curve,
            timestamp=datetime.now()
        )

        # Generate alerts if needed
        if len(risk_state.violations) > 0:
            for violation in risk_state.violations:
                self.alert_manager.create_alert(
                    alert_type="risk_violation",
                    message=violation,
                    severity="high",
                    data={'risk_state': risk_state.__dict__}
                )

        return risk_state

    def calculate_position_sizes(
        self,
        returns_dict: Dict[str, pd.Series],
        method: str = "volatility_targeting"
    ) -> Dict[str, float]:
        """
        Calculate optimal position sizes

        Args:
            returns_dict: Dictionary of symbol -> return series
            method: Sizing method ('volatility_targeting', 'kelly', 'risk_parity')

        Returns:
            Dictionary of symbol -> position size
        """
        if method == "volatility_targeting":
            results = self.vol_targeting.calculate_multi_instrument_positions(
                returns_dict=returns_dict,
                capital=self.portfolio_value,
                prices=self.current_prices,
                current_positions=self.current_positions
            )
            return {sym: r.position_size for sym, r in results.items()}

        elif method == "kelly":
            results = self.kelly_optimizer.multi_asset_kelly(
                returns_df=pd.DataFrame(returns_dict),
                capital=self.portfolio_value,
                prices=self.current_prices
            )
            return {sym: r.position_size for sym, r in results.items()}

        elif method == "risk_parity":
            # Use risk budgeting for risk parity allocation
            returns_df = pd.DataFrame(returns_dict)
            cov_matrix = returns_df.cov().values

            weights = self.risk_budgeting.risk_parity_allocation(
                cov_matrix=cov_matrix,
                target_vol=self.risk_engine.limits.target_volatility
            )

            # Convert weights to position sizes
            position_sizes = {}
            for i, symbol in enumerate(returns_dict.keys()):
                capital_allocated = weights[i] * self.portfolio_value
                price = self.current_prices.get(symbol, 1.0)
                position_sizes[symbol] = capital_allocated / price if price > 0 else 0.0

            return position_sizes

        else:
            raise ValueError(f"Unknown sizing method: {method}")

    def run_stress_tests(self) -> List:
        """
        Run all stress test scenarios

        Returns:
            List of stress test results
        """
        return self.stress_tester.run_all_scenarios(
            positions=self.current_positions,
            current_prices=self.current_prices,
            portfolio_value=self.portfolio_value
        )

    def calculate_var(
        self,
        returns: pd.Series,
        position_returns: Optional[pd.DataFrame] = None,
        weights: Optional[np.ndarray] = None
    ) -> List:
        """
        Calculate VaR using all methods

        Args:
            returns: Portfolio returns
            position_returns: Individual position returns
            weights: Position weights

        Returns:
            List of VaRResult objects
        """
        return self.var_models.calculate_all_var_metrics(
            returns=returns,
            position_returns=position_returns,
            weights=weights
        )

    def check_stop_losses(
        self,
        current_prices: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Check all stop losses

        Args:
            current_prices: Current market prices

        Returns:
            Dictionary of symbol -> stop_triggered
        """
        triggered = {}

        for symbol, position in self.current_positions.items():
            if abs(position) > 0:
                price = current_prices.get(symbol)
                if price is not None:
                    result = self.stop_loss_system.check_stop(symbol, price)
                    triggered[symbol] = result.triggered

        return triggered

    # ====================
    # Risk Reporting
    # ====================

    def get_risk_summary(self) -> str:
        """Get human-readable risk summary"""
        return self.risk_engine.get_risk_summary()

    def generate_risk_report(
        self,
        report_type: str = "daily",
        output_format: str = "html"
    ) -> str:
        """
        Generate comprehensive risk report

        Args:
            report_type: 'daily', 'weekly', or 'monthly'
            output_format: 'html', 'pdf', 'json'

        Returns:
            Path to generated report
        """
        risk_state = self.risk_engine.current_state

        if risk_state is None:
            return "No risk data available"

        # Collect all risk data
        report_data = {
            'risk_state': risk_state,
            'var_results': self.var_models.var_history[-10:] if self.var_models.var_history else [],
            'stress_tests': self.stress_tester.results_history[-5:] if self.stress_tester.results_history else [],
            'drawdown_metrics': self.drawdown_manager.get_current_metrics(),
            'circuit_breaker_status': self.circuit_breakers.get_status()
        }

        # Generate report
        return self.reporter.generate_report(
            report_type=report_type,
            data=report_data,
            output_format=output_format
        )

    def update_dashboard(self):
        """Update real-time risk dashboard"""
        risk_state = self.risk_engine.current_state

        if risk_state is not None:
            self.dashboard.update_metrics(risk_state.__dict__)

    # ====================
    # State Management
    # ====================

    def update_positions(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        portfolio_value: float
    ):
        """
        Update current portfolio state

        Args:
            positions: Current positions
            prices: Current prices
            portfolio_value: Current portfolio value
        """
        self.current_positions = positions
        self.current_prices = prices
        self.portfolio_value = portfolio_value

    def enable_emergency_mode(self):
        """Enable emergency risk mode"""
        self.risk_engine.enable_emergency_mode()
        self.logger.critical("Emergency mode enabled")

        # Generate critical alert
        self.alert_manager.create_alert(
            alert_type="emergency_mode",
            message="Emergency mode activated",
            severity="critical",
            data={}
        )

    def disable_emergency_mode(self):
        """Disable emergency mode"""
        self.risk_engine.disable_emergency_mode()
        self.logger.info("Emergency mode disabled")

    # ====================
    # Integration Points
    # ====================

    def get_risk_adjusted_positions(
        self,
        proposed_positions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Get risk-adjusted position recommendations

        Args:
            proposed_positions: Proposed positions from strategy engine

        Returns:
            Risk-adjusted positions
        """
        # Check each position against risk limits
        adjusted_positions = {}

        for symbol, size in proposed_positions.items():
            price = self.current_prices.get(symbol, 0)

            if price == 0:
                adjusted_positions[symbol] = 0
                continue

            # Pre-trade check
            check_result = self.check_pre_trade(symbol, size, price)

            if check_result.allowed:
                adjusted_positions[symbol] = size
            elif check_result.recommended_size is not None:
                adjusted_positions[symbol] = check_result.recommended_size
                self.logger.warning(
                    f"Reduced position for {symbol}: {size} -> {check_result.recommended_size}"
                )
            else:
                adjusted_positions[symbol] = 0
                self.logger.warning(f"Blocked position for {symbol}: {check_result.message}")

        return adjusted_positions

    def should_halt_trading(self) -> bool:
        """Check if trading should be halted"""
        # Check circuit breakers
        status = self.circuit_breakers.get_status()
        if status.status in ['halted', 'suspended']:
            return True

        # Check risk engine
        if self.risk_engine.trading_halted:
            return True

        # Check drawdown
        dd_metrics = self.drawdown_manager.get_current_metrics()
        if dd_metrics and abs(dd_metrics.current_drawdown) >= self.risk_engine.limits.max_drawdown:
            return True

        return False

    def get_risk_metrics_dict(self) -> Dict:
        """Get all risk metrics as dictionary"""
        risk_state = self.risk_engine.current_state

        if risk_state is None:
            return {}

        return {
            'timestamp': risk_state.timestamp,
            'portfolio_value': risk_state.portfolio_value,
            'leverage': risk_state.leverage,
            'volatility': risk_state.volatility,
            'var_95': risk_state.var_95,
            'drawdown': risk_state.drawdown,
            'max_drawdown': risk_state.max_drawdown,
            'risk_level': risk_state.risk_level.value,
            'violations': risk_state.violations,
            'trading_halted': self.risk_engine.trading_halted,
            'circuit_breaker_status': self.circuit_breakers.get_status().status.value
        }
