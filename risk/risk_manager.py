"""
Risk Manager for portfolio and position risk management.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio

from utils.logger import get_risk_logger, log_alert

log = get_risk_logger()


@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    portfolio_var: float  # Value at Risk
    portfolio_cvar: float  # Conditional Value at Risk
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    portfolio_volatility: float
    correlation_matrix: pd.DataFrame


class RiskManager:
    """Manages all risk-related operations for the trading system."""

    def __init__(self, config: dict):
        """Initialize risk manager."""
        self.config = config
        self.portfolio_config = config.get('portfolio', {})
        self.position_config = config.get('position', {})
        self.drawdown_config = config.get('drawdown', {})
        self.metrics_config = config.get('metrics', {})

        # Risk limits
        self.annual_vol_target = self.portfolio_config.get('annual_volatility_target', 0.20)
        self.max_position_risk = self.position_config.get('max_position_risk', 0.02)
        self.max_drawdown = self.drawdown_config.get('max_drawdown', 0.20)

        # Portfolio state
        self.portfolio_value = 1000000  # Default initial value
        self.positions = {}
        self.historical_returns = []
        self.peak_portfolio_value = self.portfolio_value
        self.is_risk_limit_breached = False

    async def evaluate_signals(
        self,
        signals: List[Dict],
        market_data: Dict[str, pd.DataFrame]
    ) -> List[Dict]:
        """
        Evaluate trading signals against risk constraints.

        Args:
            signals: List of trading signals
            market_data: Current market data

        Returns:
            List of approved signals
        """
        approved_signals = []

        try:
            # Check if we're in risk-off mode
            if self.is_risk_limit_breached:
                log.warning("Risk limits breached - rejecting all signals")
                return []

            # Calculate current risk metrics
            current_metrics = self._calculate_risk_metrics(market_data)

            # Check overall portfolio risk
            if current_metrics.current_drawdown > self.max_drawdown:
                log_alert(f"Maximum drawdown exceeded: {current_metrics.current_drawdown:.2%}")
                self.is_risk_limit_breached = True
                return []

            # Evaluate each signal
            for signal in signals:
                if await self._evaluate_single_signal(signal, current_metrics, market_data):
                    approved_signals.append(signal)
                    log.debug(f"Approved signal: {signal['symbol']} {signal['action']}")
                else:
                    log.debug(f"Rejected signal: {signal['symbol']} {signal['action']}")

            log.info(f"Approved {len(approved_signals)}/{len(signals)} signals")

        except Exception as e:
            log.error(f"Error evaluating signals: {str(e)}")

        return approved_signals

    async def _evaluate_single_signal(
        self,
        signal: Dict,
        metrics: RiskMetrics,
        market_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """Evaluate a single trading signal."""
        symbol = signal['symbol']
        action = signal['action']

        # Check position concentration
        position_size = self._calculate_position_size(signal)
        position_pct = position_size / self.portfolio_value

        if position_pct > self.position_config.get('max_position_size', 0.15):
            log.warning(f"Position size too large for {symbol}: {position_pct:.2%}")
            return False

        # Check correlation with existing positions
        if symbol in market_data:
            correlation_check = self._check_correlation_limits(symbol, market_data)
            if not correlation_check:
                log.warning(f"Correlation limit exceeded for {symbol}")
                return False

        # Check volatility adjustment
        if not self._check_volatility_limits(signal, market_data):
            log.warning(f"Volatility limit exceeded for {symbol}")
            return False

        # Check stop loss and take profit levels
        if action == 'BUY':
            stop_loss, take_profit = self._calculate_risk_levels(signal, market_data)
            signal['stop_loss'] = stop_loss
            signal['take_profit'] = take_profit

        return True

    def _calculate_risk_metrics(self, market_data: Dict[str, pd.DataFrame]) -> RiskMetrics:
        """Calculate current portfolio risk metrics."""
        try:
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(market_data)

            # Calculate VaR and CVaR
            var_confidence = self.metrics_config.get('var_confidence', 0.95)
            portfolio_var = self._calculate_var(portfolio_returns, var_confidence)
            portfolio_cvar = self._calculate_cvar(portfolio_returns, var_confidence)

            # Calculate Sharpe and Sortino ratios
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
            sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)

            # Calculate drawdown
            max_drawdown, current_drawdown = self._calculate_drawdown()

            # Calculate portfolio volatility
            portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)

            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(market_data)

            return RiskMetrics(
                portfolio_var=portfolio_var,
                portfolio_cvar=portfolio_cvar,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                portfolio_volatility=portfolio_volatility,
                correlation_matrix=correlation_matrix
            )

        except Exception as e:
            log.error(f"Error calculating risk metrics: {str(e)}")
            # Return conservative metrics on error
            return RiskMetrics(
                portfolio_var=0,
                portfolio_cvar=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown=1.0,
                current_drawdown=1.0,
                portfolio_volatility=1.0,
                correlation_matrix=pd.DataFrame()
            )

    def _calculate_portfolio_returns(self, market_data: Dict[str, pd.DataFrame]) -> np.ndarray:
        """Calculate portfolio returns based on current positions."""
        portfolio_returns = []

        for symbol, position in self.positions.items():
            if symbol in market_data and not market_data[symbol].empty:
                data = market_data[symbol]
                returns = data['Close'].pct_change().dropna()
                weighted_returns = returns * (position['value'] / self.portfolio_value)
                portfolio_returns.append(weighted_returns)

        if portfolio_returns:
            combined_returns = pd.concat(portfolio_returns, axis=1).sum(axis=1)
            return combined_returns.values

        return np.array([0])

    def _calculate_var(self, returns: np.ndarray, confidence: float) -> float:
        """Calculate Value at Risk."""
        if len(returns) > 0:
            return np.percentile(returns, (1 - confidence) * 100)
        return 0

    def _calculate_cvar(self, returns: np.ndarray, confidence: float) -> float:
        """Calculate Conditional Value at Risk."""
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean() if len(returns[returns <= var]) > 0 else var

    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) > 0 and np.std(returns) > 0:
            excess_returns = returns - risk_free_rate / 252
            return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
        return 0

    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        if len(returns) > 0:
            excess_returns = returns - risk_free_rate / 252
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns)
                if downside_std > 0:
                    return np.mean(excess_returns) / downside_std * np.sqrt(252)
        return 0

    def _calculate_drawdown(self) -> Tuple[float, float]:
        """Calculate maximum and current drawdown."""
        current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value

        # Update peak if necessary
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value

        # Calculate max drawdown from history
        max_drawdown = current_drawdown  # Start with current

        if self.historical_returns:
            cumulative_returns = (1 + pd.Series(self.historical_returns)).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown_series = (running_max - cumulative_returns) / running_max
            historical_max_dd = drawdown_series.max()
            max_drawdown = max(max_drawdown, historical_max_dd)

        return max_drawdown, current_drawdown

    def _calculate_correlation_matrix(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate correlation matrix for current positions."""
        returns_dict = {}

        for symbol in self.positions.keys():
            if symbol in market_data and not market_data[symbol].empty:
                returns_dict[symbol] = market_data[symbol]['Close'].pct_change().dropna()

        if returns_dict:
            returns_df = pd.DataFrame(returns_dict)
            return returns_df.corr()

        return pd.DataFrame()

    def _check_correlation_limits(
        self,
        symbol: str,
        market_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """Check if adding position would exceed correlation limits."""
        if symbol not in market_data or market_data[symbol].empty:
            return True

        max_correlation = self.portfolio_config.get('max_correlation_threshold', 0.85)
        symbol_returns = market_data[symbol]['Close'].pct_change().dropna()

        for existing_symbol in self.positions.keys():
            if existing_symbol in market_data and not market_data[existing_symbol].empty:
                existing_returns = market_data[existing_symbol]['Close'].pct_change().dropna()

                # Align the series
                aligned_new = symbol_returns.iloc[-min(len(symbol_returns), len(existing_returns)):]
                aligned_existing = existing_returns.iloc[-min(len(symbol_returns), len(existing_returns)):]

                if len(aligned_new) > 20:  # Need sufficient data for correlation
                    correlation = aligned_new.corr(aligned_existing)
                    if abs(correlation) > max_correlation:
                        return False

        return True

    def _check_volatility_limits(
        self,
        signal: Dict,
        market_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """Check if position volatility is within limits."""
        symbol = signal['symbol']

        if symbol in market_data and not market_data[symbol].empty:
            data = market_data[symbol]
            returns = data['Close'].pct_change().dropna()

            if len(returns) > 20:
                position_volatility = np.std(returns) * np.sqrt(252)

                # Adjust for position size
                position_size = self._calculate_position_size(signal)
                position_weight = position_size / self.portfolio_value
                contribution_to_vol = position_volatility * position_weight

                # Check if adding this position keeps us within volatility target
                current_portfolio_vol = self._estimate_current_portfolio_volatility(market_data)
                projected_vol = np.sqrt(current_portfolio_vol**2 + contribution_to_vol**2)

                return projected_vol <= self.annual_vol_target

        return True

    def _estimate_current_portfolio_volatility(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> float:
        """Estimate current portfolio volatility."""
        if not self.positions:
            return 0

        portfolio_returns = self._calculate_portfolio_returns(market_data)
        if len(portfolio_returns) > 0:
            return np.std(portfolio_returns) * np.sqrt(252)

        return 0

    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size based on signal and risk parameters."""
        # Kelly Criterion with safety factor
        signal_strength = signal.get('strength', 1.0)
        base_size = self.portfolio_value * self.max_position_risk

        # Apply signal strength adjustment
        adjusted_size = base_size * signal_strength

        # Apply minimum and maximum constraints
        min_size = self.portfolio_value * self.position_config.get('min_position_size', 0.01)
        max_size = self.portfolio_value * self.position_config.get('max_position_size', 0.15)

        return np.clip(adjusted_size, min_size, max_size)

    def _calculate_risk_levels(
        self,
        signal: Dict,
        market_data: Dict[str, pd.DataFrame]
    ) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels."""
        symbol = signal['symbol']
        entry_price = signal['price']

        # Default levels
        stop_loss = entry_price * 0.95  # 5% stop loss
        take_profit = entry_price * 1.10  # 10% take profit

        if symbol in market_data and not market_data[symbol].empty:
            data = market_data[symbol]

            # Calculate ATR if available
            if 'High' in data.columns and 'Low' in data.columns:
                high_low = data['High'] - data['Low']
                high_close = abs(data['High'] - data['Close'].shift())
                low_close = abs(data['Low'] - data['Close'].shift())
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=14).mean().iloc[-1]

                # Use ATR-based levels
                stop_multiplier = self.position_config.get('stop_loss_multiplier', 2.0)
                profit_multiplier = self.position_config.get('take_profit_multiplier', 3.0)

                stop_loss = entry_price - (atr * stop_multiplier)
                take_profit = entry_price + (atr * profit_multiplier)

        return stop_loss, take_profit

    def update_portfolio_value(self, new_value: float):
        """Update portfolio value and track for drawdown calculation."""
        self.portfolio_value = new_value
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value

        # Check if we can resume trading after drawdown recovery
        if self.is_risk_limit_breached:
            _, current_drawdown = self._calculate_drawdown()
            recovery_threshold = self.drawdown_config.get('recovery_threshold', 0.10)

            if current_drawdown < recovery_threshold:
                self.is_risk_limit_breached = False
                log.info("Risk limits recovered - resuming normal operation")

    def update_positions(self, positions: Dict):
        """Update current positions."""
        self.positions = positions

    async def stop(self):
        """Stop risk manager."""
        log.info("Stopping risk manager...")
        # Perform any cleanup if needed
        log.info("Risk manager stopped")