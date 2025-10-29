"""
Extreme stress testing module.

This module implements extreme stress scenarios:
- Correlation breakdown (all correlations → 1.0)
- Volatility explosion (VIX +100%, +200%)
- Liquidity evaporation (spreads widen 5x, 10x)
- Fat tail events (5-sigma, 6-sigma moves)
- Black swan simulation
- Cascade effects (multi-stage crisis)
- Contagion modeling
- Recovery scenarios (V, U, L-shaped)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


class StressType(Enum):
    """Types of stress tests."""
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    VOLATILITY_EXPLOSION = "volatility_explosion"
    LIQUIDITY_EVAPORATION = "liquidity_evaporation"
    FAT_TAIL_EVENT = "fat_tail_event"
    BLACK_SWAN = "black_swan"
    CASCADE_CRISIS = "cascade_crisis"
    CONTAGION = "contagion"


@dataclass
class StressTestResult:
    """Container for stress test results."""
    stress_type: str
    scenario_name: str
    stressed_returns: pd.Series
    portfolio_impact: Dict[str, float]
    risk_metrics: Dict[str, float]
    comparison_to_base: Dict[str, float]


class CorrelationBreakdown:
    """
    Simulate correlation breakdown scenarios.

    Models extreme market conditions where asset correlations converge to 1.0.
    """

    def __init__(self):
        """Initialize correlation breakdown scenario."""
        logger.info("CorrelationBreakdown initialized")

    def stress_test(
        self,
        returns_df: pd.DataFrame,
        target_correlation: float = 1.0,
        n_scenarios: int = 100
    ) -> List[pd.DataFrame]:
        """
        Generate scenarios with forced correlations.

        Args:
            returns_df: Multi-asset returns DataFrame
            target_correlation: Target correlation (e.g., 0.95, 1.0)
            n_scenarios: Number of scenarios to generate

        Returns:
            List of stressed return scenarios
        """
        logger.info(f"Generating correlation breakdown scenarios (target={target_correlation})")

        # Calculate original statistics
        means = returns_df.mean()
        stds = returns_df.std()

        # Create target correlation matrix
        n_assets = len(returns_df.columns)
        target_corr_matrix = np.full((n_assets, n_assets), target_correlation)
        np.fill_diagonal(target_corr_matrix, 1.0)

        # Generate scenarios
        scenarios = []

        for _ in range(n_scenarios):
            # Generate correlated normals
            L = np.linalg.cholesky(target_corr_matrix)
            uncorrelated = np.random.randn(len(returns_df), n_assets)
            correlated = uncorrelated @ L.T

            # Scale to match original distributions
            stressed_returns = pd.DataFrame(
                correlated,
                columns=returns_df.columns,
                index=returns_df.index
            )

            for col in returns_df.columns:
                stressed_returns[col] = (stressed_returns[col] * stds[col]) + means[col]

            scenarios.append(stressed_returns)

        logger.success(f"Generated {n_scenarios} correlation breakdown scenarios")
        return scenarios


class VolatilityExplosion:
    """
    Simulate volatility explosion scenarios.

    Models extreme volatility spikes (e.g., VIX doubling or tripling).
    """

    def __init__(self):
        """Initialize volatility explosion scenario."""
        logger.info("VolatilityExplosion initialized")

    def stress_test(
        self,
        returns: pd.Series,
        volatility_multiplier: float = 2.0,
        duration: int = 20,
        n_scenarios: int = 100
    ) -> List[pd.Series]:
        """
        Generate scenarios with volatility shocks.

        Args:
            returns: Historical returns
            volatility_multiplier: Multiplier for volatility (e.g., 2.0 = double)
            duration: Duration of volatility spike in periods
            n_scenarios: Number of scenarios

        Returns:
            List of stressed return scenarios
        """
        logger.info(f"Generating volatility explosion scenarios (multiplier={volatility_multiplier})")

        # Calculate base statistics
        mu = returns.mean()
        sigma = returns.std()

        scenarios = []

        for _ in range(n_scenarios):
            # Normal period
            normal_length = len(returns) - duration
            normal_returns = np.random.normal(mu, sigma, normal_length)

            # Volatility spike period
            spike_returns = np.random.normal(mu, sigma * volatility_multiplier, duration)

            # Combine
            scenario = np.concatenate([normal_returns, spike_returns])
            scenarios.append(pd.Series(scenario))

        logger.success(f"Generated {n_scenarios} volatility explosion scenarios")
        return scenarios

    def progressive_volatility_increase(
        self,
        returns: pd.Series,
        peak_multiplier: float = 3.0,
        ramp_up_periods: int = 20,
        plateau_periods: int = 10,
        ramp_down_periods: int = 20
    ) -> pd.Series:
        """
        Generate scenario with progressive volatility increase.

        Args:
            returns: Historical returns
            peak_multiplier: Peak volatility multiplier
            ramp_up_periods: Periods to reach peak
            plateau_periods: Periods at peak
            ramp_down_periods: Periods to return to normal

        Returns:
            Stressed return scenario
        """
        logger.info("Generating progressive volatility scenario")

        mu = returns.mean()
        sigma = returns.std()

        # Create volatility profile
        normal_periods = len(returns) - (ramp_up_periods + plateau_periods + ramp_down_periods)

        # Normal volatility
        normal_vol = np.ones(normal_periods)

        # Ramp up
        ramp_up = np.linspace(1.0, peak_multiplier, ramp_up_periods)

        # Plateau
        plateau = np.ones(plateau_periods) * peak_multiplier

        # Ramp down
        ramp_down = np.linspace(peak_multiplier, 1.0, ramp_down_periods)

        # Combine
        vol_profile = np.concatenate([normal_vol, ramp_up, plateau, ramp_down])

        # Generate returns with varying volatility
        stressed_returns = np.random.normal(mu, sigma * vol_profile)

        return pd.Series(stressed_returns)


class LiquidityEvaporation:
    """
    Simulate liquidity crisis scenarios.

    Models scenarios where bid-ask spreads widen dramatically.
    """

    def __init__(self):
        """Initialize liquidity evaporation scenario."""
        logger.info("LiquidityEvaporation initialized")

    def stress_test(
        self,
        returns: pd.Series,
        trades_df: Optional[pd.DataFrame] = None,
        spread_multiplier: float = 5.0,
        slippage_multiplier: float = 10.0
    ) -> Dict[str, Any]:
        """
        Calculate impact of liquidity crisis.

        Args:
            returns: Strategy returns
            trades_df: Optional trades DataFrame
            spread_multiplier: Multiplier for bid-ask spreads
            slippage_multiplier: Multiplier for slippage

        Returns:
            Dictionary with stressed performance metrics
        """
        logger.info(f"Testing liquidity evaporation (spread_multiplier={spread_multiplier})")

        results = {
            'spread_multiplier': spread_multiplier,
            'slippage_multiplier': slippage_multiplier,
        }

        # If trades data available, calculate actual impact
        if trades_df is not None and 'commission' in trades_df.columns:
            # Estimate additional costs
            base_commission = trades_df['commission'].mean()
            stressed_commission = base_commission * spread_multiplier

            base_slippage = trades_df.get('slippage', pd.Series()).mean()
            stressed_slippage = base_slippage * slippage_multiplier

            additional_cost_per_trade = (
                (stressed_commission - base_commission) +
                (stressed_slippage - base_slippage)
            )

            total_additional_cost = additional_cost_per_trade * len(trades_df)

            # Calculate impact on returns
            portfolio_value = 1000000  # Assume $1M portfolio
            return_impact = total_additional_cost / portfolio_value

            results['additional_cost_per_trade'] = additional_cost_per_trade
            results['total_additional_cost'] = total_additional_cost
            results['return_impact'] = return_impact
            results['stressed_total_return'] = returns.sum() - return_impact

        else:
            # Estimate impact based on typical trading costs
            estimated_turnover = 2.0  # Assume 200% annual turnover
            base_cost = 0.0010  # 10 bps
            stressed_cost = base_cost * spread_multiplier

            cost_impact = estimated_turnover * stressed_cost

            results['estimated_cost_impact'] = cost_impact
            results['stressed_total_return'] = returns.sum() - cost_impact

        return results


class StressTestEngine:
    """
    Main stress test engine.

    Coordinates multiple stress testing scenarios.
    """

    def __init__(self):
        """Initialize stress test engine."""
        self.correlation_breakdown = CorrelationBreakdown()
        self.volatility_explosion = VolatilityExplosion()
        self.liquidity_evaporation = LiquidityEvaporation()
        logger.info("StressTestEngine initialized")

    def run_comprehensive_stress_test(
        self,
        returns: pd.Series,
        multi_asset_returns: Optional[pd.DataFrame] = None,
        trades_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive stress testing suite.

        Args:
            returns: Strategy returns
            multi_asset_returns: Optional multi-asset returns
            trades_df: Optional trades DataFrame

        Returns:
            Dictionary with all stress test results
        """
        logger.info("Running comprehensive stress test suite")

        results = {}

        # Fat tail events
        results['fat_tail_events'] = self.test_fat_tail_events(returns)

        # Black swan scenarios
        results['black_swan'] = self.test_black_swan(returns)

        # Volatility explosions
        results['volatility_explosion'] = self.test_volatility_scenarios(returns)

        # Liquidity crisis
        results['liquidity_crisis'] = self.liquidity_evaporation.stress_test(
            returns, trades_df
        )

        # Correlation breakdown (if multi-asset data available)
        if multi_asset_returns is not None:
            results['correlation_breakdown'] = self.test_correlation_breakdown(
                multi_asset_returns
            )

        # Cascade crisis
        results['cascade_crisis'] = self.test_cascade_crisis(returns)

        logger.success("Comprehensive stress test completed")
        return results

    def test_fat_tail_events(
        self,
        returns: pd.Series,
        sigma_levels: List[float] = [3, 4, 5, 6]
    ) -> Dict[str, Any]:
        """
        Test strategy under fat tail events.

        Args:
            returns: Strategy returns
            sigma_levels: List of sigma levels to test

        Returns:
            Dictionary with fat tail test results
        """
        logger.info(f"Testing fat tail events at {sigma_levels} sigma levels")

        mu = returns.mean()
        sigma = returns.std()

        results = {}

        for sigma_level in sigma_levels:
            # Negative shock
            neg_shock = mu - (sigma_level * sigma)

            # Simulate portfolio with this shock
            shocked_returns = returns.copy()
            shock_idx = len(returns) // 2  # Place shock in middle

            shocked_returns.iloc[shock_idx] = neg_shock

            # Calculate impact
            results[f'{sigma_level}_sigma'] = {
                'shock_return': neg_shock,
                'portfolio_impact': (1 + shocked_returns).prod() - (1 + returns).prod(),
                'max_drawdown_increase': self._calculate_max_drawdown(shocked_returns) -
                                        self._calculate_max_drawdown(returns),
            }

        return results

    def test_black_swan(
        self,
        returns: pd.Series,
        swan_magnitude: float = -0.20,
        recovery_scenarios: List[str] = ['V', 'U', 'L']
    ) -> Dict[str, Any]:
        """
        Test black swan scenarios with different recovery patterns.

        Args:
            returns: Strategy returns
            swan_magnitude: Magnitude of black swan event (e.g., -0.20 = -20%)
            recovery_scenarios: Types of recovery ('V'=fast, 'U'=slow, 'L'=no recovery)

        Returns:
            Dictionary with black swan test results
        """
        logger.info(f"Testing black swan scenarios (magnitude={swan_magnitude})")

        results = {}

        for recovery_type in recovery_scenarios:
            scenario = self._simulate_black_swan_recovery(
                returns,
                swan_magnitude,
                recovery_type
            )

            results[f'recovery_{recovery_type}'] = {
                'total_return': (1 + scenario).prod() - 1,
                'max_drawdown': self._calculate_max_drawdown(scenario),
                'recovery_time': self._calculate_recovery_time(scenario),
            }

        return results

    def _simulate_black_swan_recovery(
        self,
        returns: pd.Series,
        swan_magnitude: float,
        recovery_type: str
    ) -> pd.Series:
        """Simulate black swan event with specified recovery pattern."""
        scenario = returns.copy()

        # Place swan event
        swan_idx = len(returns) // 2
        scenario.iloc[swan_idx] = swan_magnitude

        # Simulate recovery
        recovery_periods = 60  # ~3 months

        if recovery_type == 'V':
            # Fast V-shaped recovery
            recovery = np.linspace(0, abs(swan_magnitude), recovery_periods)
            for i, rec in enumerate(recovery[:min(recovery_periods, len(scenario) - swan_idx - 1)]):
                scenario.iloc[swan_idx + i + 1] += rec / recovery_periods

        elif recovery_type == 'U':
            # Slow U-shaped recovery
            recovery_periods *= 2
            recovery = np.linspace(0, abs(swan_magnitude), recovery_periods)
            for i, rec in enumerate(recovery[:min(recovery_periods, len(scenario) - swan_idx - 1)]):
                scenario.iloc[swan_idx + i + 1] += rec / recovery_periods / 2

        elif recovery_type == 'L':
            # No recovery (permanent loss)
            pass

        return scenario

    def test_volatility_scenarios(
        self,
        returns: pd.Series
    ) -> Dict[str, Any]:
        """Test multiple volatility scenarios."""
        results = {}

        # 2x volatility
        scenarios_2x = self.volatility_explosion.stress_test(
            returns,
            volatility_multiplier=2.0,
            n_scenarios=100
        )

        results['2x_volatility'] = self._analyze_scenarios(scenarios_2x)

        # 3x volatility
        scenarios_3x = self.volatility_explosion.stress_test(
            returns,
            volatility_multiplier=3.0,
            n_scenarios=100
        )

        results['3x_volatility'] = self._analyze_scenarios(scenarios_3x)

        # Progressive increase
        progressive = self.volatility_explosion.progressive_volatility_increase(returns)
        results['progressive_increase'] = {
            'total_return': (1 + progressive).prod() - 1,
            'max_drawdown': self._calculate_max_drawdown(progressive),
            'volatility': progressive.std() * np.sqrt(252),
        }

        return results

    def test_correlation_breakdown(
        self,
        returns_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Test correlation breakdown scenarios."""
        results = {}

        # Test different correlation levels
        for target_corr in [0.8, 0.9, 1.0]:
            scenarios = self.correlation_breakdown.stress_test(
                returns_df,
                target_correlation=target_corr,
                n_scenarios=100
            )

            # Analyze portfolio impact
            portfolio_returns = [s.mean(axis=1) for s in scenarios]
            avg_return = np.mean([r.sum() for r in portfolio_returns])
            avg_volatility = np.mean([r.std() * np.sqrt(252) for r in portfolio_returns])

            results[f'correlation_{target_corr}'] = {
                'avg_return': avg_return,
                'avg_volatility': avg_volatility,
                'avg_sharpe': avg_return / avg_volatility if avg_volatility > 0 else 0,
            }

        return results

    def test_cascade_crisis(
        self,
        returns: pd.Series,
        n_stages: int = 3,
        stage_magnitude: float = -0.10
    ) -> Dict[str, Any]:
        """
        Test multi-stage cascade crisis.

        Args:
            returns: Strategy returns
            n_stages: Number of crisis stages
            stage_magnitude: Magnitude of each stage

        Returns:
            Dictionary with cascade results
        """
        logger.info(f"Testing {n_stages}-stage cascade crisis")

        scenario = returns.copy()

        # Apply multiple crisis stages
        stage_spacing = len(returns) // (n_stages + 1)

        for stage in range(n_stages):
            stage_idx = stage_spacing * (stage + 1)

            if stage_idx < len(scenario):
                # Each stage compounds the crisis
                scenario.iloc[stage_idx] = stage_magnitude * (1 + stage * 0.5)

        results = {
            'n_stages': n_stages,
            'total_return': (1 + scenario).prod() - 1,
            'max_drawdown': self._calculate_max_drawdown(scenario),
            'worst_period_return': scenario.min(),
        }

        return results

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    def _calculate_recovery_time(self, returns: pd.Series) -> int:
        """Calculate time to recover from maximum drawdown."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max

        # Find max drawdown point
        max_dd_idx = drawdown.idxmin()

        # Find recovery point
        recovery_level = running_max.loc[max_dd_idx]
        recovery_mask = (cum_returns.loc[max_dd_idx:] >= recovery_level)

        if recovery_mask.any():
            recovery_idx = recovery_mask.idxmax()
            return len(returns.loc[max_dd_idx:recovery_idx])
        else:
            return len(returns) - returns.index.get_loc(max_dd_idx)

    def _analyze_scenarios(self, scenarios: List[pd.Series]) -> Dict[str, float]:
        """Analyze a list of scenarios."""
        total_returns = [(1 + s).prod() - 1 for s in scenarios]
        max_drawdowns = [self._calculate_max_drawdown(s) for s in scenarios]
        volatilities = [s.std() * np.sqrt(252) for s in scenarios]

        return {
            'mean_return': np.mean(total_returns),
            'worst_return': np.min(total_returns),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.min(max_drawdowns),
            'mean_volatility': np.mean(volatilities),
            'max_volatility': np.max(volatilities),
        }
