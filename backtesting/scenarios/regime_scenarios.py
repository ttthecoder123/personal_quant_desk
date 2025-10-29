"""
Regime-specific scenario testing module.

This module generates and tests strategies under different market regimes:
- Bull market scenarios (steady growth)
- Bear market scenarios (recession, drawdown)
- Sideways/ranging market scenarios
- High volatility regimes (VIX > 30)
- Low volatility regimes (VIX < 15)
- Rising/falling rate environments
- Inflation/deflation scenarios
- Sector rotation scenarios
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.cluster import KMeans
from hmmlearn import hmm


class RegimeType(Enum):
    """Types of market regimes."""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RISING_RATES = "rising_rates"
    FALLING_RATES = "falling_rates"
    INFLATION = "inflation"
    DEFLATION = "deflation"
    SECTOR_ROTATION = "sector_rotation"


@dataclass
class MarketRegime:
    """Container for market regime definition."""
    regime_type: RegimeType
    name: str
    description: str
    characteristics: Dict[str, Any]
    typical_duration_days: int


class RegimeScenarioGenerator:
    """
    Generate and test regime-specific scenarios.

    Identifies market regimes and generates scenarios for each regime type.
    """

    def __init__(self):
        """Initialize regime scenario generator."""
        logger.info("RegimeScenarioGenerator initialized")

        # Define regime templates
        self.regime_templates = self._define_regime_templates()

    def _define_regime_templates(self) -> Dict[RegimeType, MarketRegime]:
        """Define templates for different market regimes."""
        templates = {}

        # Bull Market
        templates[RegimeType.BULL_MARKET] = MarketRegime(
            regime_type=RegimeType.BULL_MARKET,
            name="Bull Market",
            description="Strong upward trend with low volatility",
            characteristics={
                'mean_return': 0.0010,  # 1 bps per day
                'volatility': 0.01,
                'trend': 'up',
                'volatility_regime': 'low',
                'typical_sharpe': 1.5,
            },
            typical_duration_days=500
        )

        # Bear Market
        templates[RegimeType.BEAR_MARKET] = MarketRegime(
            regime_type=RegimeType.BEAR_MARKET,
            name="Bear Market",
            description="Declining market with elevated volatility",
            characteristics={
                'mean_return': -0.0005,  # -5 bps per day
                'volatility': 0.02,
                'trend': 'down',
                'volatility_regime': 'high',
                'typical_sharpe': -0.5,
            },
            typical_duration_days=200
        )

        # Sideways Market
        templates[RegimeType.SIDEWAYS] = MarketRegime(
            regime_type=RegimeType.SIDEWAYS,
            name="Sideways/Ranging Market",
            description="No clear trend, mean-reverting",
            characteristics={
                'mean_return': 0.0,
                'volatility': 0.012,
                'trend': 'none',
                'mean_reversion': True,
                'typical_sharpe': 0.0,
            },
            typical_duration_days=150
        )

        # High Volatility
        templates[RegimeType.HIGH_VOLATILITY] = MarketRegime(
            regime_type=RegimeType.HIGH_VOLATILITY,
            name="High Volatility Regime",
            description="Elevated volatility (VIX > 30)",
            characteristics={
                'mean_return': -0.0002,
                'volatility': 0.025,
                'vix_level': 35,
                'volatility_regime': 'high',
            },
            typical_duration_days=60
        )

        # Low Volatility
        templates[RegimeType.LOW_VOLATILITY] = MarketRegime(
            regime_type=RegimeType.LOW_VOLATILITY,
            name="Low Volatility Regime",
            description="Suppressed volatility (VIX < 15)",
            characteristics={
                'mean_return': 0.0005,
                'volatility': 0.006,
                'vix_level': 12,
                'volatility_regime': 'low',
            },
            typical_duration_days=120
        )

        # Rising Rates
        templates[RegimeType.RISING_RATES] = MarketRegime(
            regime_type=RegimeType.RISING_RATES,
            name="Rising Rate Environment",
            description="Central bank tightening cycle",
            characteristics={
                'mean_return': 0.0002,
                'volatility': 0.015,
                'rate_trend': 'up',
                'bond_performance': 'negative',
                'growth_stock_pressure': True,
            },
            typical_duration_days=300
        )

        # Falling Rates
        templates[RegimeType.FALLING_RATES] = MarketRegime(
            regime_type=RegimeType.FALLING_RATES,
            name="Falling Rate Environment",
            description="Central bank easing cycle",
            characteristics={
                'mean_return': 0.0008,
                'volatility': 0.018,
                'rate_trend': 'down',
                'bond_performance': 'positive',
                'growth_stock_support': True,
            },
            typical_duration_days=250
        )

        # Inflation
        templates[RegimeType.INFLATION] = MarketRegime(
            regime_type=RegimeType.INFLATION,
            name="Inflationary Environment",
            description="Rising inflation pressures",
            characteristics={
                'mean_return': 0.0001,
                'volatility': 0.02,
                'inflation_regime': 'high',
                'commodity_strength': True,
                'real_rate_pressure': True,
            },
            typical_duration_days=400
        )

        return templates

    def identify_historical_regimes(
        self,
        returns: pd.Series,
        n_regimes: int = 3,
        method: str = 'hmm'
    ) -> Tuple[pd.Series, Dict]:
        """
        Identify market regimes in historical data.

        Args:
            returns: Historical returns
            n_regimes: Number of regimes to identify
            method: Identification method ('hmm', 'kmeans', 'rolling')

        Returns:
            Tuple of (regime labels, regime statistics)
        """
        logger.info(f"Identifying {n_regimes} market regimes using {method}")

        if method == 'hmm':
            labels, stats = self._identify_regimes_hmm(returns, n_regimes)
        elif method == 'kmeans':
            labels, stats = self._identify_regimes_kmeans(returns, n_regimes)
        elif method == 'rolling':
            labels, stats = self._identify_regimes_rolling(returns, n_regimes)
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.success(f"Identified {n_regimes} regimes")
        return labels, stats

    def _identify_regimes_hmm(
        self,
        returns: pd.Series,
        n_regimes: int
    ) -> Tuple[pd.Series, Dict]:
        """Identify regimes using Hidden Markov Model."""
        # Prepare features: returns and rolling volatility
        vol_window = 20
        features = pd.DataFrame({
            'returns': returns,
            'volatility': returns.rolling(vol_window).std()
        }).dropna()

        # Fit HMM
        model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )

        model.fit(features.values)

        # Predict regimes
        labels = model.predict(features.values)

        # Align with original index
        regime_series = pd.Series(labels, index=features.index)

        # Calculate regime statistics
        stats = self._calculate_regime_statistics(returns, regime_series, n_regimes)

        return regime_series, stats

    def _identify_regimes_kmeans(
        self,
        returns: pd.Series,
        n_regimes: int
    ) -> Tuple[pd.Series, Dict]:
        """Identify regimes using K-means clustering."""
        # Prepare features
        vol_window = 20
        features = pd.DataFrame({
            'returns': returns,
            'volatility': returns.rolling(vol_window).std(),
            'trend': returns.rolling(vol_window).mean()
        }).dropna()

        # Fit K-means
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features.values)

        # Create regime series
        regime_series = pd.Series(labels, index=features.index)

        # Calculate statistics
        stats = self._calculate_regime_statistics(returns, regime_series, n_regimes)

        return regime_series, stats

    def _identify_regimes_rolling(
        self,
        returns: pd.Series,
        n_regimes: int
    ) -> Tuple[pd.Series, Dict]:
        """Identify regimes using rolling statistics and quantiles."""
        window = 60

        # Calculate rolling metrics
        rolling_return = returns.rolling(window).mean()
        rolling_vol = returns.rolling(window).std()

        # Classify based on quantiles
        return_quantiles = pd.qcut(rolling_return.dropna(), q=n_regimes, labels=False)
        vol_quantiles = pd.qcut(rolling_vol.dropna(), q=n_regimes, labels=False)

        # Combine (simplified - use volatility primarily)
        regime_series = vol_quantiles

        # Calculate statistics
        stats = self._calculate_regime_statistics(returns, regime_series, n_regimes)

        return regime_series, stats

    def _calculate_regime_statistics(
        self,
        returns: pd.Series,
        regimes: pd.Series,
        n_regimes: int
    ) -> Dict[int, Dict]:
        """Calculate statistics for each identified regime."""
        stats = {}

        for regime_id in range(n_regimes):
            regime_mask = regimes == regime_id
            regime_returns = returns[regime_mask]

            if len(regime_returns) > 0:
                stats[regime_id] = {
                    'mean_return': regime_returns.mean(),
                    'volatility': regime_returns.std(),
                    'sharpe': (regime_returns.mean() / regime_returns.std() * np.sqrt(252)
                             if regime_returns.std() > 0 else 0),
                    'skewness': stats.skew(regime_returns),
                    'n_periods': len(regime_returns),
                    'percentage_of_time': len(regime_returns) / len(returns) * 100,
                }

        return stats

    def generate_regime_scenario(
        self,
        regime_type: RegimeType,
        duration: Optional[int] = None
    ) -> pd.Series:
        """
        Generate synthetic scenario for specific regime.

        Args:
            regime_type: Type of regime to simulate
            duration: Duration in days (None for template default)

        Returns:
            Simulated returns for the regime
        """
        logger.info(f"Generating {regime_type.value} scenario")

        if regime_type not in self.regime_templates:
            raise ValueError(f"Unknown regime type: {regime_type}")

        template = self.regime_templates[regime_type]

        if duration is None:
            duration = template.typical_duration_days

        # Generate based on regime characteristics
        if template.characteristics.get('mean_reversion', False):
            # Mean-reverting process for sideways market
            scenario = self._generate_mean_reverting_scenario(
                duration,
                template.characteristics['mean_return'],
                template.characteristics['volatility']
            )
        elif template.characteristics.get('trend') == 'up':
            # Trending up for bull market
            scenario = self._generate_trending_scenario(
                duration,
                template.characteristics['mean_return'],
                template.characteristics['volatility'],
                trend_strength=1.0
            )
        elif template.characteristics.get('trend') == 'down':
            # Trending down for bear market
            scenario = self._generate_trending_scenario(
                duration,
                template.characteristics['mean_return'],
                template.characteristics['volatility'],
                trend_strength=-1.0
            )
        else:
            # Default: simple random walk
            scenario = pd.Series(
                np.random.normal(
                    template.characteristics['mean_return'],
                    template.characteristics['volatility'],
                    duration
                )
            )

        logger.success(f"Generated {duration}-day {regime_type.value} scenario")
        return scenario

    def _generate_mean_reverting_scenario(
        self,
        duration: int,
        mean_return: float,
        volatility: float,
        reversion_speed: float = 0.1
    ) -> pd.Series:
        """Generate mean-reverting scenario (Ornstein-Uhlenbeck process)."""
        returns = np.zeros(duration)
        value = 0.0

        for i in range(duration):
            # Mean reversion term
            reversion = -reversion_speed * value

            # Random shock
            shock = np.random.normal(0, volatility)

            # Update
            value += reversion + shock
            returns[i] = mean_return + value

        return pd.Series(returns)

    def _generate_trending_scenario(
        self,
        duration: int,
        mean_return: float,
        volatility: float,
        trend_strength: float = 1.0
    ) -> pd.Series:
        """Generate trending scenario with momentum."""
        returns = np.zeros(duration)

        # Add trend component
        trend = np.linspace(0, trend_strength * mean_return * 2, duration)

        # Add random component
        random_component = np.random.normal(mean_return, volatility, duration)

        returns = trend + random_component

        return pd.Series(returns)

    def test_strategy_across_regimes(
        self,
        strategy_returns: pd.Series,
        market_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Test strategy performance across different market regimes.

        Args:
            strategy_returns: Strategy returns
            market_returns: Optional market returns for regime identification

        Returns:
            Dictionary with regime-specific performance
        """
        logger.info("Testing strategy across market regimes")

        # Identify regimes
        if market_returns is not None:
            regimes, regime_stats = self.identify_historical_regimes(market_returns)
        else:
            regimes, regime_stats = self.identify_historical_regimes(strategy_returns)

        # Align strategy returns with regimes
        aligned_returns = strategy_returns[strategy_returns.index.isin(regimes.index)]
        aligned_regimes = regimes[regimes.index.isin(strategy_returns.index)]

        # Calculate performance for each regime
        results = {}

        for regime_id, stats in regime_stats.items():
            regime_mask = aligned_regimes == regime_id
            regime_performance = aligned_returns[regime_mask]

            if len(regime_performance) > 0:
                results[f'regime_{regime_id}'] = {
                    'market_characteristics': stats,
                    'strategy_performance': {
                        'total_return': (1 + regime_performance).prod() - 1,
                        'mean_return': regime_performance.mean(),
                        'volatility': regime_performance.std() * np.sqrt(252),
                        'sharpe': (regime_performance.mean() / regime_performance.std() * np.sqrt(252)
                                 if regime_performance.std() > 0 else 0),
                        'max_drawdown': self._calculate_max_drawdown(regime_performance),
                        'win_rate': (regime_performance > 0).sum() / len(regime_performance),
                    }
                }

        # Summary
        results['summary'] = self._summarize_regime_performance(results)

        logger.success("Regime analysis completed")
        return results

    def generate_multi_regime_scenario(
        self,
        regime_sequence: List[Tuple[RegimeType, int]]
    ) -> pd.Series:
        """
        Generate scenario with multiple regime transitions.

        Args:
            regime_sequence: List of (regime_type, duration) tuples

        Returns:
            Combined scenario with regime transitions
        """
        logger.info(f"Generating multi-regime scenario with {len(regime_sequence)} regimes")

        scenarios = []

        for regime_type, duration in regime_sequence:
            scenario = self.generate_regime_scenario(regime_type, duration)
            scenarios.append(scenario)

        # Concatenate
        combined = pd.concat(scenarios, ignore_index=True)

        logger.success(f"Generated {len(combined)}-day multi-regime scenario")
        return combined

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0

        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    def _summarize_regime_performance(self, results: Dict) -> Dict:
        """Summarize performance across all regimes."""
        regime_results = {k: v for k, v in results.items() if k.startswith('regime_')}

        if not regime_results:
            return {}

        # Best and worst regimes
        sharpe_ratios = {k: v['strategy_performance']['sharpe']
                        for k, v in regime_results.items()}

        best_regime = max(sharpe_ratios, key=sharpe_ratios.get)
        worst_regime = min(sharpe_ratios, key=sharpe_ratios.get)

        # Average across regimes
        avg_sharpe = np.mean(list(sharpe_ratios.values()))
        avg_volatility = np.mean([v['strategy_performance']['volatility']
                                 for v in regime_results.values()])

        return {
            'n_regimes_identified': len(regime_results),
            'best_regime': best_regime,
            'best_regime_sharpe': sharpe_ratios[best_regime],
            'worst_regime': worst_regime,
            'worst_regime_sharpe': sharpe_ratios[worst_regime],
            'avg_sharpe_across_regimes': avg_sharpe,
            'avg_volatility_across_regimes': avg_volatility,
            'regime_consistency': min(sharpe_ratios.values()) / max(sharpe_ratios.values())
                                 if max(sharpe_ratios.values()) != 0 else 0,
        }

    def plot_regime_analysis(
        self,
        returns: pd.Series,
        regimes: pd.Series,
        regime_stats: Dict
    ) -> 'matplotlib.figure.Figure':
        """
        Plot regime analysis results.

        Args:
            returns: Returns series
            regimes: Regime labels
            regime_stats: Regime statistics

        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(14, 12))

        # 1. Cumulative returns colored by regime
        ax = axes[0]

        cum_returns = (1 + returns).cumprod()

        # Plot with regime coloring
        for regime_id in sorted(regimes.unique()):
            regime_mask = regimes == regime_id
            regime_data = cum_returns[regime_mask]

            ax.plot(regime_data.index, regime_data.values,
                   label=f'Regime {regime_id}', linewidth=2)

        ax.set_title('Cumulative Returns by Regime', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Regime characteristics
        ax = axes[1]

        regime_ids = list(regime_stats.keys())
        mean_returns = [regime_stats[r]['mean_return'] * 252 * 100 for r in regime_ids]
        volatilities = [regime_stats[r]['volatility'] * np.sqrt(252) * 100 for r in regime_ids]

        x = np.arange(len(regime_ids))
        width = 0.35

        ax.bar(x - width/2, mean_returns, width, label='Annual Return (%)', alpha=0.7)
        ax.bar(x + width/2, volatilities, width, label='Annual Volatility (%)', alpha=0.7)

        ax.set_title('Regime Characteristics', fontsize=14, fontweight='bold')
        ax.set_xlabel('Regime', fontsize=12)
        ax.set_ylabel('Percentage', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Regime {r}' for r in regime_ids])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Regime duration
        ax = axes[2]

        durations = [regime_stats[r]['n_periods'] for r in regime_ids]
        percentages = [regime_stats[r]['percentage_of_time'] for r in regime_ids]

        ax.bar(x, percentages, alpha=0.7, edgecolor='black')
        ax.set_title('Regime Duration Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Regime', fontsize=12)
        ax.set_ylabel('Percentage of Time (%)', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Regime {r}' for r in regime_ids])
        ax.grid(True, alpha=0.3, axis='y')

        # Add duration annotations
        for i, (dur, pct) in enumerate(zip(durations, percentages)):
            ax.text(i, pct + 1, f'{dur} days', ha='center', fontsize=10)

        plt.tight_layout()
        return fig
