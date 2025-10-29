"""
Historical crisis scenario testing module.

This module provides testing against historical market crises:
- Dot-com bubble (2000-2002)
- Financial crisis (2008-2009)
- Flash crash (2010)
- Taper tantrum (2013)
- China devaluation (2015)
- Volatility spike (2018)
- COVID-19 crash (2020)
- Rate shock / inflation surge (2022)

Tests strategy performance during extreme historical events.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


class CrisisType(Enum):
    """Types of historical crises."""
    DOTCOM_BUBBLE = "dotcom_bubble"
    FINANCIAL_CRISIS = "financial_crisis"
    FLASH_CRASH = "flash_crash"
    TAPER_TANTRUM = "taper_tantrum"
    CHINA_DEVALUATION = "china_devaluation"
    VOLATILITY_SPIKE = "volatility_spike"
    COVID_CRASH = "covid_crash"
    RATE_SHOCK = "rate_shock"


@dataclass
class CrisisScenario:
    """Container for crisis scenario definition."""
    name: str
    crisis_type: CrisisType
    start_date: datetime
    end_date: datetime
    description: str
    characteristics: Dict[str, Any]


class HistoricalScenarioTester:
    """
    Test strategy performance during historical crisis periods.

    Analyzes how strategies perform during extreme market events and
    compares crisis performance to normal market conditions.
    """

    def __init__(self):
        """Initialize historical scenario tester."""
        logger.info("HistoricalScenarioTester initialized")

        # Define predefined crisis scenarios
        self.crisis_scenarios = self._define_crisis_scenarios()

    def _define_crisis_scenarios(self) -> Dict[CrisisType, CrisisScenario]:
        """Define all predefined crisis scenarios."""
        scenarios = {}

        # Dot-com Bubble Burst (2000-2002)
        scenarios[CrisisType.DOTCOM_BUBBLE] = CrisisScenario(
            name="Dot-com Bubble Burst",
            crisis_type=CrisisType.DOTCOM_BUBBLE,
            start_date=datetime(2000, 3, 10),
            end_date=datetime(2002, 10, 9),
            description="Tech bubble collapse with NASDAQ losing ~78% from peak",
            characteristics={
                'duration_days': 943,
                'market_decline': -0.78,
                'sector_rotation': True,
                'liquidity_crisis': False,
                'credit_crisis': False,
            }
        )

        # Financial Crisis (2008-2009)
        scenarios[CrisisType.FINANCIAL_CRISIS] = CrisisScenario(
            name="Global Financial Crisis",
            crisis_type=CrisisType.FINANCIAL_CRISIS,
            start_date=datetime(2007, 10, 9),
            end_date=datetime(2009, 3, 9),
            description="Subprime mortgage crisis and credit market collapse",
            characteristics={
                'duration_days': 517,
                'market_decline': -0.57,
                'sector_rotation': True,
                'liquidity_crisis': True,
                'credit_crisis': True,
                'volatility_spike': True,
            }
        )

        # Flash Crash (2010)
        scenarios[CrisisType.FLASH_CRASH] = CrisisScenario(
            name="Flash Crash",
            crisis_type=CrisisType.FLASH_CRASH,
            start_date=datetime(2010, 5, 6),
            end_date=datetime(2010, 5, 6),
            description="Rapid intraday market crash and recovery",
            characteristics={
                'duration_days': 1,
                'intraday_decline': -0.09,
                'liquidity_crisis': True,
                'algorithmic_failure': True,
                'rapid_recovery': True,
            }
        )

        # Taper Tantrum (2013)
        scenarios[CrisisType.TAPER_TANTRUM] = CrisisScenario(
            name="Taper Tantrum",
            crisis_type=CrisisType.TAPER_TANTRUM,
            start_date=datetime(2013, 5, 22),
            end_date=datetime(2013, 9, 18),
            description="Market reaction to Fed QE tapering announcement",
            characteristics={
                'duration_days': 119,
                'bond_selloff': True,
                'emerging_market_impact': True,
                'currency_volatility': True,
            }
        )

        # China Devaluation (2015)
        scenarios[CrisisType.CHINA_DEVALUATION] = CrisisScenario(
            name="China Devaluation",
            crisis_type=CrisisType.CHINA_DEVALUATION,
            start_date=datetime(2015, 8, 11),
            end_date=datetime(2015, 9, 29),
            description="Chinese yuan devaluation and market turmoil",
            characteristics={
                'duration_days': 49,
                'market_decline': -0.12,
                'currency_crisis': True,
                'emerging_market_contagion': True,
                'commodity_impact': True,
            }
        )

        # Volatility Spike / VIXplosion (2018)
        scenarios[CrisisType.VOLATILITY_SPIKE] = CrisisScenario(
            name="Volmageddon / VIX Spike",
            crisis_type=CrisisType.VOLATILITY_SPIKE,
            start_date=datetime(2018, 1, 29),
            end_date=datetime(2018, 2, 8),
            description="Extreme VIX spike and volatility product collapse",
            characteristics={
                'duration_days': 10,
                'market_decline': -0.10,
                'vix_spike': True,
                'volatility_product_collapse': True,
                'systematic_selling': True,
            }
        )

        # COVID-19 Crash (2020)
        scenarios[CrisisType.COVID_CRASH] = CrisisScenario(
            name="COVID-19 Pandemic Crash",
            crisis_type=CrisisType.COVID_CRASH,
            start_date=datetime(2020, 2, 19),
            end_date=datetime(2020, 3, 23),
            description="Rapid pandemic-driven market collapse",
            characteristics={
                'duration_days': 33,
                'market_decline': -0.34,
                'liquidity_crisis': True,
                'volatility_spike': True,
                'unprecedented_intervention': True,
                'rapid_recovery': True,
            }
        )

        # Rate Shock / Inflation Surge (2022)
        scenarios[CrisisType.RATE_SHOCK] = CrisisScenario(
            name="Rate Shock & Inflation Surge",
            crisis_type=CrisisType.RATE_SHOCK,
            start_date=datetime(2022, 1, 3),
            end_date=datetime(2022, 10, 13),
            description="Aggressive Fed rate hikes and inflation concerns",
            characteristics={
                'duration_days': 283,
                'market_decline': -0.25,
                'rate_shock': True,
                'bond_bear_market': True,
                'growth_stock_collapse': True,
                'inflation_surge': True,
            }
        )

        return scenarios

    def test_strategy_against_crises(
        self,
        strategy_returns: pd.Series,
        market_returns: Optional[pd.Series] = None,
        crises: Optional[List[CrisisType]] = None
    ) -> Dict[str, Any]:
        """
        Test strategy performance during historical crises.

        Args:
            strategy_returns: Strategy returns time series
            market_returns: Optional market returns for comparison
            crises: List of crisis types to test (None for all)

        Returns:
            Dictionary with crisis test results
        """
        logger.info("Testing strategy against historical crises")

        if crises is None:
            crises = list(CrisisType)

        results = {}

        for crisis_type in crises:
            if crisis_type not in self.crisis_scenarios:
                logger.warning(f"Crisis scenario {crisis_type} not defined")
                continue

            scenario = self.crisis_scenarios[crisis_type]

            # Test this crisis
            crisis_result = self._test_single_crisis(
                strategy_returns,
                scenario,
                market_returns
            )

            results[crisis_type.value] = crisis_result

        # Overall summary
        results['summary'] = self._summarize_crisis_tests(results)

        logger.success(f"Tested strategy against {len(crises)} crisis scenarios")
        return results

    def _test_single_crisis(
        self,
        strategy_returns: pd.Series,
        scenario: CrisisScenario,
        market_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Test strategy during a single crisis period.

        Args:
            strategy_returns: Strategy returns
            scenario: Crisis scenario
            market_returns: Optional market returns

        Returns:
            Dictionary with test results
        """
        logger.debug(f"Testing crisis: {scenario.name}")

        # Extract crisis period returns
        crisis_mask = (
            (strategy_returns.index >= scenario.start_date) &
            (strategy_returns.index <= scenario.end_date)
        )

        crisis_returns = strategy_returns[crisis_mask]

        if len(crisis_returns) == 0:
            logger.warning(f"No data available for crisis period: {scenario.name}")
            return {
                'scenario': scenario.name,
                'data_available': False,
            }

        # Calculate crisis metrics
        result = {
            'scenario': scenario.name,
            'crisis_type': scenario.crisis_type.value,
            'start_date': scenario.start_date,
            'end_date': scenario.end_date,
            'data_available': True,
            'n_periods': len(crisis_returns),
        }

        # Performance metrics
        result['total_return'] = (1 + crisis_returns).prod() - 1
        result['annualized_return'] = self._annualize_return(
            result['total_return'],
            len(crisis_returns)
        )
        result['volatility'] = crisis_returns.std() * np.sqrt(252)
        result['max_drawdown'] = self._calculate_max_drawdown(crisis_returns)

        # Risk metrics
        result['var_95'] = np.percentile(crisis_returns, 5)
        result['cvar_95'] = crisis_returns[crisis_returns <= result['var_95']].mean()
        result['worst_day'] = crisis_returns.min()
        result['best_day'] = crisis_returns.max()

        # Win rate
        result['positive_days'] = (crisis_returns > 0).sum() / len(crisis_returns)

        # Compare to normal periods if market returns available
        if market_returns is not None:
            result['market_comparison'] = self._compare_to_market(
                crisis_returns,
                market_returns,
                crisis_mask
            )

        # Compare to normal market conditions
        normal_returns = strategy_returns[~crisis_mask]
        if len(normal_returns) > 0:
            result['normal_comparison'] = self._compare_to_normal(
                crisis_returns,
                normal_returns
            )

        return result

    def _annualize_return(self, total_return: float, n_periods: int, periods_per_year: int = 252) -> float:
        """Annualize a total return."""
        if n_periods == 0:
            return 0.0

        years = n_periods / periods_per_year
        if years <= 0:
            return 0.0

        return (1 + total_return) ** (1 / years) - 1

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        return drawdown.min()

    def _compare_to_market(
        self,
        strategy_crisis_returns: pd.Series,
        market_returns: pd.Series,
        crisis_mask: pd.Series
    ) -> Dict[str, float]:
        """Compare strategy to market during crisis."""
        market_crisis_returns = market_returns[crisis_mask]

        # Align returns
        common_idx = strategy_crisis_returns.index.intersection(market_crisis_returns.index)

        if len(common_idx) == 0:
            return {}

        strat_aligned = strategy_crisis_returns.loc[common_idx]
        market_aligned = market_crisis_returns.loc[common_idx]

        return {
            'beta': self._calculate_beta(strat_aligned, market_aligned),
            'correlation': strat_aligned.corr(market_aligned),
            'relative_return': (1 + strat_aligned).prod() - (1 + market_aligned).prod(),
            'outperformance_days': (strat_aligned > market_aligned).sum() / len(common_idx),
        }

    def _calculate_beta(self, strategy_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta."""
        covariance = np.cov(strategy_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)

        if market_variance == 0:
            return 0.0

        return covariance / market_variance

    def _compare_to_normal(
        self,
        crisis_returns: pd.Series,
        normal_returns: pd.Series
    ) -> Dict[str, float]:
        """Compare crisis performance to normal market conditions."""
        return {
            'return_ratio': ((1 + crisis_returns).prod() - 1) / ((1 + normal_returns).prod() - 1)
                          if (1 + normal_returns).prod() - 1 != 0 else 0,
            'volatility_ratio': (crisis_returns.std() / normal_returns.std()
                               if normal_returns.std() != 0 else 0),
            't_test_returns': self._t_test_returns(crisis_returns, normal_returns),
        }

    def _t_test_returns(
        self,
        crisis_returns: pd.Series,
        normal_returns: pd.Series
    ) -> Dict[str, float]:
        """Perform t-test comparing crisis vs normal returns."""
        t_stat, p_value = stats.ttest_ind(crisis_returns, normal_returns)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
        }

    def _summarize_crisis_tests(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize all crisis test results."""
        # Filter out summary key
        crisis_results = {k: v for k, v in results.items() if k != 'summary'}

        # Filter out results without data
        valid_results = [r for r in crisis_results.values() if r.get('data_available', False)]

        if not valid_results:
            return {'n_crises_tested': 0}

        summary = {
            'n_crises_tested': len(valid_results),
            'avg_crisis_return': np.mean([r['total_return'] for r in valid_results]),
            'avg_crisis_drawdown': np.mean([r['max_drawdown'] for r in valid_results]),
            'avg_crisis_volatility': np.mean([r['volatility'] for r in valid_results]),
            'positive_crisis_count': sum(1 for r in valid_results if r['total_return'] > 0),
            'negative_crisis_count': sum(1 for r in valid_results if r['total_return'] < 0),
            'worst_crisis': min(valid_results, key=lambda r: r['total_return'])['scenario'],
            'best_crisis': max(valid_results, key=lambda r: r['total_return'])['scenario'],
        }

        summary['crisis_win_rate'] = (summary['positive_crisis_count'] /
                                      summary['n_crises_tested'])

        return summary

    def extract_crisis_period_data(
        self,
        data: pd.DataFrame,
        crisis_type: CrisisType
    ) -> pd.DataFrame:
        """
        Extract data for a specific crisis period.

        Args:
            data: Full dataset
            crisis_type: Type of crisis

        Returns:
            DataFrame filtered to crisis period
        """
        if crisis_type not in self.crisis_scenarios:
            logger.error(f"Crisis type {crisis_type} not defined")
            return pd.DataFrame()

        scenario = self.crisis_scenarios[crisis_type]

        mask = (data.index >= scenario.start_date) & (data.index <= scenario.end_date)
        crisis_data = data[mask]

        logger.info(f"Extracted {len(crisis_data)} records for {scenario.name}")

        return crisis_data

    def get_crisis_characteristics(self, crisis_type: CrisisType) -> Dict[str, Any]:
        """
        Get characteristics of a specific crisis.

        Args:
            crisis_type: Type of crisis

        Returns:
            Dictionary with crisis characteristics
        """
        if crisis_type not in self.crisis_scenarios:
            logger.error(f"Crisis type {crisis_type} not defined")
            return {}

        scenario = self.crisis_scenarios[crisis_type]

        return {
            'name': scenario.name,
            'description': scenario.description,
            'start_date': scenario.start_date,
            'end_date': scenario.end_date,
            'characteristics': scenario.characteristics,
        }

    def compare_crises(
        self,
        strategy_returns: pd.Series,
        crises: Optional[List[CrisisType]] = None
    ) -> pd.DataFrame:
        """
        Compare strategy performance across multiple crises.

        Args:
            strategy_returns: Strategy returns time series
            crises: List of crises to compare (None for all)

        Returns:
            DataFrame with comparison metrics
        """
        logger.info("Comparing strategy performance across crises")

        results = self.test_strategy_against_crises(strategy_returns, crises=crises)

        # Build comparison table
        comparison_data = []

        for crisis_name, crisis_result in results.items():
            if crisis_name == 'summary' or not crisis_result.get('data_available', False):
                continue

            comparison_data.append({
                'Crisis': crisis_result['scenario'],
                'Start Date': crisis_result['start_date'],
                'End Date': crisis_result['end_date'],
                'Total Return': crisis_result['total_return'],
                'Annualized Return': crisis_result['annualized_return'],
                'Volatility': crisis_result['volatility'],
                'Max Drawdown': crisis_result['max_drawdown'],
                'VaR (95%)': crisis_result['var_95'],
                'Worst Day': crisis_result['worst_day'],
                'Positive Days %': crisis_result['positive_days'] * 100,
            })

        comparison_df = pd.DataFrame(comparison_data)

        return comparison_df

    def plot_crisis_performance(
        self,
        strategy_returns: pd.Series,
        crisis_type: CrisisType
    ) -> 'matplotlib.figure.Figure':
        """
        Plot strategy performance during a specific crisis.

        Args:
            strategy_returns: Strategy returns time series
            crisis_type: Type of crisis to plot

        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt

        if crisis_type not in self.crisis_scenarios:
            logger.error(f"Crisis type {crisis_type} not defined")
            return plt.figure()

        scenario = self.crisis_scenarios[crisis_type]

        # Extract crisis period
        crisis_mask = (
            (strategy_returns.index >= scenario.start_date) &
            (strategy_returns.index <= scenario.end_date)
        )

        crisis_returns = strategy_returns[crisis_mask]

        if len(crisis_returns) == 0:
            logger.warning(f"No data for crisis: {scenario.name}")
            return plt.figure()

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Cumulative returns
        ax = axes[0, 0]
        cum_returns = (1 + crisis_returns).cumprod() - 1
        ax.plot(cum_returns.index, cum_returns * 100, linewidth=2)
        ax.set_title(f'{scenario.name} - Cumulative Returns', fontweight='bold')
        ax.set_ylabel('Return (%)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

        # 2. Daily returns
        ax = axes[0, 1]
        ax.bar(crisis_returns.index, crisis_returns * 100, alpha=0.7)
        ax.set_title('Daily Returns', fontweight='bold')
        ax.set_ylabel('Return (%)')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

        # 3. Drawdown
        ax = axes[1, 0]
        cum_ret_series = (1 + crisis_returns).cumprod()
        running_max = cum_ret_series.expanding().max()
        drawdown = (cum_ret_series - running_max) / running_max

        ax.fill_between(drawdown.index, drawdown * 100, 0, alpha=0.5, color='red')
        ax.plot(drawdown.index, drawdown * 100, linewidth=1, color='darkred')
        ax.set_title('Drawdown', fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)

        # 4. Returns distribution
        ax = axes[1, 1]
        ax.hist(crisis_returns * 100, bins=30, alpha=0.7, edgecolor='black')
        ax.set_title('Returns Distribution', fontweight='bold')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2)

        plt.suptitle(f'Crisis Performance Analysis: {scenario.name}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig
