"""
Performance attribution analysis for portfolio backtesting.

This module provides comprehensive attribution analysis including:
- Return attribution by strategy component
- Return attribution by asset class/sector
- Time-based attribution
- Risk factor attribution (Fama-French style)
- Brinson attribution (allocation vs selection)
- Risk attribution (contribution to portfolio variance)
- Alpha decomposition
"""

from typing import Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings

from utils.logger import get_backtest_logger

log = get_backtest_logger()


class AttributionAnalysis:
    """
    Comprehensive performance attribution analysis.

    Decomposes portfolio returns and risk into various components
    to understand sources of performance.
    """

    def __init__(
        self,
        portfolio_returns: pd.Series,
        holdings: Optional[pd.DataFrame] = None,
        component_returns: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
        factor_returns: Optional[pd.DataFrame] = None
    ):
        """
        Initialize attribution analysis.

        Args:
            portfolio_returns: Portfolio return series
            holdings: DataFrame of holdings/weights over time
            component_returns: Returns of individual components/assets
            benchmark_returns: Benchmark return series
            factor_returns: Factor return series (e.g., Fama-French factors)
        """
        self.portfolio_returns = portfolio_returns.dropna()
        self.holdings = holdings
        self.component_returns = component_returns
        self.benchmark_returns = benchmark_returns.dropna() if benchmark_returns is not None else None
        self.factor_returns = factor_returns

        log.info("Initialized AttributionAnalysis")

    def calculate_full_attribution(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate comprehensive attribution analysis.

        Returns:
            Dictionary of attribution DataFrames
        """
        log.info("Calculating full attribution analysis")

        results = {}

        try:
            # Component attribution
            if self.component_returns is not None:
                results['component_attribution'] = self.component_attribution()

            # Time-based attribution
            results['monthly_attribution'] = self.time_period_attribution('M')
            results['quarterly_attribution'] = self.time_period_attribution('Q')
            results['annual_attribution'] = self.time_period_attribution('Y')

            # Benchmark attribution (Brinson)
            if self.benchmark_returns is not None and self.holdings is not None:
                results['brinson_attribution'] = self.brinson_attribution()

            # Factor attribution
            if self.factor_returns is not None:
                results['factor_attribution'] = self.factor_attribution()

            # Risk attribution
            if self.component_returns is not None and self.holdings is not None:
                results['risk_attribution'] = self.risk_attribution()

            log.success(f"Calculated {len(results)} attribution analyses")

        except Exception as e:
            log.error(f"Error in attribution analysis: {str(e)}")
            raise

        return results

    # ===========================
    # Component Attribution
    # ===========================

    def component_attribution(self) -> pd.DataFrame:
        """
        Calculate return attribution by component/asset.

        Returns:
            DataFrame with component attribution
        """
        if self.component_returns is None or self.holdings is None:
            log.warning("Component returns or holdings not provided")
            return pd.DataFrame()

        # Align data
        common_idx = self.component_returns.index.intersection(self.holdings.index)
        component_returns = self.component_returns.loc[common_idx]
        holdings = self.holdings.loc[common_idx]

        # Calculate weighted returns for each component
        weighted_returns = component_returns * holdings

        # Calculate contribution to total return
        total_return = weighted_returns.sum(axis=1)

        # Attribution summary
        attribution = pd.DataFrame({
            'contribution': weighted_returns.sum(axis=0),
            'avg_weight': holdings.mean(axis=0),
            'component_return': component_returns.mean(axis=0),
        })

        attribution['contribution_pct'] = (
            attribution['contribution'] / total_return.sum() * 100
        )

        return attribution.sort_values('contribution', ascending=False)

    def component_attribution_by_period(
        self,
        period: str = 'M'
    ) -> pd.DataFrame:
        """
        Calculate component attribution by time period.

        Args:
            period: Resampling period ('D', 'W', 'M', 'Q', 'Y')

        Returns:
            DataFrame with periodic component attribution
        """
        if self.component_returns is None or self.holdings is None:
            return pd.DataFrame()

        common_idx = self.component_returns.index.intersection(self.holdings.index)
        component_returns = self.component_returns.loc[common_idx]
        holdings = self.holdings.loc[common_idx]

        # Calculate weighted returns
        weighted_returns = component_returns * holdings

        # Resample to period
        period_attribution = weighted_returns.resample(period).sum()

        return period_attribution

    # ===========================
    # Time-Based Attribution
    # ===========================

    def time_period_attribution(self, period: str = 'M') -> pd.DataFrame:
        """
        Calculate attribution by time period.

        Args:
            period: Resampling period ('M', 'Q', 'Y')

        Returns:
            DataFrame with period attribution
        """
        # Resample returns
        period_returns = self.portfolio_returns.resample(period).apply(
            lambda x: (1 + x).prod() - 1
        )

        attribution = pd.DataFrame({
            'return': period_returns,
            'contribution': period_returns,
        })

        # Calculate cumulative contribution
        attribution['cumulative_return'] = (1 + attribution['return']).cumprod() - 1

        if self.benchmark_returns is not None:
            benchmark_period = self.benchmark_returns.resample(period).apply(
                lambda x: (1 + x).prod() - 1
            )
            attribution['benchmark_return'] = benchmark_period
            attribution['active_return'] = attribution['return'] - benchmark_period

        return attribution

    def rolling_attribution(
        self,
        window: int = 252,
        components: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling attribution analysis.

        Args:
            window: Rolling window size
            components: List of components to analyze

        Returns:
            DataFrame with rolling attribution
        """
        if self.component_returns is None or self.holdings is None:
            return pd.DataFrame()

        if components is None:
            components = list(self.component_returns.columns)

        rolling_attribution = {}

        for component in components:
            if component not in self.component_returns.columns:
                continue

            weighted_return = (
                self.component_returns[component] * self.holdings[component]
            )

            rolling_contribution = weighted_return.rolling(window).sum()
            rolling_attribution[component] = rolling_contribution

        return pd.DataFrame(rolling_attribution)

    # ===========================
    # Brinson Attribution
    # ===========================

    def brinson_attribution(self) -> pd.DataFrame:
        """
        Calculate Brinson attribution (allocation vs selection).

        Decomposes active return into:
        - Allocation effect: Portfolio weight vs benchmark weight
        - Selection effect: Asset return vs benchmark return
        - Interaction effect: Combined allocation and selection

        Returns:
            DataFrame with Brinson attribution
        """
        if self.holdings is None or self.benchmark_returns is None:
            log.warning("Holdings or benchmark not provided for Brinson attribution")
            return pd.DataFrame()

        if self.component_returns is None:
            log.warning("Component returns not provided for Brinson attribution")
            return pd.DataFrame()

        # For simplified Brinson, we need benchmark weights
        # Assuming equal-weighted benchmark for demonstration
        assets = list(self.component_returns.columns)
        benchmark_weights = pd.DataFrame(
            1.0 / len(assets),
            index=self.holdings.index,
            columns=assets
        )

        # Align data
        common_idx = (
            self.component_returns.index
            .intersection(self.holdings.index)
            .intersection(benchmark_weights.index)
        )

        portfolio_weights = self.holdings.loc[common_idx]
        benchmark_weights = benchmark_weights.loc[common_idx]
        component_returns = self.component_returns.loc[common_idx]

        # Calculate benchmark component returns
        benchmark_returns = component_returns.mean(axis=1)

        # Allocation effect: (w_p - w_b) * (r_b - r_benchmark)
        weight_diff = portfolio_weights - benchmark_weights
        return_diff_from_bench = component_returns.sub(benchmark_returns, axis=0)

        allocation_effect = (weight_diff * return_diff_from_bench).sum(axis=0)

        # Selection effect: w_b * (r_p - r_b)
        # Assuming portfolio return equals component return (simplified)
        selection_effect = (
            benchmark_weights * (component_returns - component_returns.mean(axis=1, keepdims=True))
        ).sum(axis=0).sum()

        # Interaction effect: (w_p - w_b) * (r_p - r_b)
        interaction_effect = (
            weight_diff * return_diff_from_bench
        ).sum(axis=0)

        # Create attribution summary
        attribution = pd.DataFrame({
            'allocation_effect': allocation_effect,
            'selection_effect': selection_effect / len(assets),  # Distribute equally
            'interaction_effect': interaction_effect,
        })

        attribution['total_effect'] = (
            attribution['allocation_effect'] +
            attribution['selection_effect'] +
            attribution['interaction_effect']
        )

        return attribution

    def brinson_attribution_by_period(
        self,
        period: str = 'M'
    ) -> pd.DataFrame:
        """
        Calculate Brinson attribution by time period.

        Args:
            period: Time period for attribution ('M', 'Q', 'Y')

        Returns:
            DataFrame with periodic Brinson attribution
        """
        # This is a simplified version
        # Full implementation would require period-by-period calculation
        log.warning("Period-based Brinson attribution is simplified")

        return self.brinson_attribution()

    # ===========================
    # Factor Attribution
    # ===========================

    def factor_attribution(self) -> pd.DataFrame:
        """
        Calculate factor-based attribution (Fama-French style).

        Uses factor returns to explain portfolio performance.

        Returns:
            DataFrame with factor attribution
        """
        if self.factor_returns is None:
            log.warning("Factor returns not provided")
            return pd.DataFrame()

        # Align returns
        common_idx = self.portfolio_returns.index.intersection(
            self.factor_returns.index
        )

        if len(common_idx) < 20:
            log.warning("Insufficient data for factor attribution")
            return pd.DataFrame()

        portfolio_returns = self.portfolio_returns.loc[common_idx]
        factor_returns = self.factor_returns.loc[common_idx]

        # Run factor regression
        X = factor_returns.values
        y = portfolio_returns.values

        # Add constant for alpha
        X_with_const = np.column_stack([np.ones(len(X)), X])

        # OLS regression
        try:
            from numpy.linalg import lstsq
            coefficients, residuals, rank, s = lstsq(X_with_const, y, rcond=None)

            alpha = coefficients[0]
            factor_betas = coefficients[1:]

            # Calculate factor contributions
            factor_contributions = factor_betas * factor_returns.mean().values

            # Create attribution DataFrame
            attribution = pd.DataFrame({
                'factor': ['Alpha'] + list(factor_returns.columns),
                'beta': [1.0] + list(factor_betas),
                'factor_return': [alpha] + list(factor_returns.mean().values),
                'contribution': [alpha] + list(factor_contributions),
            })

            # Calculate R-squared
            ss_res = np.sum((y - X_with_const @ coefficients) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            attribution['r_squared'] = r_squared

            return attribution

        except Exception as e:
            log.error(f"Error in factor regression: {str(e)}")
            return pd.DataFrame()

    def rolling_factor_attribution(
        self,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Calculate rolling factor attribution.

        Args:
            window: Rolling window size

        Returns:
            DataFrame with rolling factor betas
        """
        if self.factor_returns is None:
            return pd.DataFrame()

        common_idx = self.portfolio_returns.index.intersection(
            self.factor_returns.index
        )

        if len(common_idx) < window:
            log.warning(f"Insufficient data for rolling factor attribution")
            return pd.DataFrame()

        portfolio_returns = self.portfolio_returns.loc[common_idx]
        factor_returns = self.factor_returns.loc[common_idx]

        rolling_betas = {}

        # Calculate rolling betas for each factor
        for factor in factor_returns.columns:
            rolling_betas[f'{factor}_beta'] = portfolio_returns.rolling(
                window
            ).cov(factor_returns[factor]) / factor_returns[factor].rolling(window).var()

        # Calculate rolling alpha
        def calculate_alpha(idx):
            if idx < window:
                return np.nan

            window_portfolio = portfolio_returns.iloc[idx-window:idx]
            window_factors = factor_returns.iloc[idx-window:idx]

            X = window_factors.values
            y = window_portfolio.values

            X_with_const = np.column_stack([np.ones(len(X)), X])

            try:
                coefficients, _, _, _ = np.linalg.lstsq(X_with_const, y, rcond=None)
                return coefficients[0]
            except:
                return np.nan

        rolling_betas['alpha'] = [
            calculate_alpha(i) for i in range(len(portfolio_returns))
        ]

        return pd.DataFrame(rolling_betas, index=common_idx)

    def factor_tilts(self) -> Dict[str, float]:
        """
        Calculate factor tilts (exposures) relative to market.

        Returns:
            Dictionary of factor tilts
        """
        attribution = self.factor_attribution()

        if attribution.empty:
            return {}

        tilts = {}
        for _, row in attribution.iterrows():
            if row['factor'] != 'Alpha':
                tilts[row['factor']] = row['beta']

        return tilts

    # ===========================
    # Risk Attribution
    # ===========================

    def risk_attribution(self) -> pd.DataFrame:
        """
        Calculate risk attribution (contribution to portfolio variance).

        Decomposes portfolio variance into component contributions.

        Returns:
            DataFrame with risk attribution
        """
        if self.component_returns is None or self.holdings is None:
            log.warning("Component returns or holdings required for risk attribution")
            return pd.DataFrame()

        # Get average weights
        avg_weights = self.holdings.mean(axis=0)

        # Calculate covariance matrix
        cov_matrix = self.component_returns.cov()

        # Portfolio variance
        portfolio_variance = avg_weights @ cov_matrix @ avg_weights

        # Marginal contribution to risk (MCR)
        mcr = cov_matrix @ avg_weights

        # Component contribution to risk (CCR)
        ccr = avg_weights * mcr

        # Percentage contribution to risk
        pcr = ccr / portfolio_variance * 100 if portfolio_variance > 0 else ccr * 0

        # Create attribution DataFrame
        attribution = pd.DataFrame({
            'weight': avg_weights,
            'marginal_contrib': mcr,
            'component_contrib': ccr,
            'pct_contrib': pcr,
        })

        # Calculate component volatilities
        attribution['component_vol'] = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)

        return attribution.sort_values('component_contrib', ascending=False)

    def marginal_risk_contribution(
        self,
        weights: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate marginal risk contribution for each component.

        Args:
            weights: Component weights (uses average if None)

        Returns:
            Series of marginal risk contributions
        """
        if self.component_returns is None:
            return pd.Series()

        if weights is None:
            if self.holdings is None:
                return pd.Series()
            weights = self.holdings.mean(axis=0)

        # Calculate covariance matrix
        cov_matrix = self.component_returns.cov()

        # Marginal contribution to risk
        mcr = cov_matrix @ weights

        # Portfolio volatility
        portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)

        # Normalized marginal contribution
        if portfolio_vol > 0:
            mcr = mcr / portfolio_vol

        return mcr

    def risk_decomposition(self) -> Dict[str, float]:
        """
        Decompose portfolio risk into components.

        Returns:
            Dictionary with risk decomposition
        """
        if self.component_returns is None or self.holdings is None:
            return {}

        # Calculate portfolio variance
        avg_weights = self.holdings.mean(axis=0)
        cov_matrix = self.component_returns.cov()

        portfolio_variance = avg_weights @ cov_matrix @ avg_weights
        portfolio_vol = np.sqrt(portfolio_variance) * np.sqrt(252)

        # Individual component volatilities
        component_vols = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)

        # Weighted average of component vols (diversification benchmark)
        weighted_vol = np.sum(avg_weights * component_vols)

        # Diversification benefit
        diversification_benefit = weighted_vol - portfolio_vol

        return {
            'portfolio_volatility': portfolio_vol,
            'weighted_avg_volatility': weighted_vol,
            'diversification_benefit': diversification_benefit,
            'diversification_ratio': portfolio_vol / weighted_vol if weighted_vol > 0 else 0,
        }

    # ===========================
    # Alpha Decomposition
    # ===========================

    def alpha_decomposition(self) -> Dict[str, float]:
        """
        Decompose alpha into components.

        Returns:
            Dictionary with alpha decomposition
        """
        if self.benchmark_returns is None:
            log.warning("Benchmark required for alpha decomposition")
            return {}

        # Calculate total alpha
        common_idx = self.portfolio_returns.index.intersection(
            self.benchmark_returns.index
        )

        portfolio_return = self.portfolio_returns.loc[common_idx].mean() * 252
        benchmark_return = self.benchmark_returns.loc[common_idx].mean() * 252

        # Simple alpha (excess return)
        total_alpha = portfolio_return - benchmark_return

        decomposition = {
            'total_alpha': total_alpha,
            'portfolio_return': portfolio_return,
            'benchmark_return': benchmark_return,
        }

        # Factor-based alpha if available
        if self.factor_returns is not None:
            factor_attr = self.factor_attribution()
            if not factor_attr.empty:
                alpha_row = factor_attr[factor_attr['factor'] == 'Alpha']
                if not alpha_row.empty:
                    decomposition['factor_alpha'] = alpha_row['contribution'].iloc[0]

        # Component-based alpha if available
        if self.component_returns is not None and self.holdings is not None:
            component_attr = self.component_attribution()
            if not component_attr.empty:
                # Attribution by component
                top_contributors = component_attr.nlargest(3, 'contribution')
                decomposition['top_contributors'] = top_contributors['contribution'].to_dict()

        return decomposition

    # ===========================
    # Sector/Asset Class Attribution
    # ===========================

    def sector_attribution(
        self,
        sector_mapping: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Calculate attribution by sector/asset class.

        Args:
            sector_mapping: Dictionary mapping assets to sectors

        Returns:
            DataFrame with sector attribution
        """
        if self.component_returns is None or self.holdings is None:
            return pd.DataFrame()

        # Group components by sector
        sector_returns = {}
        sector_weights = {}

        for asset, sector in sector_mapping.items():
            if asset not in self.component_returns.columns:
                continue

            if sector not in sector_returns:
                sector_returns[sector] = []
                sector_weights[sector] = []

            sector_returns[sector].append(self.component_returns[asset])
            sector_weights[sector].append(self.holdings[asset])

        # Calculate sector-level metrics
        attribution = []

        for sector in sector_returns.keys():
            # Aggregate returns and weights
            returns = pd.concat(sector_returns[sector], axis=1)
            weights = pd.concat(sector_weights[sector], axis=1)

            # Calculate weighted returns
            weighted_ret = (returns * weights).sum(axis=1)

            # Calculate average weight
            avg_weight = weights.sum(axis=1).mean()

            # Calculate sector return
            sector_return = returns.mean(axis=1).mean()

            attribution.append({
                'sector': sector,
                'contribution': weighted_ret.sum(),
                'avg_weight': avg_weight,
                'sector_return': sector_return,
            })

        attribution_df = pd.DataFrame(attribution)

        if not attribution_df.empty:
            total_contribution = attribution_df['contribution'].sum()
            if total_contribution != 0:
                attribution_df['contribution_pct'] = (
                    attribution_df['contribution'] / total_contribution * 100
                )

        return attribution_df.sort_values('contribution', ascending=False)

    # ===========================
    # Utility Functions
    # ===========================

    def active_return_decomposition(self) -> Dict[str, float]:
        """
        Decompose active return into components.

        Returns:
            Dictionary with active return decomposition
        """
        if self.benchmark_returns is None:
            return {}

        common_idx = self.portfolio_returns.index.intersection(
            self.benchmark_returns.index
        )

        portfolio_ret = self.portfolio_returns.loc[common_idx]
        benchmark_ret = self.benchmark_returns.loc[common_idx]

        active_return = (portfolio_ret - benchmark_ret).sum()
        portfolio_total = portfolio_ret.sum()
        benchmark_total = benchmark_ret.sum()

        return {
            'active_return': active_return,
            'portfolio_return': portfolio_total,
            'benchmark_return': benchmark_total,
            'active_return_annualized': active_return * 252 / len(common_idx),
            'information_ratio': (
                active_return / (portfolio_ret - benchmark_ret).std()
                if (portfolio_ret - benchmark_ret).std() > 0 else 0
            ),
        }

    def summary_attribution(self) -> pd.DataFrame:
        """
        Generate summary attribution report.

        Returns:
            DataFrame with summary attribution
        """
        summary = []

        # Component attribution summary
        if self.component_returns is not None and self.holdings is not None:
            comp_attr = self.component_attribution()
            if not comp_attr.empty:
                top_3 = comp_attr.nlargest(3, 'contribution')
                for _, row in top_3.iterrows():
                    summary.append({
                        'type': 'Component',
                        'name': row.name,
                        'contribution': row['contribution'],
                        'weight': row['avg_weight'],
                    })

        # Factor attribution summary
        if self.factor_returns is not None:
            factor_attr = self.factor_attribution()
            if not factor_attr.empty:
                for _, row in factor_attr.iterrows():
                    summary.append({
                        'type': 'Factor',
                        'name': row['factor'],
                        'contribution': row['contribution'],
                        'weight': row['beta'],
                    })

        # Time period summary
        annual_attr = self.time_period_attribution('Y')
        if not annual_attr.empty:
            for idx, row in annual_attr.iterrows():
                summary.append({
                    'type': 'Annual',
                    'name': str(idx.year),
                    'contribution': row['return'],
                    'weight': 1.0,
                })

        return pd.DataFrame(summary)


def create_attribution_report(
    portfolio_returns: pd.Series,
    holdings: Optional[pd.DataFrame] = None,
    component_returns: Optional[pd.DataFrame] = None,
    benchmark_returns: Optional[pd.Series] = None,
    factor_returns: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Create comprehensive attribution report.

    Args:
        portfolio_returns: Portfolio returns
        holdings: Holdings over time
        component_returns: Component returns
        benchmark_returns: Benchmark returns
        factor_returns: Factor returns
        output_path: Optional path to save report

    Returns:
        Dictionary of attribution DataFrames
    """
    analyzer = AttributionAnalysis(
        portfolio_returns=portfolio_returns,
        holdings=holdings,
        component_returns=component_returns,
        benchmark_returns=benchmark_returns,
        factor_returns=factor_returns
    )

    results = analyzer.calculate_full_attribution()

    if output_path:
        # Save to Excel
        try:
            with pd.ExcelWriter(output_path) as writer:
                for sheet_name, df in results.items():
                    df.to_excel(writer, sheet_name=sheet_name[:31])  # Excel limit
            log.success(f"Attribution report saved to {output_path}")
        except Exception as e:
            log.error(f"Could not save attribution report: {str(e)}")

    return results
