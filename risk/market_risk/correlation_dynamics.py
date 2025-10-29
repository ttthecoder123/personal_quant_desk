"""
Dynamic Correlation Analysis using DCC-GARCH and advanced techniques.

Implements institutional-grade correlation modeling:
- DCC-GARCH for time-varying correlations
- Correlation regime identification
- Stress correlation scenarios
- Correlation breakdown detection
- Cross-asset correlation monitoring
- Lead-lag relationships
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
from scipy.signal import correlate
from loguru import logger

try:
    from arch import arch_model
    from arch.univariate import GARCH
    ARCH_AVAILABLE = True
except ImportError:
    logger.warning("arch package not available - DCC-GARCH features limited")
    ARCH_AVAILABLE = False


@dataclass
class CorrelationMetrics:
    """Container for correlation metrics."""
    correlation_matrix: pd.DataFrame
    average_correlation: float
    max_correlation: float
    min_correlation: float
    correlation_dispersion: float
    correlation_regime: str
    breakdown_detected: bool


@dataclass
class LeadLagRelationship:
    """Container for lead-lag analysis."""
    asset_1: str
    asset_2: str
    optimal_lag: int
    max_correlation: float
    lead_asset: str
    confidence: float


class CorrelationDynamics:
    """
    Advanced dynamic correlation analysis.

    Implements:
    - DCC-GARCH for time-varying correlations
    - Rolling correlation analysis
    - Correlation regime detection
    - Stress correlation estimation
    - Correlation breakdown detection
    - Lead-lag relationships
    - Cross-asset correlation monitoring
    """

    def __init__(self,
                 window: int = 60,
                 min_periods: int = 30,
                 correlation_threshold: float = 0.7,
                 breakdown_threshold: float = 0.3,
                 stress_quantile: float = 0.05):
        """
        Initialize correlation dynamics analyzer.

        Args:
            window: Rolling window for correlation calculation
            min_periods: Minimum periods for valid correlation
            correlation_threshold: Threshold for high correlation
            breakdown_threshold: Threshold for correlation breakdown
            stress_quantile: Quantile for stress scenario definition
        """
        self.window = window
        self.min_periods = min_periods
        self.correlation_threshold = correlation_threshold
        self.breakdown_threshold = breakdown_threshold
        self.stress_quantile = stress_quantile

        # DCC-GARCH components
        self.univariate_models = {}
        self.standardized_residuals = {}
        self.is_fitted = False

        # Correlation tracking
        self.correlation_history = []
        self.regime_history = []

        logger.info(f"Initialized CorrelationDynamics with window={window}")

    def calculate_rolling_correlation(self,
                                     returns_df: pd.DataFrame,
                                     method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate rolling pairwise correlations.

        Args:
            returns_df: DataFrame with returns for multiple assets
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            DataFrame with correlation time series for each pair
        """
        n_assets = len(returns_df.columns)
        pairs = []

        # Generate all pairs
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                asset_i = returns_df.columns[i]
                asset_j = returns_df.columns[j]
                pairs.append((asset_i, asset_j))

        # Calculate rolling correlations
        correlations = pd.DataFrame(index=returns_df.index)

        for asset_i, asset_j in pairs:
            pair_name = f"{asset_i}_{asset_j}"

            if method == 'pearson':
                corr = returns_df[asset_i].rolling(
                    self.window, min_periods=self.min_periods
                ).corr(returns_df[asset_j])
            elif method == 'spearman':
                corr = returns_df[asset_i].rolling(
                    self.window, min_periods=self.min_periods
                ).apply(lambda x: stats.spearmanr(x, returns_df[asset_j].loc[x.index])[0]
                       if len(x) >= self.min_periods else np.nan)
            else:
                corr = returns_df[asset_i].rolling(
                    self.window, min_periods=self.min_periods
                ).corr(returns_df[asset_j])

            correlations[pair_name] = corr

        return correlations

    def calculate_ewma_correlation(self,
                                   returns_df: pd.DataFrame,
                                   lambda_: float = 0.94) -> pd.DataFrame:
        """
        Calculate EWMA (exponentially weighted) correlation matrix.

        Args:
            returns_df: DataFrame with returns for multiple assets
            lambda_: Decay parameter (RiskMetrics uses 0.94)

        Returns:
            Current correlation matrix
        """
        # Calculate EWMA covariance matrix
        ewma_cov = returns_df.ewm(alpha=1-lambda_, min_periods=self.min_periods).cov()

        # Extract most recent covariance matrix
        latest_cov = ewma_cov.iloc[-len(returns_df.columns):]

        # Convert covariance to correlation
        std_matrix = np.sqrt(np.diag(latest_cov))
        correlation_matrix = latest_cov / np.outer(std_matrix, std_matrix)

        return correlation_matrix

    def fit_dcc_garch(self, returns_df: pd.DataFrame) -> 'CorrelationDynamics':
        """
        Fit DCC-GARCH model for dynamic correlations.

        Two-step estimation:
        1. Fit univariate GARCH models for each asset
        2. Fit DCC model on standardized residuals

        Args:
            returns_df: DataFrame with returns for multiple assets

        Returns:
            self
        """
        if not ARCH_AVAILABLE:
            logger.warning("arch package not available")
            return self

        try:
            # Step 1: Fit univariate GARCH models
            for column in returns_df.columns:
                returns = returns_df[column].dropna() * 100  # Convert to percentage

                # Fit GARCH(1,1)
                model = arch_model(returns, vol='GARCH', p=1, q=1)
                result = model.fit(disp='off', show_warning=False)

                self.univariate_models[column] = result

                # Store standardized residuals
                std_residuals = result.resid / result.conditional_volatility
                self.standardized_residuals[column] = std_residuals

                logger.debug(f"Fitted GARCH model for {column}")

            self.is_fitted = True
            logger.success(f"DCC-GARCH fitted for {len(returns_df.columns)} assets")

        except Exception as e:
            logger.error(f"Error fitting DCC-GARCH: {str(e)}")
            self.is_fitted = False

        return self

    def calculate_dcc_correlation(self,
                                  returns_df: pd.DataFrame,
                                  alpha: float = 0.05,
                                  beta: float = 0.90) -> pd.DataFrame:
        """
        Calculate Dynamic Conditional Correlation.

        DCC evolution:
        Q_t = (1 - α - β)Q̄ + α(ε_{t-1}ε'_{t-1}) + βQ_{t-1}

        Args:
            returns_df: DataFrame with returns for multiple assets
            alpha: DCC alpha parameter
            beta: DCC beta parameter

        Returns:
            DataFrame with time-varying correlation matrices
        """
        if not self.is_fitted:
            logger.warning("DCC-GARCH not fitted, fitting now...")
            self.fit_dcc_garch(returns_df)

        if not self.is_fitted:
            return pd.DataFrame()

        try:
            # Align standardized residuals
            residuals_df = pd.DataFrame(self.standardized_residuals)
            residuals_df = residuals_df.dropna()

            n_assets = len(residuals_df.columns)
            n_obs = len(residuals_df)

            # Unconditional correlation (Q̄)
            Q_bar = residuals_df.corr().values

            # Initialize Q_t
            Q_t = Q_bar.copy()

            # Store correlation matrices over time
            correlation_matrices = np.zeros((n_obs, n_assets, n_assets))

            for t in range(n_obs):
                # Current standardized residuals
                eps_t = residuals_df.iloc[t].values.reshape(-1, 1)

                # Update Q_t
                if t > 0:
                    eps_prev = residuals_df.iloc[t-1].values.reshape(-1, 1)
                    Q_t = (1 - alpha - beta) * Q_bar + \
                          alpha * (eps_prev @ eps_prev.T) + \
                          beta * Q_t

                # Convert Q_t to correlation matrix R_t
                Q_diag_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(Q_t)))
                R_t = Q_diag_inv_sqrt @ Q_t @ Q_diag_inv_sqrt

                correlation_matrices[t] = R_t

            # Create DataFrame with most recent correlation matrix
            latest_corr = pd.DataFrame(
                correlation_matrices[-1],
                index=residuals_df.columns,
                columns=residuals_df.columns
            )

            return latest_corr

        except Exception as e:
            logger.error(f"Error calculating DCC correlation: {str(e)}")
            return pd.DataFrame()

    def detect_correlation_regime(self, correlation_series: pd.Series) -> Dict[str, any]:
        """
        Detect correlation regime (low/medium/high).

        Args:
            correlation_series: Time series of correlation values

        Returns:
            Dictionary with regime information
        """
        current_corr = correlation_series.iloc[-1]

        # Historical percentile
        corr_percentile = stats.percentileofscore(
            correlation_series.dropna(), current_corr
        ) / 100

        # Regime classification
        if corr_percentile < 0.33:
            regime = "low"
            regime_id = 0
        elif corr_percentile < 0.67:
            regime = "medium"
            regime_id = 1
        else:
            regime = "high"
            regime_id = 2

        # Trend analysis
        recent_corr = correlation_series.iloc[-20:]
        if len(recent_corr) >= 10:
            x = np.arange(len(recent_corr))
            slope, _, r_value, _, _ = stats.linregress(x, recent_corr.values)
            trend = "increasing" if slope > 0.001 else "decreasing" if slope < -0.001 else "stable"
            trend_strength = abs(r_value)
        else:
            trend = "unknown"
            trend_strength = 0.0

        return {
            'regime': regime,
            'regime_id': regime_id,
            'current_correlation': float(current_corr),
            'percentile': float(corr_percentile),
            'trend': trend,
            'trend_strength': float(trend_strength),
            'volatility': float(correlation_series.std())
        }

    def detect_correlation_breakdown(self,
                                    returns_df: pd.DataFrame,
                                    stress_returns: Optional[pd.Series] = None) -> Dict[str, any]:
        """
        Detect correlation breakdown during stress periods.

        During market stress, correlations often increase toward 1.

        Args:
            returns_df: DataFrame with returns for multiple assets
            stress_returns: Optional series indicating stress periods

        Returns:
            Dictionary with breakdown detection results
        """
        # Calculate rolling correlations
        rolling_corr = self.calculate_rolling_correlation(returns_df)

        # Average pairwise correlation
        avg_corr = rolling_corr.mean(axis=1)

        if stress_returns is not None:
            # Define stress periods (e.g., large negative returns)
            stress_threshold = stress_returns.quantile(self.stress_quantile)
            stress_periods = stress_returns < stress_threshold

            # Correlation during stress vs normal
            stress_corr = avg_corr[stress_periods].mean()
            normal_corr = avg_corr[~stress_periods].mean()

            breakdown_ratio = stress_corr / (normal_corr + 1e-8)
            breakdown_detected = breakdown_ratio > 1.3  # 30% increase

        else:
            # Use recent volatility as stress indicator
            returns_vol = returns_df.iloc[:, 0].rolling(20).std()
            high_vol = returns_vol > returns_vol.quantile(0.9)

            stress_corr = avg_corr[high_vol].mean()
            normal_corr = avg_corr[~high_vol].mean()

            breakdown_ratio = stress_corr / (normal_corr + 1e-8)
            breakdown_detected = breakdown_ratio > 1.3

        # Current vs historical correlation
        current_avg_corr = avg_corr.iloc[-1]
        historical_avg = avg_corr.mean()

        return {
            'breakdown_detected': bool(breakdown_detected),
            'current_avg_correlation': float(current_avg_corr),
            'historical_avg_correlation': float(historical_avg),
            'stress_correlation': float(stress_corr),
            'normal_correlation': float(normal_corr),
            'breakdown_ratio': float(breakdown_ratio),
            'correlation_change': float(current_avg_corr - historical_avg)
        }

    def calculate_stress_correlation(self,
                                    returns_df: pd.DataFrame,
                                    quantile: Optional[float] = None) -> pd.DataFrame:
        """
        Calculate correlation matrix during stress scenarios.

        Args:
            returns_df: DataFrame with returns for multiple assets
            quantile: Quantile for stress definition (default: self.stress_quantile)

        Returns:
            Correlation matrix during stress periods
        """
        quantile = quantile or self.stress_quantile

        # Define stress as extreme negative returns on first asset (or portfolio)
        stress_indicator = returns_df.iloc[:, 0]
        stress_threshold = stress_indicator.quantile(quantile)

        # Filter to stress periods
        stress_returns = returns_df[stress_indicator < stress_threshold]

        if len(stress_returns) < 10:
            logger.warning("Insufficient stress observations")
            return returns_df.corr()

        # Calculate correlation during stress
        stress_corr_matrix = stress_returns.corr()

        return stress_corr_matrix

    def analyze_lead_lag_relationships(self,
                                      returns_df: pd.DataFrame,
                                      max_lag: int = 10) -> List[LeadLagRelationship]:
        """
        Analyze lead-lag relationships between assets.

        Args:
            returns_df: DataFrame with returns for multiple assets
            max_lag: Maximum lag to consider

        Returns:
            List of LeadLagRelationship objects
        """
        relationships = []
        assets = returns_df.columns.tolist()

        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                asset_1 = assets[i]
                asset_2 = assets[j]

                # Calculate cross-correlation
                series_1 = returns_df[asset_1].dropna()
                series_2 = returns_df[asset_2].dropna()

                # Align series
                aligned = pd.concat([series_1, series_2], axis=1).dropna()

                if len(aligned) < 50:
                    continue

                # Cross-correlation at different lags
                correlations = []
                for lag in range(-max_lag, max_lag + 1):
                    if lag < 0:
                        # asset_2 leads asset_1
                        corr = aligned.iloc[:lag, 0].corr(aligned.iloc[-lag:, 1])
                    elif lag > 0:
                        # asset_1 leads asset_2
                        corr = aligned.iloc[lag:, 0].corr(aligned.iloc[:-lag, 1])
                    else:
                        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

                    correlations.append(corr)

                # Find maximum correlation and corresponding lag
                max_corr_idx = np.argmax(np.abs(correlations))
                optimal_lag = max_corr_idx - max_lag
                max_correlation = correlations[max_corr_idx]

                # Determine lead asset
                if optimal_lag < 0:
                    lead_asset = asset_2
                elif optimal_lag > 0:
                    lead_asset = asset_1
                else:
                    lead_asset = "simultaneous"

                # Statistical significance
                n = len(aligned)
                t_stat = max_correlation * np.sqrt((n - 2) / (1 - max_correlation ** 2))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                confidence = 1 - p_value

                relationship = LeadLagRelationship(
                    asset_1=asset_1,
                    asset_2=asset_2,
                    optimal_lag=int(optimal_lag),
                    max_correlation=float(max_correlation),
                    lead_asset=lead_asset,
                    confidence=float(confidence)
                )

                relationships.append(relationship)

        # Sort by correlation strength
        relationships.sort(key=lambda x: abs(x.max_correlation), reverse=True)

        return relationships

    def calculate_comprehensive_metrics(self, returns_df: pd.DataFrame) -> CorrelationMetrics:
        """
        Calculate comprehensive correlation metrics.

        Args:
            returns_df: DataFrame with returns for multiple assets

        Returns:
            CorrelationMetrics object
        """
        # Current correlation matrix
        correlation_matrix = returns_df.corr()

        # Extract upper triangle (unique pairs)
        mask = np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)
        correlations = correlation_matrix.where(mask).stack()

        # Metrics
        avg_corr = correlations.mean()
        max_corr = correlations.max()
        min_corr = correlations.min()
        corr_dispersion = correlations.std()

        # Regime detection (using average correlation)
        rolling_corr = self.calculate_rolling_correlation(returns_df)
        avg_rolling_corr = rolling_corr.mean(axis=1)

        if len(avg_rolling_corr.dropna()) > 0:
            regime_info = self.detect_correlation_regime(avg_rolling_corr)
            regime = regime_info['regime']
        else:
            regime = "unknown"

        # Breakdown detection
        breakdown_info = self.detect_correlation_breakdown(returns_df)
        breakdown_detected = breakdown_info['breakdown_detected']

        return CorrelationMetrics(
            correlation_matrix=correlation_matrix,
            average_correlation=float(avg_corr),
            max_correlation=float(max_corr),
            min_correlation=float(min_corr),
            correlation_dispersion=float(corr_dispersion),
            correlation_regime=regime,
            breakdown_detected=breakdown_detected
        )

    def generate_correlation_report(self, returns_df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive correlation analysis report.

        Args:
            returns_df: DataFrame with returns for multiple assets

        Returns:
            Dictionary with complete correlation analysis
        """
        # Comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(returns_df)

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'average_correlation': metrics.average_correlation,
                'max_correlation': metrics.max_correlation,
                'min_correlation': metrics.min_correlation,
                'correlation_dispersion': metrics.correlation_dispersion,
                'correlation_regime': metrics.correlation_regime,
                'breakdown_detected': metrics.breakdown_detected
            },
            'correlation_matrix': metrics.correlation_matrix.to_dict()
        }

        # Breakdown analysis
        breakdown_info = self.detect_correlation_breakdown(returns_df)
        report['breakdown_analysis'] = breakdown_info

        # Stress correlation
        stress_corr = self.calculate_stress_correlation(returns_df)
        report['stress_correlation'] = {
            'average': float(stress_corr.where(
                np.triu(np.ones_like(stress_corr), k=1).astype(bool)
            ).stack().mean())
        }

        # Lead-lag relationships
        relationships = self.analyze_lead_lag_relationships(returns_df, max_lag=5)
        report['lead_lag_relationships'] = [
            {
                'asset_1': r.asset_1,
                'asset_2': r.asset_2,
                'optimal_lag': r.optimal_lag,
                'correlation': r.max_correlation,
                'lead_asset': r.lead_asset,
                'confidence': r.confidence
            }
            for r in relationships[:5]  # Top 5 relationships
        ]

        # DCC-GARCH if fitted
        if self.is_fitted:
            dcc_corr = self.calculate_dcc_correlation(returns_df)
            if not dcc_corr.empty:
                report['dcc_correlation'] = dcc_corr.to_dict()

        return report

    def monitor_correlation_limits(self,
                                   returns_df: pd.DataFrame,
                                   max_avg_correlation: float = 0.7) -> Dict[str, any]:
        """
        Monitor correlation limits for risk management.

        Args:
            returns_df: DataFrame with returns for multiple assets
            max_avg_correlation: Maximum acceptable average correlation

        Returns:
            Dictionary with limit monitoring results
        """
        metrics = self.calculate_comprehensive_metrics(returns_df)

        limit_breached = metrics.average_correlation > max_avg_correlation

        # Find pairs exceeding threshold
        correlation_matrix = metrics.correlation_matrix
        mask = np.triu(np.ones_like(correlation_matrix), k=1).astype(bool)
        high_corr_pairs = []

        for i, row in enumerate(correlation_matrix.index):
            for j, col in enumerate(correlation_matrix.columns):
                if mask[i, j] and abs(correlation_matrix.iloc[i, j]) > max_avg_correlation:
                    high_corr_pairs.append({
                        'asset_1': row,
                        'asset_2': col,
                        'correlation': float(correlation_matrix.iloc[i, j])
                    })

        return {
            'limit_breached': limit_breached,
            'average_correlation': metrics.average_correlation,
            'max_allowed_correlation': max_avg_correlation,
            'high_correlation_pairs': high_corr_pairs,
            'n_breaches': len(high_corr_pairs),
            'recommendation': 'Reduce positions' if limit_breached else 'OK'
        }
