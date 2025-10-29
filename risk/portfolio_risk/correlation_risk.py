"""
Correlation Risk Management Module

This module implements correlation risk monitoring and management for portfolio risk,
including correlation matrix analysis, regime detection, and diversification metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.linalg import eigvalsh
import logging

logger = logging.getLogger(__name__)


@dataclass
class CorrelationAlert:
    """Alert for correlation risk events"""
    timestamp: datetime
    alert_type: str  # 'breakdown', 'regime_change', 'concentration', 'spike'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    affected_assets: List[str]
    correlation_value: float
    threshold: float
    details: Dict = field(default_factory=dict)


@dataclass
class CorrelationMetrics:
    """Comprehensive correlation risk metrics"""
    timestamp: datetime
    avg_correlation: float
    median_correlation: float
    max_correlation: float
    min_correlation: float
    correlation_matrix: pd.DataFrame
    eigenvalues: np.ndarray
    effective_bets: float
    correlation_penalty: float
    regime: str  # 'low', 'normal', 'high', 'crisis'
    regime_probability: float
    cross_asset_correlations: Dict[str, float]
    correlation_breakdown_detected: bool
    diversification_ratio: float
    alerts: List[CorrelationAlert]


@dataclass
class CorrelationRegime:
    """Correlation regime characteristics"""
    regime_name: str
    avg_correlation: float
    volatility_regime: str
    duration: int  # days
    stability: float  # 0-1 score
    transition_probability: Dict[str, float]


class CorrelationRisk:
    """
    Correlation Risk Monitor

    Monitors and analyzes correlation structure of portfolio positions,
    detecting correlation breakdowns, regime changes, and concentration risks.

    Features:
    - Rolling correlation matrix calculation
    - Correlation breakdown detection
    - Correlation regime monitoring
    - Effective number of bets calculation
    - Correlation penalty calculation
    - Cross-asset correlation tracking

    Attributes:
        window (int): Rolling window for correlation calculation
        regime_lookback (int): Lookback period for regime identification
        breakdown_threshold (float): Threshold for correlation breakdown
        high_corr_threshold (float): Threshold for high correlation alert
    """

    def __init__(
        self,
        window: int = 60,
        regime_lookback: int = 252,
        breakdown_threshold: float = 0.3,
        high_corr_threshold: float = 0.7,
        crisis_threshold: float = 0.85
    ):
        """
        Initialize Correlation Risk Monitor

        Args:
            window: Rolling window for correlation calculation (default: 60 days)
            regime_lookback: Lookback period for regime analysis (default: 252 days)
            breakdown_threshold: Threshold for correlation breakdown detection
            high_corr_threshold: Threshold for high correlation warning
            crisis_threshold: Threshold for crisis correlation regime
        """
        self.window = window
        self.regime_lookback = regime_lookback
        self.breakdown_threshold = breakdown_threshold
        self.high_corr_threshold = high_corr_threshold
        self.crisis_threshold = crisis_threshold

        # Store historical correlations for regime detection
        self.correlation_history: List[Tuple[datetime, float]] = []
        self.regime_history: List[Tuple[datetime, str]] = []
        self.last_correlation_matrix: Optional[pd.DataFrame] = None

        logger.info(f"CorrelationRisk initialized with window={window}")

    def calculate_correlation_matrix(
        self,
        returns: pd.DataFrame,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for portfolio returns

        Args:
            returns: DataFrame of asset returns (assets as columns)
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Correlation matrix as DataFrame
        """
        if len(returns) < self.window:
            logger.warning(f"Insufficient data for correlation calculation: {len(returns)} < {self.window}")
            return returns.corr(method=method)

        # Use rolling window
        rolling_returns = returns.tail(self.window)

        if method == 'spearman':
            # Use scipy for spearman correlation
            corr_matrix = pd.DataFrame(
                spearmanr(rolling_returns)[0],
                index=rolling_returns.columns,
                columns=rolling_returns.columns
            )
        else:
            corr_matrix = rolling_returns.corr(method=method)

        self.last_correlation_matrix = corr_matrix
        return corr_matrix

    def calculate_rolling_correlation(
        self,
        returns: pd.DataFrame,
        asset1: str,
        asset2: str
    ) -> pd.Series:
        """
        Calculate rolling correlation between two assets

        Args:
            returns: DataFrame of asset returns
            asset1: First asset identifier
            asset2: Second asset identifier

        Returns:
            Series of rolling correlations
        """
        if asset1 not in returns.columns or asset2 not in returns.columns:
            raise ValueError(f"Assets {asset1} or {asset2} not found in returns")

        rolling_corr = returns[asset1].rolling(window=self.window).corr(returns[asset2])
        return rolling_corr

    def detect_correlation_breakdown(
        self,
        current_corr: pd.DataFrame,
        historical_corr: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> Tuple[bool, List[Tuple[str, str, float]]]:
        """
        Detect significant correlation breakdown events

        Args:
            current_corr: Current correlation matrix
            historical_corr: Historical average correlation matrix
            threshold: Breakdown threshold (uses class default if None)

        Returns:
            Tuple of (breakdown_detected, list of breakdown pairs with delta)
        """
        threshold = threshold or self.breakdown_threshold
        breakdowns = []

        # Get upper triangle indices (exclude diagonal)
        assets = current_corr.columns
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets[i+1:], i+1):
                current_val = current_corr.loc[asset1, asset2]
                historical_val = historical_corr.loc[asset1, asset2]

                # Check for significant decrease
                delta = abs(current_val - historical_val)
                if delta > threshold and current_val < historical_val:
                    breakdowns.append((asset1, asset2, delta))

        breakdown_detected = len(breakdowns) > 0
        return breakdown_detected, breakdowns

    def identify_correlation_regime(
        self,
        avg_correlation: float
    ) -> Tuple[str, float]:
        """
        Identify current correlation regime

        Args:
            avg_correlation: Average pairwise correlation

        Returns:
            Tuple of (regime_name, confidence_score)
        """
        # Define regime thresholds
        if avg_correlation >= self.crisis_threshold:
            regime = 'crisis'
            confidence = min(1.0, (avg_correlation - self.crisis_threshold) / 0.1 + 0.7)
        elif avg_correlation >= self.high_corr_threshold:
            regime = 'high'
            confidence = min(1.0, (avg_correlation - self.high_corr_threshold) / 0.15 + 0.6)
        elif avg_correlation >= 0.4:
            regime = 'normal'
            confidence = 0.8
        else:
            regime = 'low'
            confidence = min(1.0, (0.4 - avg_correlation) / 0.3 + 0.5)

        return regime, confidence

    def calculate_effective_bets(
        self,
        correlation_matrix: pd.DataFrame,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate effective number of bets (diversification measure)

        Uses eigenvalue decomposition to estimate true number of independent bets.

        Args:
            correlation_matrix: Asset correlation matrix
            weights: Portfolio weights (equal weight if None)

        Returns:
            Effective number of bets (1 to N assets)
        """
        n_assets = len(correlation_matrix)

        if weights is None:
            weights = np.ones(n_assets) / n_assets

        # Calculate eigenvalues of correlation matrix
        eigenvalues = eigvalsh(correlation_matrix.values)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative

        # Normalize eigenvalues
        eigenvalues = eigenvalues / eigenvalues.sum()

        # Calculate effective number of bets using entropy
        # ENB = exp(-sum(lambda_i * log(lambda_i)))
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Avoid log(0)
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        effective_bets = np.exp(entropy)

        return float(effective_bets)

    def calculate_correlation_penalty(
        self,
        correlation_matrix: pd.DataFrame,
        weights: np.ndarray
    ) -> float:
        """
        Calculate correlation penalty on portfolio diversification

        Args:
            correlation_matrix: Asset correlation matrix
            weights: Portfolio weights

        Returns:
            Correlation penalty (0 = no correlation, 1 = perfect correlation)
        """
        n_assets = len(weights)

        # Calculate weighted average correlation
        total_weight = 0.0
        weighted_corr = 0.0

        for i in range(n_assets):
            for j in range(i+1, n_assets):
                weight_product = weights[i] * weights[j]
                correlation = correlation_matrix.iloc[i, j]
                weighted_corr += weight_product * abs(correlation)
                total_weight += weight_product

        penalty = weighted_corr / total_weight if total_weight > 0 else 0.0
        return float(penalty)

    def calculate_cross_asset_correlations(
        self,
        returns: pd.DataFrame,
        asset_classes: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate correlations across different asset classes

        Args:
            returns: DataFrame of asset returns
            asset_classes: Dict mapping asset symbols to asset classes

        Returns:
            Dict of cross-asset class average correlations
        """
        corr_matrix = self.calculate_correlation_matrix(returns)
        cross_correlations = {}

        # Get unique asset classes
        unique_classes = list(set(asset_classes.values()))

        for i, class1 in enumerate(unique_classes):
            for class2 in unique_classes[i+1:]:
                # Get assets in each class
                assets1 = [a for a, c in asset_classes.items() if c == class1 and a in returns.columns]
                assets2 = [a for a, c in asset_classes.items() if c == class2 and a in returns.columns]

                if not assets1 or not assets2:
                    continue

                # Calculate average correlation between classes
                correlations = []
                for a1 in assets1:
                    for a2 in assets2:
                        if a1 in corr_matrix.index and a2 in corr_matrix.columns:
                            correlations.append(corr_matrix.loc[a1, a2])

                if correlations:
                    avg_corr = np.mean(correlations)
                    cross_correlations[f"{class1}-{class2}"] = float(avg_corr)

        return cross_correlations

    def calculate_diversification_ratio(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray
    ) -> float:
        """
        Calculate portfolio diversification ratio

        DR = (weighted sum of volatilities) / (portfolio volatility)
        DR > 1 indicates diversification benefit

        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights

        Returns:
            Diversification ratio
        """
        # Calculate individual volatilities
        volatilities = returns.std()

        # Weighted sum of volatilities
        weighted_vol_sum = np.sum(weights * volatilities.values)

        # Portfolio volatility
        cov_matrix = returns.cov()
        portfolio_var = np.dot(weights, np.dot(cov_matrix.values, weights))
        portfolio_vol = np.sqrt(portfolio_var)

        if portfolio_vol < 1e-10:
            return 1.0

        diversification_ratio = weighted_vol_sum / portfolio_vol
        return float(diversification_ratio)

    def generate_correlation_alerts(
        self,
        metrics: CorrelationMetrics,
        historical_avg: float
    ) -> List[CorrelationAlert]:
        """
        Generate alerts based on correlation risk metrics

        Args:
            metrics: Current correlation metrics
            historical_avg: Historical average correlation

        Returns:
            List of correlation alerts
        """
        alerts = []
        timestamp = metrics.timestamp

        # High correlation alert
        if metrics.avg_correlation >= self.crisis_threshold:
            alerts.append(CorrelationAlert(
                timestamp=timestamp,
                alert_type='crisis',
                severity='critical',
                message=f"Crisis correlation regime detected: {metrics.avg_correlation:.3f}",
                affected_assets=list(metrics.correlation_matrix.columns),
                correlation_value=metrics.avg_correlation,
                threshold=self.crisis_threshold,
                details={'regime': metrics.regime, 'effective_bets': metrics.effective_bets}
            ))
        elif metrics.avg_correlation >= self.high_corr_threshold:
            alerts.append(CorrelationAlert(
                timestamp=timestamp,
                alert_type='high_correlation',
                severity='high',
                message=f"High correlation detected: {metrics.avg_correlation:.3f}",
                affected_assets=list(metrics.correlation_matrix.columns),
                correlation_value=metrics.avg_correlation,
                threshold=self.high_corr_threshold,
                details={'regime': metrics.regime}
            ))

        # Correlation breakdown alert
        if metrics.correlation_breakdown_detected:
            alerts.append(CorrelationAlert(
                timestamp=timestamp,
                alert_type='breakdown',
                severity='medium',
                message="Correlation breakdown detected in portfolio",
                affected_assets=list(metrics.correlation_matrix.columns),
                correlation_value=metrics.avg_correlation,
                threshold=self.breakdown_threshold,
                details={'historical_avg': historical_avg}
            ))

        # Low effective bets alert
        n_assets = len(metrics.correlation_matrix)
        if metrics.effective_bets < n_assets * 0.3:
            alerts.append(CorrelationAlert(
                timestamp=timestamp,
                alert_type='concentration',
                severity='high',
                message=f"Low effective bets: {metrics.effective_bets:.2f} (assets: {n_assets})",
                affected_assets=list(metrics.correlation_matrix.columns),
                correlation_value=metrics.effective_bets,
                threshold=n_assets * 0.3,
                details={'penalty': metrics.correlation_penalty}
            ))

        # Regime change alert
        if len(self.regime_history) > 0:
            last_regime = self.regime_history[-1][1]
            if last_regime != metrics.regime:
                alerts.append(CorrelationAlert(
                    timestamp=timestamp,
                    alert_type='regime_change',
                    severity='medium',
                    message=f"Correlation regime changed: {last_regime} -> {metrics.regime}",
                    affected_assets=list(metrics.correlation_matrix.columns),
                    correlation_value=metrics.avg_correlation,
                    threshold=0.0,
                    details={'previous_regime': last_regime, 'confidence': metrics.regime_probability}
                ))

        return alerts

    def calculate_correlation_risk(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        asset_classes: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> CorrelationMetrics:
        """
        Calculate comprehensive correlation risk metrics

        Args:
            returns: DataFrame of asset returns
            weights: Portfolio weights
            asset_classes: Optional dict mapping assets to asset classes
            timestamp: Current timestamp (uses now if None)

        Returns:
            CorrelationMetrics with comprehensive analysis
        """
        timestamp = timestamp or datetime.now()

        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(returns)

        # Calculate summary statistics
        # Get upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        correlations = corr_matrix.values[mask]

        avg_correlation = float(np.mean(correlations))
        median_correlation = float(np.median(correlations))
        max_correlation = float(np.max(correlations))
        min_correlation = float(np.min(correlations))

        # Store in history
        self.correlation_history.append((timestamp, avg_correlation))
        if len(self.correlation_history) > self.regime_lookback:
            self.correlation_history.pop(0)

        # Calculate eigenvalues
        eigenvalues = eigvalsh(corr_matrix.values)

        # Calculate effective bets
        effective_bets = self.calculate_effective_bets(corr_matrix, weights)

        # Calculate correlation penalty
        correlation_penalty = self.calculate_correlation_penalty(corr_matrix, weights)

        # Identify regime
        regime, regime_prob = self.identify_correlation_regime(avg_correlation)
        self.regime_history.append((timestamp, regime))
        if len(self.regime_history) > 100:
            self.regime_history.pop(0)

        # Calculate cross-asset correlations
        cross_asset_corr = {}
        if asset_classes is not None:
            cross_asset_corr = self.calculate_cross_asset_correlations(returns, asset_classes)

        # Detect correlation breakdown
        breakdown_detected = False
        if len(self.correlation_history) >= self.window:
            historical_avg_corr = np.mean([c for _, c in self.correlation_history[:-1]])
            if abs(avg_correlation - historical_avg_corr) > self.breakdown_threshold:
                breakdown_detected = True

        # Calculate diversification ratio
        diversification_ratio = self.calculate_diversification_ratio(returns, weights)

        # Create metrics object
        metrics = CorrelationMetrics(
            timestamp=timestamp,
            avg_correlation=avg_correlation,
            median_correlation=median_correlation,
            max_correlation=max_correlation,
            min_correlation=min_correlation,
            correlation_matrix=corr_matrix,
            eigenvalues=eigenvalues,
            effective_bets=effective_bets,
            correlation_penalty=correlation_penalty,
            regime=regime,
            regime_probability=regime_prob,
            cross_asset_correlations=cross_asset_corr,
            correlation_breakdown_detected=breakdown_detected,
            diversification_ratio=diversification_ratio,
            alerts=[]
        )

        # Generate alerts
        historical_avg = np.mean([c for _, c in self.correlation_history]) if self.correlation_history else avg_correlation
        metrics.alerts = self.generate_correlation_alerts(metrics, historical_avg)

        logger.info(f"Correlation risk calculated: avg={avg_correlation:.3f}, regime={regime}, eff_bets={effective_bets:.2f}")

        return metrics

    def get_correlation_report(self, metrics: CorrelationMetrics) -> str:
        """
        Generate human-readable correlation risk report

        Args:
            metrics: Correlation metrics

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("CORRELATION RISK REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {metrics.timestamp}")
        report.append("")

        report.append("Summary Statistics:")
        report.append(f"  Average Correlation: {metrics.avg_correlation:.3f}")
        report.append(f"  Median Correlation:  {metrics.median_correlation:.3f}")
        report.append(f"  Max Correlation:     {metrics.max_correlation:.3f}")
        report.append(f"  Min Correlation:     {metrics.min_correlation:.3f}")
        report.append("")

        report.append("Diversification Metrics:")
        report.append(f"  Effective Number of Bets: {metrics.effective_bets:.2f}")
        report.append(f"  Correlation Penalty:      {metrics.correlation_penalty:.3f}")
        report.append(f"  Diversification Ratio:    {metrics.diversification_ratio:.3f}")
        report.append("")

        report.append(f"Correlation Regime: {metrics.regime.upper()} (confidence: {metrics.regime_probability:.2f})")
        report.append(f"Breakdown Detected: {'YES' if metrics.correlation_breakdown_detected else 'NO'}")
        report.append("")

        if metrics.cross_asset_correlations:
            report.append("Cross-Asset Correlations:")
            for pair, corr in metrics.cross_asset_correlations.items():
                report.append(f"  {pair}: {corr:.3f}")
            report.append("")

        if metrics.alerts:
            report.append(f"ALERTS ({len(metrics.alerts)}):")
            for alert in metrics.alerts:
                report.append(f"  [{alert.severity.upper()}] {alert.alert_type}: {alert.message}")
            report.append("")

        report.append("=" * 60)
        return "\n".join(report)
