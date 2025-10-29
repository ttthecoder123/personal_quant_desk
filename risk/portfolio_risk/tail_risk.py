"""
Tail Risk Management Module

This module implements tail risk monitoring and management using Extreme Value Theory (EVT),
fat tail indicators, and tail risk scenario analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


@dataclass
class TailRiskAlert:
    """Alert for tail risk events"""
    timestamp: datetime
    alert_type: str  # 'extreme_loss', 'fat_tail', 'jump_risk', 'tail_hedge', 'kurtosis'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    asset: Optional[str]
    metric_value: float
    threshold: float
    details: Dict = field(default_factory=dict)


@dataclass
class EVTMetrics:
    """Extreme Value Theory metrics"""
    threshold: float
    n_exceedances: int
    shape_parameter: float  # xi (tail index)
    scale_parameter: float  # sigma
    location_parameter: float  # mu
    tail_index: float
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    extreme_quantiles: Dict[float, float]


@dataclass
class TailRiskMetrics:
    """Comprehensive tail risk metrics"""
    timestamp: datetime
    kurtosis: float
    excess_kurtosis: float
    skewness: float
    jarque_bera_stat: float
    jarque_bera_pvalue: float
    is_normal: bool
    evt_metrics: Optional[EVTMetrics]
    tail_var_95: float
    tail_var_99: float
    tail_cvar_95: float
    tail_cvar_99: float
    max_drawdown_tail: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    tail_ratio: float  # Ratio of right tail to left tail
    downside_deviation: float
    jump_probability: float
    largest_losses: List[float]
    tail_correlation: float
    fat_tail_indicator: float
    hedge_recommendations: List[str]
    scenario_losses: Dict[str, float]
    alerts: List[TailRiskAlert]


@dataclass
class TailScenario:
    """Tail risk scenario definition"""
    name: str
    description: str
    probability: float
    severity: float  # Expected loss as % of portfolio
    conditions: Dict[str, float]
    hedge_strategies: List[str]


class TailRisk:
    """
    Tail Risk Monitor

    Monitors and analyzes tail risk using Extreme Value Theory (EVT),
    fat tail indicators, and tail risk scenario analysis.

    Features:
    - Extreme Value Theory (EVT) implementation
    - Tail risk metrics calculation
    - Fat tail indicators monitoring
    - Tail hedging recommendations
    - Kurtosis and skewness tracking
    - Tail risk scenario generation
    - Jump risk monitoring

    Attributes:
        evt_threshold_pct (float): Percentile for EVT threshold
        kurtosis_threshold (float): Threshold for excess kurtosis alert
        jump_threshold (float): Threshold for jump detection
        tail_var_confidence (List[float]): Confidence levels for tail VaR
    """

    def __init__(
        self,
        evt_threshold_pct: float = 0.95,
        kurtosis_threshold: float = 3.0,
        jump_threshold: float = 3.0,
        tail_var_confidence: Optional[List[float]] = None,
        min_exceedances: int = 30
    ):
        """
        Initialize Tail Risk Monitor

        Args:
            evt_threshold_pct: Percentile for EVT threshold (e.g., 0.95 = 95th percentile)
            kurtosis_threshold: Threshold for excess kurtosis warning
            jump_threshold: Number of standard deviations for jump detection
            tail_var_confidence: Confidence levels for tail VaR calculation
            min_exceedances: Minimum number of exceedances for EVT fitting
        """
        self.evt_threshold_pct = evt_threshold_pct
        self.kurtosis_threshold = kurtosis_threshold
        self.jump_threshold = jump_threshold
        self.tail_var_confidence = tail_var_confidence or [0.95, 0.99, 0.999]
        self.min_exceedances = min_exceedances

        # Historical data storage
        self.tail_metrics_history: List[Tuple[datetime, float]] = []
        self.extreme_events: List[Tuple[datetime, float]] = []

        logger.info(f"TailRisk initialized with evt_threshold={evt_threshold_pct}")

    def fit_gpd(
        self,
        exceedances: np.ndarray,
        initial_params: Optional[Tuple[float, float]] = None
    ) -> Tuple[float, float]:
        """
        Fit Generalized Pareto Distribution (GPD) to exceedances

        GPD is used in Peaks Over Threshold (POT) method of EVT

        Args:
            exceedances: Array of threshold exceedances
            initial_params: Optional initial (shape, scale) parameters

        Returns:
            Tuple of (shape, scale) parameters (xi, sigma)
        """
        if len(exceedances) < 10:
            logger.warning(f"Insufficient exceedances for GPD fitting: {len(exceedances)}")
            return 0.1, np.std(exceedances)

        # Method of Maximum Likelihood Estimation
        def negative_log_likelihood(params):
            xi, sigma = params
            if sigma <= 0:
                return np.inf

            n = len(exceedances)

            if abs(xi) < 1e-6:
                # xi ≈ 0: Exponential distribution
                return n * np.log(sigma) + np.sum(exceedances) / sigma
            else:
                # General GPD
                term = 1 + xi * exceedances / sigma
                if np.any(term <= 0):
                    return np.inf
                return n * np.log(sigma) + (1 + 1/xi) * np.sum(np.log(term))

        # Initial parameters
        if initial_params is None:
            initial_xi = 0.1
            initial_sigma = np.std(exceedances)
        else:
            initial_xi, initial_sigma = initial_params

        # Optimize
        result = minimize(
            negative_log_likelihood,
            x0=[initial_xi, initial_sigma],
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5), (0.001, None)]
        )

        if result.success:
            xi, sigma = result.x
        else:
            logger.warning("GPD optimization failed, using moment estimators")
            # Fall back to moment estimators
            mean_exc = np.mean(exceedances)
            var_exc = np.var(exceedances)
            xi = 0.5 * (1 - mean_exc**2 / var_exc)
            sigma = 0.5 * mean_exc * (1 + xi)

        return float(xi), float(sigma)

    def calculate_evt_metrics(
        self,
        returns: np.ndarray,
        threshold_pct: Optional[float] = None
    ) -> EVTMetrics:
        """
        Calculate Extreme Value Theory metrics using Peaks Over Threshold (POT)

        Args:
            returns: Array of portfolio returns
            threshold_pct: Percentile for threshold (uses class default if None)

        Returns:
            EVTMetrics with fitted parameters and quantiles
        """
        threshold_pct = threshold_pct or self.evt_threshold_pct

        # Use negative returns for loss distribution
        losses = -returns

        # Determine threshold
        threshold = np.quantile(losses, threshold_pct)

        # Extract exceedances
        exceedances = losses[losses > threshold] - threshold

        if len(exceedances) < self.min_exceedances:
            logger.warning(
                f"Insufficient exceedances: {len(exceedances)} < {self.min_exceedances}"
            )

        # Fit GPD to exceedances
        xi, sigma = self.fit_gpd(exceedances)

        # Calculate tail index
        if xi > 0:
            tail_index = 1.0 / xi
        else:
            tail_index = float('inf')

        # Calculate VaR and CVaR using GPD
        n_total = len(losses)
        n_exceedances = len(exceedances)
        zeta = n_exceedances / n_total  # Proportion exceeding threshold

        extreme_quantiles = {}
        var_dict = {}
        cvar_dict = {}

        for confidence in self.tail_var_confidence:
            # VaR calculation
            if confidence <= threshold_pct:
                var = np.quantile(losses, confidence)
            else:
                # Use GPD for extreme quantiles
                p = (confidence - threshold_pct) / (1 - threshold_pct)
                if abs(xi) < 1e-6:
                    var = threshold + sigma * (-np.log(1 - p))
                else:
                    var = threshold + (sigma / xi) * ((1 - p) ** (-xi) - 1)

            var_dict[confidence] = float(var)
            extreme_quantiles[confidence] = float(var)

            # CVaR (Expected Shortfall) calculation
            if abs(xi) < 1e-6:
                cvar = var + sigma
            elif xi < 1:
                cvar = var / (1 - xi) + (sigma - xi * threshold) / (1 - xi)
            else:
                cvar = float('inf')

            cvar_dict[confidence] = float(cvar)

        # Create EVT metrics
        evt_metrics = EVTMetrics(
            threshold=float(threshold),
            n_exceedances=int(n_exceedances),
            shape_parameter=xi,
            scale_parameter=sigma,
            location_parameter=float(threshold),
            tail_index=tail_index,
            var_95=var_dict.get(0.95, 0.0),
            var_99=var_dict.get(0.99, 0.0),
            cvar_95=cvar_dict.get(0.95, 0.0),
            cvar_99=cvar_dict.get(0.99, 0.0),
            extreme_quantiles=extreme_quantiles
        )

        return evt_metrics

    def calculate_higher_moments(
        self,
        returns: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate higher moments of return distribution

        Args:
            returns: Array of returns

        Returns:
            Tuple of (skewness, kurtosis, excess_kurtosis)
        """
        skewness = float(stats.skew(returns))
        kurtosis = float(stats.kurtosis(returns, fisher=False))  # Pearson's definition
        excess_kurtosis = kurtosis - 3.0  # Excess over normal distribution

        return skewness, kurtosis, excess_kurtosis

    def test_normality(
        self,
        returns: np.ndarray
    ) -> Tuple[float, float, bool]:
        """
        Test if returns follow normal distribution using Jarque-Bera test

        Args:
            returns: Array of returns

        Returns:
            Tuple of (test_statistic, p_value, is_normal)
        """
        jb_stat, jb_pvalue = stats.jarque_bera(returns)

        # Reject normality if p-value < 0.05
        is_normal = jb_pvalue > 0.05

        return float(jb_stat), float(jb_pvalue), is_normal

    def calculate_downside_deviation(
        self,
        returns: np.ndarray,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate downside deviation (semi-deviation)

        Args:
            returns: Array of returns
            threshold: Return threshold (default: 0)

        Returns:
            Downside deviation
        """
        downside_returns = returns[returns < threshold]

        if len(downside_returns) == 0:
            return 0.0

        downside_deviation = np.sqrt(np.mean((downside_returns - threshold) ** 2))

        return float(downside_deviation)

    def calculate_tail_ratio(
        self,
        returns: np.ndarray,
        percentile: float = 0.05
    ) -> float:
        """
        Calculate tail ratio (right tail to left tail)

        Measures asymmetry in tail behavior
        Ratio > 1: right tail fatter (positive skew)
        Ratio < 1: left tail fatter (negative skew)

        Args:
            returns: Array of returns
            percentile: Percentile for tail definition (e.g., 0.05 = 5%)

        Returns:
            Tail ratio
        """
        right_tail = np.abs(np.quantile(returns, 1 - percentile))
        left_tail = np.abs(np.quantile(returns, percentile))

        if left_tail == 0:
            return float('inf') if right_tail > 0 else 1.0

        tail_ratio = right_tail / left_tail

        return float(tail_ratio)

    def detect_jumps(
        self,
        returns: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[List[int], float]:
        """
        Detect jump events in returns

        Args:
            returns: Array of returns
            threshold: Jump threshold in standard deviations

        Returns:
            Tuple of (list of jump indices, jump probability)
        """
        threshold = threshold or self.jump_threshold

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Identify jumps
        jump_mask = np.abs(returns - mean_return) > threshold * std_return
        jump_indices = np.where(jump_mask)[0].tolist()

        # Calculate jump probability
        jump_probability = float(np.sum(jump_mask) / len(returns))

        return jump_indices, jump_probability

    def calculate_fat_tail_indicator(
        self,
        returns: np.ndarray,
        evt_metrics: Optional[EVTMetrics] = None
    ) -> float:
        """
        Calculate fat tail indicator

        Combines multiple metrics to assess tail fatness
        Range: [0, 1] where 1 = extremely fat tails

        Args:
            returns: Array of returns
            evt_metrics: Optional pre-calculated EVT metrics

        Returns:
            Fat tail indicator (0-1)
        """
        indicators = []

        # Excess kurtosis indicator
        _, _, excess_kurtosis = self.calculate_higher_moments(returns)
        kurtosis_indicator = min(1.0, max(0.0, excess_kurtosis / 10.0))
        indicators.append(kurtosis_indicator)

        # EVT tail index indicator
        if evt_metrics is not None:
            if evt_metrics.shape_parameter > 0:
                # Fatter tails have higher xi
                evt_indicator = min(1.0, evt_metrics.shape_parameter * 2)
            else:
                evt_indicator = 0.0
            indicators.append(evt_indicator)

        # Hill estimator (alternative tail index)
        losses = -np.sort(returns)[:int(len(returns) * 0.1)]  # Top 10% losses
        if len(losses) > 1:
            hill_estimate = np.mean(np.log(losses[:-1] / losses[-1]))
            hill_indicator = min(1.0, max(0.0, 1.0 / hill_estimate / 5.0))
            indicators.append(hill_indicator)

        # Aggregate indicators
        fat_tail_indicator = np.mean(indicators)

        return float(fat_tail_indicator)

    def calculate_tail_correlation(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray,
        percentile: float = 0.05
    ) -> float:
        """
        Calculate tail correlation between two return series

        Measures correlation in extreme events

        Args:
            returns1: First return series
            returns2: Second return series
            percentile: Percentile for tail definition

        Returns:
            Tail correlation coefficient
        """
        # Identify tail events for returns1
        threshold1 = np.quantile(returns1, percentile)
        tail_mask = returns1 <= threshold1

        if np.sum(tail_mask) < 10:
            return 0.0

        # Calculate correlation in tail events
        tail_corr = np.corrcoef(returns1[tail_mask], returns2[tail_mask])[0, 1]

        return float(tail_corr)

    def generate_tail_scenarios(
        self,
        current_positions: Dict[str, float],
        historical_returns: pd.DataFrame,
        evt_metrics: Optional[EVTMetrics] = None
    ) -> Dict[str, float]:
        """
        Generate tail risk scenarios and estimate losses

        Args:
            current_positions: Dict mapping assets to positions
            historical_returns: DataFrame of historical returns
            evt_metrics: Optional EVT metrics for scenario generation

        Returns:
            Dict mapping scenario names to estimated losses
        """
        scenarios = {}

        # Scenario 1: Historical worst day
        worst_day_returns = historical_returns.min()
        worst_day_loss = sum(
            current_positions.get(asset, 0) * worst_day_returns[asset]
            for asset in worst_day_returns.index
        )
        scenarios['historical_worst_day'] = float(worst_day_loss)

        # Scenario 2: Historical worst week
        weekly_returns = historical_returns.rolling(window=5).sum()
        worst_week_returns = weekly_returns.min()
        worst_week_loss = sum(
            current_positions.get(asset, 0) * worst_week_returns[asset]
            for asset in worst_week_returns.index
            if not np.isnan(worst_week_returns[asset])
        )
        scenarios['historical_worst_week'] = float(worst_week_loss)

        # Scenario 3: 3-sigma event
        mean_returns = historical_returns.mean()
        std_returns = historical_returns.std()
        sigma3_returns = mean_returns - 3 * std_returns
        sigma3_loss = sum(
            current_positions.get(asset, 0) * sigma3_returns[asset]
            for asset in sigma3_returns.index
        )
        scenarios['3_sigma_event'] = float(sigma3_loss)

        # Scenario 4: Correlated crash (all assets move to -2 sigma)
        sigma2_returns = mean_returns - 2 * std_returns
        correlated_crash_loss = sum(
            current_positions.get(asset, 0) * sigma2_returns[asset]
            for asset in sigma2_returns.index
        )
        scenarios['correlated_crash'] = float(correlated_crash_loss)

        # Scenario 5: EVT extreme scenario (if available)
        if evt_metrics is not None:
            # Use 99.9th percentile from EVT
            evt_var_999 = evt_metrics.extreme_quantiles.get(0.999, 0.0)
            portfolio_returns = historical_returns.sum(axis=1)
            current_total = sum(abs(v) for v in current_positions.values())
            evt_loss = -current_total * evt_var_999
            scenarios['evt_extreme_999'] = float(evt_loss)

        return scenarios

    def generate_hedge_recommendations(
        self,
        tail_metrics: 'TailRiskMetrics',
        risk_budget: float
    ) -> List[str]:
        """
        Generate tail hedging recommendations

        Args:
            tail_metrics: Calculated tail risk metrics
            risk_budget: Available budget for hedging (% of portfolio)

        Returns:
            List of hedge recommendations
        """
        recommendations = []

        # High fat tail indicator
        if tail_metrics.fat_tail_indicator > 0.6:
            recommendations.append(
                f"Fat tails detected ({tail_metrics.fat_tail_indicator:.2f}): "
                "Consider put options or VIX futures for tail protection"
            )

        # Negative skewness
        if tail_metrics.skewness < -0.5:
            recommendations.append(
                f"Negative skewness ({tail_metrics.skewness:.2f}): "
                "Consider out-of-the-money put spreads"
            )

        # High jump probability
        if tail_metrics.jump_probability > 0.05:
            recommendations.append(
                f"High jump risk ({tail_metrics.jump_probability:.1%}): "
                "Consider straddle or strangle strategies"
            )

        # High excess kurtosis
        if tail_metrics.excess_kurtosis > self.kurtosis_threshold:
            recommendations.append(
                f"High excess kurtosis ({tail_metrics.excess_kurtosis:.2f}): "
                "Increase tail hedge allocation to 2-3% of portfolio"
            )

        # EVT-based recommendations
        if tail_metrics.evt_metrics is not None:
            evt = tail_metrics.evt_metrics
            if evt.shape_parameter > 0.2:
                recommendations.append(
                    f"Heavy tail detected (ξ={evt.shape_parameter:.2f}): "
                    "Consider dynamic tail hedging with option overlays"
                )

        # Large expected shortfall
        es_ratio = tail_metrics.expected_shortfall_99 / np.std(tail_metrics.largest_losses) if len(tail_metrics.largest_losses) > 0 else 0
        if es_ratio > 3:
            recommendations.append(
                "Large expected shortfall: Implement systematic rebalancing during stress"
            )

        # Default recommendation if no specific issues
        if not recommendations:
            recommendations.append(
                "Tail risk appears moderate: Maintain 1-2% allocation to protective puts"
            )

        return recommendations

    def generate_tail_alerts(
        self,
        metrics: 'TailRiskMetrics'
    ) -> List[TailRiskAlert]:
        """
        Generate alerts based on tail risk metrics

        Args:
            metrics: Tail risk metrics

        Returns:
            List of tail risk alerts
        """
        alerts = []
        timestamp = metrics.timestamp

        # High excess kurtosis alert
        if metrics.excess_kurtosis > self.kurtosis_threshold:
            severity = 'critical' if metrics.excess_kurtosis > 10 else 'high'
            alerts.append(TailRiskAlert(
                timestamp=timestamp,
                alert_type='kurtosis',
                severity=severity,
                message=f"High excess kurtosis detected: {metrics.excess_kurtosis:.2f}",
                asset=None,
                metric_value=metrics.excess_kurtosis,
                threshold=self.kurtosis_threshold,
                details={'kurtosis': metrics.kurtosis}
            ))

        # Fat tail alert
        if metrics.fat_tail_indicator > 0.6:
            alerts.append(TailRiskAlert(
                timestamp=timestamp,
                alert_type='fat_tail',
                severity='high',
                message=f"Fat tail indicator elevated: {metrics.fat_tail_indicator:.2f}",
                asset=None,
                metric_value=metrics.fat_tail_indicator,
                threshold=0.6,
                details={'skewness': metrics.skewness}
            ))

        # Non-normal distribution alert
        if not metrics.is_normal:
            alerts.append(TailRiskAlert(
                timestamp=timestamp,
                alert_type='distribution',
                severity='medium',
                message=f"Returns deviate from normal (JB p-value: {metrics.jarque_bera_pvalue:.4f})",
                asset=None,
                metric_value=metrics.jarque_bera_stat,
                threshold=0.05,
                details={'is_normal': False}
            ))

        # Jump risk alert
        if metrics.jump_probability > 0.05:
            alerts.append(TailRiskAlert(
                timestamp=timestamp,
                alert_type='jump_risk',
                severity='high',
                message=f"High jump probability: {metrics.jump_probability:.1%}",
                asset=None,
                metric_value=metrics.jump_probability,
                threshold=0.05,
                details={'jump_threshold': self.jump_threshold}
            ))

        # Negative skewness alert
        if metrics.skewness < -1.0:
            alerts.append(TailRiskAlert(
                timestamp=timestamp,
                alert_type='skewness',
                severity='medium',
                message=f"Significant negative skewness: {metrics.skewness:.2f}",
                asset=None,
                metric_value=metrics.skewness,
                threshold=-1.0,
                details={'interpretation': 'Higher probability of large losses'}
            ))

        # Extreme CVaR alert
        if metrics.tail_cvar_99 > 0.1:  # 10% CVaR
            alerts.append(TailRiskAlert(
                timestamp=timestamp,
                alert_type='extreme_loss',
                severity='critical',
                message=f"Extreme CVaR 99%: {metrics.tail_cvar_99:.1%}",
                asset=None,
                metric_value=metrics.tail_cvar_99,
                threshold=0.1,
                details={'var_99': metrics.tail_var_99}
            ))

        # Tail hedge recommendation alert
        if metrics.hedge_recommendations:
            alerts.append(TailRiskAlert(
                timestamp=timestamp,
                alert_type='tail_hedge',
                severity='medium',
                message=f"Tail hedging recommended: {len(metrics.hedge_recommendations)} strategies",
                asset=None,
                metric_value=float(len(metrics.hedge_recommendations)),
                threshold=1.0,
                details={'recommendations': metrics.hedge_recommendations}
            ))

        return alerts

    def calculate_tail_risk(
        self,
        returns: np.ndarray,
        positions: Optional[Dict[str, float]] = None,
        historical_returns: Optional[pd.DataFrame] = None,
        risk_budget: float = 0.02,
        timestamp: Optional[datetime] = None
    ) -> TailRiskMetrics:
        """
        Calculate comprehensive tail risk metrics

        Args:
            returns: Array of portfolio returns
            positions: Optional dict of current positions
            historical_returns: Optional DataFrame of historical returns by asset
            risk_budget: Budget for tail hedging (% of portfolio)
            timestamp: Current timestamp (uses now if None)

        Returns:
            TailRiskMetrics with comprehensive analysis
        """
        timestamp = timestamp or datetime.now()

        # Calculate higher moments
        skewness, kurtosis, excess_kurtosis = self.calculate_higher_moments(returns)

        # Test normality
        jb_stat, jb_pvalue, is_normal = self.test_normality(returns)

        # Calculate EVT metrics
        try:
            evt_metrics = self.calculate_evt_metrics(returns)
            tail_var_95 = evt_metrics.var_95
            tail_var_99 = evt_metrics.var_99
            tail_cvar_95 = evt_metrics.cvar_95
            tail_cvar_99 = evt_metrics.cvar_99
        except Exception as e:
            logger.warning(f"EVT calculation failed: {e}")
            evt_metrics = None
            tail_var_95 = float(np.quantile(-returns, 0.95))
            tail_var_99 = float(np.quantile(-returns, 0.99))
            # Simple CVaR estimate
            losses = -returns
            tail_cvar_95 = float(np.mean(losses[losses > np.quantile(losses, 0.95)]))
            tail_cvar_99 = float(np.mean(losses[losses > np.quantile(losses, 0.99)]))

        # Calculate downside deviation
        downside_deviation = self.calculate_downside_deviation(returns)

        # Calculate tail ratio
        tail_ratio = self.calculate_tail_ratio(returns)

        # Detect jumps
        jump_indices, jump_probability = self.detect_jumps(returns)

        # Identify largest losses
        losses = -np.sort(returns)
        largest_losses = losses[:10].tolist()

        # Calculate max drawdown in tail events
        tail_returns = returns[returns < np.quantile(returns, 0.1)]
        cumulative_tail = np.cumprod(1 + tail_returns)
        running_max_tail = np.maximum.accumulate(cumulative_tail)
        drawdowns_tail = (cumulative_tail - running_max_tail) / running_max_tail
        max_drawdown_tail = float(np.min(drawdowns_tail)) if len(drawdowns_tail) > 0 else 0.0

        # Expected shortfall (same as CVaR but using different method)
        expected_shortfall_95 = tail_cvar_95
        expected_shortfall_99 = tail_cvar_99

        # Calculate tail correlation (if historical data available)
        tail_correlation = 0.0
        if historical_returns is not None and len(historical_returns.columns) > 1:
            # Average pairwise tail correlation
            tail_corrs = []
            for i, col1 in enumerate(historical_returns.columns):
                for col2 in historical_returns.columns[i+1:]:
                    tc = self.calculate_tail_correlation(
                        historical_returns[col1].values,
                        historical_returns[col2].values
                    )
                    tail_corrs.append(tc)
            tail_correlation = float(np.mean(tail_corrs)) if tail_corrs else 0.0

        # Calculate fat tail indicator
        fat_tail_indicator = self.calculate_fat_tail_indicator(returns, evt_metrics)

        # Generate scenarios
        scenario_losses = {}
        if positions is not None and historical_returns is not None:
            scenario_losses = self.generate_tail_scenarios(
                positions,
                historical_returns,
                evt_metrics
            )

        # Store extreme events
        for idx in jump_indices[-10:]:  # Last 10 jumps
            if idx < len(returns):
                self.extreme_events.append((timestamp, float(returns[idx])))
        if len(self.extreme_events) > 1000:
            self.extreme_events = self.extreme_events[-1000:]

        # Create metrics object
        metrics = TailRiskMetrics(
            timestamp=timestamp,
            kurtosis=kurtosis,
            excess_kurtosis=excess_kurtosis,
            skewness=skewness,
            jarque_bera_stat=jb_stat,
            jarque_bera_pvalue=jb_pvalue,
            is_normal=is_normal,
            evt_metrics=evt_metrics,
            tail_var_95=tail_var_95,
            tail_var_99=tail_var_99,
            tail_cvar_95=tail_cvar_95,
            tail_cvar_99=tail_cvar_99,
            max_drawdown_tail=max_drawdown_tail,
            expected_shortfall_95=expected_shortfall_95,
            expected_shortfall_99=expected_shortfall_99,
            tail_ratio=tail_ratio,
            downside_deviation=downside_deviation,
            jump_probability=jump_probability,
            largest_losses=largest_losses,
            tail_correlation=tail_correlation,
            fat_tail_indicator=fat_tail_indicator,
            hedge_recommendations=[],
            scenario_losses=scenario_losses,
            alerts=[]
        )

        # Generate hedge recommendations
        metrics.hedge_recommendations = self.generate_hedge_recommendations(metrics, risk_budget)

        # Generate alerts
        metrics.alerts = self.generate_tail_alerts(metrics)

        # Store in history
        self.tail_metrics_history.append((timestamp, fat_tail_indicator))
        if len(self.tail_metrics_history) > 1000:
            self.tail_metrics_history.pop(0)

        logger.info(
            f"Tail risk calculated: excess_kurt={excess_kurtosis:.2f}, "
            f"skew={skewness:.2f}, fat_tail={fat_tail_indicator:.2f}"
        )

        return metrics

    def get_tail_risk_report(self, metrics: TailRiskMetrics) -> str:
        """
        Generate human-readable tail risk report

        Args:
            metrics: Tail risk metrics

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("TAIL RISK REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {metrics.timestamp}")
        report.append("")

        report.append("Distribution Characteristics:")
        report.append(f"  Skewness:                {metrics.skewness:>8.3f}")
        report.append(f"  Kurtosis:                {metrics.kurtosis:>8.3f}")
        report.append(f"  Excess Kurtosis:         {metrics.excess_kurtosis:>8.3f}")
        report.append(f"  Jarque-Bera Stat:        {metrics.jarque_bera_stat:>8.2f}")
        report.append(f"  Normal Distribution:     {'Yes' if metrics.is_normal else 'No'}")
        report.append(f"  Fat Tail Indicator:      {metrics.fat_tail_indicator:>8.3f}")
        report.append("")

        report.append("Tail Risk Metrics:")
        report.append(f"  Tail VaR 95%:            {metrics.tail_var_95:>8.2%}")
        report.append(f"  Tail VaR 99%:            {metrics.tail_var_99:>8.2%}")
        report.append(f"  Tail CVaR 95%:           {metrics.tail_cvar_95:>8.2%}")
        report.append(f"  Tail CVaR 99%:           {metrics.tail_cvar_99:>8.2%}")
        report.append(f"  Expected Shortfall 95%:  {metrics.expected_shortfall_95:>8.2%}")
        report.append(f"  Expected Shortfall 99%:  {metrics.expected_shortfall_99:>8.2%}")
        report.append(f"  Downside Deviation:      {metrics.downside_deviation:>8.2%}")
        report.append(f"  Tail Ratio:              {metrics.tail_ratio:>8.3f}")
        report.append(f"  Max Drawdown (Tail):     {metrics.max_drawdown_tail:>8.2%}")
        report.append("")

        report.append("Jump Risk:")
        report.append(f"  Jump Probability:        {metrics.jump_probability:>8.2%}")
        report.append(f"  Tail Correlation:        {metrics.tail_correlation:>8.3f}")
        report.append("")

        if metrics.evt_metrics is not None:
            evt = metrics.evt_metrics
            report.append("Extreme Value Theory (EVT):")
            report.append(f"  Shape Parameter (ξ):     {evt.shape_parameter:>8.4f}")
            report.append(f"  Scale Parameter (σ):     {evt.scale_parameter:>8.4f}")
            report.append(f"  Tail Index:              {evt.tail_index:>8.2f}")
            report.append(f"  Threshold:               {evt.threshold:>8.4f}")
            report.append(f"  Number of Exceedances:   {evt.n_exceedances:>8d}")
            report.append("")

        if metrics.largest_losses:
            report.append("Top 10 Largest Losses:")
            for i, loss in enumerate(metrics.largest_losses, 1):
                report.append(f"  {i:2d}. {loss:>8.2%}")
            report.append("")

        if metrics.scenario_losses:
            report.append("Tail Risk Scenarios:")
            for scenario, loss in metrics.scenario_losses.items():
                report.append(f"  {scenario:30s} ${loss:>12,.0f}")
            report.append("")

        if metrics.hedge_recommendations:
            report.append(f"Hedge Recommendations ({len(metrics.hedge_recommendations)}):")
            for i, rec in enumerate(metrics.hedge_recommendations, 1):
                report.append(f"  {i}. {rec}")
            report.append("")

        if metrics.alerts:
            report.append(f"ALERTS ({len(metrics.alerts)}):")
            for alert in metrics.alerts:
                report.append(f"  [{alert.severity.upper()}] {alert.alert_type}: {alert.message}")
            report.append("")

        report.append("=" * 60)
        return "\n".join(report)
