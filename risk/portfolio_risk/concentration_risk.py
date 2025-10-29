"""
Concentration Risk Management Module

This module implements concentration risk monitoring and management for portfolio risk,
including position limits, Herfindahl index, and diversification metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConcentrationAlert:
    """Alert for concentration risk events"""
    timestamp: datetime
    alert_type: str  # 'position', 'sector', 'strategy', 'asset_class', 'herfindahl'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    entity: str  # Asset, sector, or strategy name
    concentration: float
    limit: float
    details: Dict = field(default_factory=dict)


@dataclass
class ConcentrationMetrics:
    """Comprehensive concentration risk metrics"""
    timestamp: datetime
    position_concentrations: Dict[str, float]
    sector_concentrations: Dict[str, float]
    asset_class_concentrations: Dict[str, float]
    strategy_concentrations: Dict[str, float]
    herfindahl_index: float
    herfindahl_normalized: float
    effective_positions: float
    max_position_weight: float
    top_5_concentration: float
    top_10_concentration: float
    gini_coefficient: float
    diversification_score: float
    concentration_contribution: Dict[str, float]
    limit_breaches: List[str]
    alerts: List[ConcentrationAlert]


@dataclass
class ConcentrationLimit:
    """Concentration limit definition"""
    entity_type: str  # 'position', 'sector', 'asset_class', 'strategy'
    entity_name: str
    limit: float  # Maximum allowed weight
    soft_limit: float  # Warning threshold
    hard_limit: float  # Absolute maximum


class ConcentrationRisk:
    """
    Concentration Risk Monitor

    Monitors and manages concentration risk across positions, sectors,
    asset classes, and strategies to ensure proper diversification.

    Features:
    - Position concentration monitoring
    - Herfindahl index calculation
    - Sector/asset class concentration tracking
    - Concentration limit enforcement
    - Concentration contribution to risk
    - Strategy concentration monitoring
    - Diversification metrics

    Attributes:
        max_position_weight (float): Maximum weight per position
        max_sector_weight (float): Maximum weight per sector
        max_strategy_weight (float): Maximum weight per strategy
        herfindahl_threshold (float): Threshold for Herfindahl index alert
    """

    def __init__(
        self,
        max_position_weight: float = 0.20,
        max_sector_weight: float = 0.40,
        max_asset_class_weight: float = 0.50,
        max_strategy_weight: float = 0.35,
        herfindahl_threshold: float = 0.15,
        top5_threshold: float = 0.60
    ):
        """
        Initialize Concentration Risk Monitor

        Args:
            max_position_weight: Maximum allowed weight per position (default: 20%)
            max_sector_weight: Maximum allowed weight per sector (default: 40%)
            max_asset_class_weight: Maximum allowed weight per asset class (default: 50%)
            max_strategy_weight: Maximum allowed weight per strategy (default: 35%)
            herfindahl_threshold: Alert threshold for Herfindahl index (default: 0.15)
            top5_threshold: Alert threshold for top 5 concentration (default: 60%)
        """
        self.max_position_weight = max_position_weight
        self.max_sector_weight = max_sector_weight
        self.max_asset_class_weight = max_asset_class_weight
        self.max_strategy_weight = max_strategy_weight
        self.herfindahl_threshold = herfindahl_threshold
        self.top5_threshold = top5_threshold

        # Custom limits
        self.custom_limits: Dict[str, ConcentrationLimit] = {}

        # History tracking
        self.concentration_history: List[Tuple[datetime, float]] = []

        logger.info(f"ConcentrationRisk initialized with max_position={max_position_weight:.1%}")

    def add_concentration_limit(self, limit: ConcentrationLimit) -> None:
        """
        Add custom concentration limit

        Args:
            limit: ConcentrationLimit object
        """
        key = f"{limit.entity_type}_{limit.entity_name}"
        self.custom_limits[key] = limit
        logger.info(f"Added concentration limit: {key} = {limit.limit:.1%}")

    def calculate_herfindahl_index(self, weights: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Herfindahl-Hirschman Index (HHI)

        HHI measures market concentration. For portfolio, measures weight concentration.
        HHI = sum of squared weights
        Range: [1/N, 1] where N is number of assets

        Args:
            weights: Portfolio weights (should sum to 1)

        Returns:
            Tuple of (HHI, normalized_HHI)
            - HHI: Raw Herfindahl index
            - normalized_HHI: Normalized to [0, 1] where 0=equal weight, 1=single asset
        """
        # Calculate HHI
        hhi = np.sum(weights ** 2)

        # Normalize to [0, 1]
        n = len(weights)
        min_hhi = 1.0 / n  # Equal weights
        max_hhi = 1.0       # Single asset

        if max_hhi - min_hhi > 0:
            normalized_hhi = (hhi - min_hhi) / (max_hhi - min_hhi)
        else:
            normalized_hhi = 0.0

        return float(hhi), float(normalized_hhi)

    def calculate_effective_positions(self, weights: np.ndarray) -> float:
        """
        Calculate effective number of positions

        Effective N = 1 / HHI
        Indicates how many equal-weighted positions would give same concentration

        Args:
            weights: Portfolio weights

        Returns:
            Effective number of positions
        """
        hhi, _ = self.calculate_herfindahl_index(weights)
        if hhi > 0:
            effective_n = 1.0 / hhi
        else:
            effective_n = float(len(weights))

        return float(effective_n)

    def calculate_gini_coefficient(self, weights: np.ndarray) -> float:
        """
        Calculate Gini coefficient for portfolio weights

        Measures inequality in weight distribution.
        Range: [0, 1] where 0=perfect equality, 1=perfect inequality

        Args:
            weights: Portfolio weights (absolute values)

        Returns:
            Gini coefficient
        """
        # Sort weights
        sorted_weights = np.sort(np.abs(weights))
        n = len(sorted_weights)

        if n == 0 or np.sum(sorted_weights) == 0:
            return 0.0

        # Calculate Gini coefficient
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_weights)) / (n * np.sum(sorted_weights)) - (n + 1) / n

        return float(gini)

    def calculate_top_concentration(
        self,
        weights: np.ndarray,
        top_n: int = 5
    ) -> float:
        """
        Calculate concentration in top N positions

        Args:
            weights: Portfolio weights
            top_n: Number of top positions to consider

        Returns:
            Sum of top N weights
        """
        abs_weights = np.abs(weights)
        sorted_weights = np.sort(abs_weights)[::-1]  # Descending order

        top_n = min(top_n, len(sorted_weights))
        concentration = np.sum(sorted_weights[:top_n])

        return float(concentration)

    def calculate_position_concentrations(
        self,
        positions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate concentration for each position

        Args:
            positions: Dict mapping asset symbols to weights

        Returns:
            Dict mapping asset symbols to absolute weights
        """
        total_weight = sum(abs(w) for w in positions.values())

        if total_weight == 0:
            return {asset: 0.0 for asset in positions}

        concentrations = {
            asset: abs(weight) / total_weight
            for asset, weight in positions.items()
        }

        return concentrations

    def calculate_sector_concentrations(
        self,
        positions: Dict[str, float],
        sector_map: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate concentration by sector

        Args:
            positions: Dict mapping asset symbols to weights
            sector_map: Dict mapping asset symbols to sectors

        Returns:
            Dict mapping sectors to total weights
        """
        sector_weights: Dict[str, float] = {}

        for asset, weight in positions.items():
            sector = sector_map.get(asset, 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0.0) + abs(weight)

        total_weight = sum(sector_weights.values())

        if total_weight > 0:
            sector_weights = {
                sector: weight / total_weight
                for sector, weight in sector_weights.items()
            }

        return sector_weights

    def calculate_asset_class_concentrations(
        self,
        positions: Dict[str, float],
        asset_class_map: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate concentration by asset class

        Args:
            positions: Dict mapping asset symbols to weights
            asset_class_map: Dict mapping asset symbols to asset classes

        Returns:
            Dict mapping asset classes to total weights
        """
        class_weights: Dict[str, float] = {}

        for asset, weight in positions.items():
            asset_class = asset_class_map.get(asset, 'Unknown')
            class_weights[asset_class] = class_weights.get(asset_class, 0.0) + abs(weight)

        total_weight = sum(class_weights.values())

        if total_weight > 0:
            class_weights = {
                asset_class: weight / total_weight
                for asset_class, weight in class_weights.items()
            }

        return class_weights

    def calculate_strategy_concentrations(
        self,
        positions: Dict[str, float],
        strategy_map: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate concentration by strategy

        Args:
            positions: Dict mapping asset symbols to weights
            strategy_map: Dict mapping asset symbols to strategies

        Returns:
            Dict mapping strategies to total weights
        """
        strategy_weights: Dict[str, float] = {}

        for asset, weight in positions.items():
            strategy = strategy_map.get(asset, 'Unknown')
            strategy_weights[strategy] = strategy_weights.get(strategy, 0.0) + abs(weight)

        total_weight = sum(strategy_weights.values())

        if total_weight > 0:
            strategy_weights = {
                strategy: weight / total_weight
                for strategy, weight in strategy_weights.items()
            }

        return strategy_weights

    def calculate_concentration_contribution(
        self,
        weights: np.ndarray,
        volatilities: np.ndarray,
        correlation_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate each position's contribution to concentration risk

        Measures marginal contribution to portfolio variance from concentration

        Args:
            weights: Portfolio weights
            volatilities: Asset volatilities
            correlation_matrix: Asset correlation matrix

        Returns:
            Dict mapping position indices to concentration contribution
        """
        # Calculate covariance matrix
        vol_matrix = np.outer(volatilities, volatilities)
        cov_matrix = correlation_matrix * vol_matrix

        # Portfolio variance
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))

        if portfolio_var == 0:
            return {str(i): 0.0 for i in range(len(weights))}

        # Marginal contribution to variance
        marginal_contrib = np.dot(cov_matrix, weights)

        # Component contribution to variance
        component_contrib = weights * marginal_contrib

        # Normalize to get percentage contribution
        contrib_pct = component_contrib / portfolio_var

        contributions = {
            str(i): float(contrib_pct[i])
            for i in range(len(weights))
        }

        return contributions

    def check_concentration_limits(
        self,
        position_concentrations: Dict[str, float],
        sector_concentrations: Dict[str, float],
        asset_class_concentrations: Dict[str, float],
        strategy_concentrations: Dict[str, float]
    ) -> List[str]:
        """
        Check all concentration limits and identify breaches

        Args:
            position_concentrations: Position weight dict
            sector_concentrations: Sector weight dict
            asset_class_concentrations: Asset class weight dict
            strategy_concentrations: Strategy weight dict

        Returns:
            List of breach descriptions
        """
        breaches = []

        # Check position limits
        for asset, weight in position_concentrations.items():
            # Check custom limit first
            key = f"position_{asset}"
            if key in self.custom_limits:
                limit = self.custom_limits[key].limit
            else:
                limit = self.max_position_weight

            if weight > limit:
                breaches.append(f"Position {asset}: {weight:.1%} exceeds limit {limit:.1%}")

        # Check sector limits
        for sector, weight in sector_concentrations.items():
            key = f"sector_{sector}"
            if key in self.custom_limits:
                limit = self.custom_limits[key].limit
            else:
                limit = self.max_sector_weight

            if weight > limit:
                breaches.append(f"Sector {sector}: {weight:.1%} exceeds limit {limit:.1%}")

        # Check asset class limits
        for asset_class, weight in asset_class_concentrations.items():
            key = f"asset_class_{asset_class}"
            if key in self.custom_limits:
                limit = self.custom_limits[key].limit
            else:
                limit = self.max_asset_class_weight

            if weight > limit:
                breaches.append(f"Asset class {asset_class}: {weight:.1%} exceeds limit {limit:.1%}")

        # Check strategy limits
        for strategy, weight in strategy_concentrations.items():
            key = f"strategy_{strategy}"
            if key in self.custom_limits:
                limit = self.custom_limits[key].limit
            else:
                limit = self.max_strategy_weight

            if weight > limit:
                breaches.append(f"Strategy {strategy}: {weight:.1%} exceeds limit {limit:.1%}")

        return breaches

    def calculate_diversification_score(
        self,
        herfindahl_normalized: float,
        gini_coefficient: float,
        effective_positions: float,
        total_positions: int
    ) -> float:
        """
        Calculate overall diversification score

        Combines multiple diversification metrics into single score
        Range: [0, 1] where 1 = perfect diversification

        Args:
            herfindahl_normalized: Normalized Herfindahl index
            gini_coefficient: Gini coefficient
            effective_positions: Effective number of positions
            total_positions: Total number of positions

        Returns:
            Diversification score (0-1)
        """
        # Component scores (inverted so higher is better)
        hhi_score = 1.0 - herfindahl_normalized
        gini_score = 1.0 - gini_coefficient

        # Effective position ratio
        if total_positions > 0:
            position_score = effective_positions / total_positions
        else:
            position_score = 0.0

        # Weighted average
        weights = [0.4, 0.3, 0.3]  # HHI, Gini, Position ratio
        diversification_score = (
            weights[0] * hhi_score +
            weights[1] * gini_score +
            weights[2] * position_score
        )

        return float(np.clip(diversification_score, 0.0, 1.0))

    def generate_concentration_alerts(
        self,
        metrics: ConcentrationMetrics
    ) -> List[ConcentrationAlert]:
        """
        Generate alerts based on concentration metrics

        Args:
            metrics: Concentration metrics

        Returns:
            List of concentration alerts
        """
        alerts = []
        timestamp = metrics.timestamp

        # Position concentration alerts
        for asset, concentration in metrics.position_concentrations.items():
            if concentration > self.max_position_weight:
                severity = 'critical' if concentration > self.max_position_weight * 1.2 else 'high'
                alerts.append(ConcentrationAlert(
                    timestamp=timestamp,
                    alert_type='position',
                    severity=severity,
                    message=f"Position {asset} exceeds concentration limit",
                    entity=asset,
                    concentration=concentration,
                    limit=self.max_position_weight,
                    details={'excess': concentration - self.max_position_weight}
                ))

        # Sector concentration alerts
        for sector, concentration in metrics.sector_concentrations.items():
            if concentration > self.max_sector_weight:
                alerts.append(ConcentrationAlert(
                    timestamp=timestamp,
                    alert_type='sector',
                    severity='high',
                    message=f"Sector {sector} exceeds concentration limit",
                    entity=sector,
                    concentration=concentration,
                    limit=self.max_sector_weight,
                    details={'excess': concentration - self.max_sector_weight}
                ))

        # Asset class concentration alerts
        for asset_class, concentration in metrics.asset_class_concentrations.items():
            if concentration > self.max_asset_class_weight:
                alerts.append(ConcentrationAlert(
                    timestamp=timestamp,
                    alert_type='asset_class',
                    severity='medium',
                    message=f"Asset class {asset_class} exceeds concentration limit",
                    entity=asset_class,
                    concentration=concentration,
                    limit=self.max_asset_class_weight,
                    details={'excess': concentration - self.max_asset_class_weight}
                ))

        # Strategy concentration alerts
        for strategy, concentration in metrics.strategy_concentrations.items():
            if concentration > self.max_strategy_weight:
                alerts.append(ConcentrationAlert(
                    timestamp=timestamp,
                    alert_type='strategy',
                    severity='high',
                    message=f"Strategy {strategy} exceeds concentration limit",
                    entity=strategy,
                    concentration=concentration,
                    limit=self.max_strategy_weight,
                    details={'excess': concentration - self.max_strategy_weight}
                ))

        # Herfindahl index alert
        if metrics.herfindahl_index > self.herfindahl_threshold:
            alerts.append(ConcentrationAlert(
                timestamp=timestamp,
                alert_type='herfindahl',
                severity='medium',
                message=f"High Herfindahl index indicates concentration",
                entity='portfolio',
                concentration=metrics.herfindahl_index,
                limit=self.herfindahl_threshold,
                details={'effective_positions': metrics.effective_positions}
            ))

        # Top 5 concentration alert
        if metrics.top_5_concentration > self.top5_threshold:
            alerts.append(ConcentrationAlert(
                timestamp=timestamp,
                alert_type='top_concentration',
                severity='medium',
                message=f"Top 5 positions represent {metrics.top_5_concentration:.1%} of portfolio",
                entity='top_5',
                concentration=metrics.top_5_concentration,
                limit=self.top5_threshold,
                details={'top_10': metrics.top_10_concentration}
            ))

        # Low diversification score alert
        if metrics.diversification_score < 0.5:
            alerts.append(ConcentrationAlert(
                timestamp=timestamp,
                alert_type='diversification',
                severity='high',
                message=f"Low diversification score: {metrics.diversification_score:.2f}",
                entity='portfolio',
                concentration=1.0 - metrics.diversification_score,
                limit=0.5,
                details={
                    'herfindahl': metrics.herfindahl_normalized,
                    'gini': metrics.gini_coefficient
                }
            ))

        return alerts

    def calculate_concentration_risk(
        self,
        positions: Dict[str, float],
        sector_map: Optional[Dict[str, str]] = None,
        asset_class_map: Optional[Dict[str, str]] = None,
        strategy_map: Optional[Dict[str, str]] = None,
        volatilities: Optional[np.ndarray] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None
    ) -> ConcentrationMetrics:
        """
        Calculate comprehensive concentration risk metrics

        Args:
            positions: Dict mapping asset symbols to weights
            sector_map: Optional dict mapping assets to sectors
            asset_class_map: Optional dict mapping assets to asset classes
            strategy_map: Optional dict mapping assets to strategies
            volatilities: Optional array of asset volatilities
            correlation_matrix: Optional asset correlation matrix
            timestamp: Current timestamp (uses now if None)

        Returns:
            ConcentrationMetrics with comprehensive analysis
        """
        timestamp = timestamp or datetime.now()

        # Convert positions to arrays
        assets = list(positions.keys())
        weights = np.array([positions[asset] for asset in assets])

        # Calculate position concentrations
        position_concentrations = self.calculate_position_concentrations(positions)

        # Calculate sector concentrations
        sector_concentrations = {}
        if sector_map is not None:
            sector_concentrations = self.calculate_sector_concentrations(positions, sector_map)

        # Calculate asset class concentrations
        asset_class_concentrations = {}
        if asset_class_map is not None:
            asset_class_concentrations = self.calculate_asset_class_concentrations(
                positions, asset_class_map
            )

        # Calculate strategy concentrations
        strategy_concentrations = {}
        if strategy_map is not None:
            strategy_concentrations = self.calculate_strategy_concentrations(
                positions, strategy_map
            )

        # Calculate Herfindahl index
        abs_weights = np.abs(weights)
        if abs_weights.sum() > 0:
            normalized_weights = abs_weights / abs_weights.sum()
        else:
            normalized_weights = abs_weights

        herfindahl_index, herfindahl_normalized = self.calculate_herfindahl_index(normalized_weights)

        # Calculate effective positions
        effective_positions = self.calculate_effective_positions(normalized_weights)

        # Calculate max position weight
        max_position_weight = max(position_concentrations.values()) if position_concentrations else 0.0

        # Calculate top N concentrations
        top_5_concentration = self.calculate_top_concentration(normalized_weights, top_n=5)
        top_10_concentration = self.calculate_top_concentration(normalized_weights, top_n=10)

        # Calculate Gini coefficient
        gini_coefficient = self.calculate_gini_coefficient(normalized_weights)

        # Calculate concentration contribution to risk
        concentration_contribution = {}
        if volatilities is not None and correlation_matrix is not None:
            concentration_contribution = self.calculate_concentration_contribution(
                normalized_weights, volatilities, correlation_matrix
            )
            # Map indices to asset names
            concentration_contribution = {
                assets[int(k)]: v
                for k, v in concentration_contribution.items()
            }

        # Calculate diversification score
        diversification_score = self.calculate_diversification_score(
            herfindahl_normalized,
            gini_coefficient,
            effective_positions,
            len(positions)
        )

        # Check concentration limits
        limit_breaches = self.check_concentration_limits(
            position_concentrations,
            sector_concentrations,
            asset_class_concentrations,
            strategy_concentrations
        )

        # Store in history
        self.concentration_history.append((timestamp, herfindahl_index))
        if len(self.concentration_history) > 1000:
            self.concentration_history.pop(0)

        # Create metrics object
        metrics = ConcentrationMetrics(
            timestamp=timestamp,
            position_concentrations=position_concentrations,
            sector_concentrations=sector_concentrations,
            asset_class_concentrations=asset_class_concentrations,
            strategy_concentrations=strategy_concentrations,
            herfindahl_index=herfindahl_index,
            herfindahl_normalized=herfindahl_normalized,
            effective_positions=effective_positions,
            max_position_weight=max_position_weight,
            top_5_concentration=top_5_concentration,
            top_10_concentration=top_10_concentration,
            gini_coefficient=gini_coefficient,
            diversification_score=diversification_score,
            concentration_contribution=concentration_contribution,
            limit_breaches=limit_breaches,
            alerts=[]
        )

        # Generate alerts
        metrics.alerts = self.generate_concentration_alerts(metrics)

        logger.info(
            f"Concentration risk calculated: HHI={herfindahl_index:.3f}, "
            f"eff_pos={effective_positions:.2f}, div_score={diversification_score:.2f}"
        )

        return metrics

    def get_concentration_report(self, metrics: ConcentrationMetrics) -> str:
        """
        Generate human-readable concentration risk report

        Args:
            metrics: Concentration metrics

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("CONCENTRATION RISK REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {metrics.timestamp}")
        report.append("")

        report.append("Concentration Metrics:")
        report.append(f"  Herfindahl Index:       {metrics.herfindahl_index:.4f}")
        report.append(f"  Normalized HHI:         {metrics.herfindahl_normalized:.4f}")
        report.append(f"  Effective Positions:    {metrics.effective_positions:.2f}")
        report.append(f"  Gini Coefficient:       {metrics.gini_coefficient:.4f}")
        report.append(f"  Diversification Score:  {metrics.diversification_score:.2f}")
        report.append("")

        report.append("Position Concentration:")
        report.append(f"  Max Position Weight:    {metrics.max_position_weight:.1%}")
        report.append(f"  Top 5 Concentration:    {metrics.top_5_concentration:.1%}")
        report.append(f"  Top 10 Concentration:   {metrics.top_10_concentration:.1%}")
        report.append("")

        if metrics.position_concentrations:
            report.append("Top Positions:")
            sorted_positions = sorted(
                metrics.position_concentrations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for asset, weight in sorted_positions:
                report.append(f"  {asset:20s} {weight:7.1%}")
            report.append("")

        if metrics.sector_concentrations:
            report.append("Sector Concentration:")
            for sector, weight in sorted(metrics.sector_concentrations.items(), key=lambda x: x[1], reverse=True):
                status = " [BREACH]" if weight > self.max_sector_weight else ""
                report.append(f"  {sector:20s} {weight:7.1%}{status}")
            report.append("")

        if metrics.asset_class_concentrations:
            report.append("Asset Class Concentration:")
            for asset_class, weight in sorted(metrics.asset_class_concentrations.items(), key=lambda x: x[1], reverse=True):
                status = " [BREACH]" if weight > self.max_asset_class_weight else ""
                report.append(f"  {asset_class:20s} {weight:7.1%}{status}")
            report.append("")

        if metrics.strategy_concentrations:
            report.append("Strategy Concentration:")
            for strategy, weight in sorted(metrics.strategy_concentrations.items(), key=lambda x: x[1], reverse=True):
                status = " [BREACH]" if weight > self.max_strategy_weight else ""
                report.append(f"  {strategy:20s} {weight:7.1%}{status}")
            report.append("")

        if metrics.limit_breaches:
            report.append(f"LIMIT BREACHES ({len(metrics.limit_breaches)}):")
            for breach in metrics.limit_breaches:
                report.append(f"  - {breach}")
            report.append("")

        if metrics.alerts:
            report.append(f"ALERTS ({len(metrics.alerts)}):")
            for alert in metrics.alerts:
                report.append(f"  [{alert.severity.upper()}] {alert.alert_type}: {alert.message}")
            report.append("")

        report.append("=" * 60)
        return "\n".join(report)
