"""
Liquidity Risk Management Module

This module implements liquidity risk monitoring and management for portfolio risk,
including liquidity scoring, spread monitoring, and liquidation cost estimation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class LiquidityAlert:
    """Alert for liquidity risk events"""
    timestamp: datetime
    alert_type: str  # 'low_liquidity', 'high_spread', 'low_volume', 'liquidation_risk'
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    asset: str
    liquidity_score: float
    threshold: float
    details: Dict = field(default_factory=dict)


@dataclass
class LiquidityMetrics:
    """Comprehensive liquidity risk metrics"""
    timestamp: datetime
    asset_liquidity_scores: Dict[str, float]
    portfolio_liquidity_score: float
    weighted_liquidity_score: float
    bid_ask_spreads: Dict[str, float]
    avg_spread: float
    max_spread: float
    volume_profiles: Dict[str, Dict[str, float]]
    liquidation_costs: Dict[str, float]
    total_liquidation_cost: float
    time_to_liquidate: Dict[str, float]
    max_liquidation_time: float
    market_depth_scores: Dict[str, float]
    illiquid_positions: List[str]
    liquidity_concentration: float
    alerts: List[LiquidityAlert]


@dataclass
class MarketDepthData:
    """Market depth information for an asset"""
    asset: str
    timestamp: datetime
    bid_levels: List[Tuple[float, float]]  # (price, volume)
    ask_levels: List[Tuple[float, float]]  # (price, volume)
    bid_depth: float  # Total volume at all bid levels
    ask_depth: float  # Total volume at all ask levels
    mid_price: float
    spread: float
    spread_bps: float


class LiquidityRisk:
    """
    Liquidity Risk Monitor

    Monitors and analyzes liquidity risk for portfolio positions,
    including spread monitoring, volume analysis, and liquidation cost estimation.

    Features:
    - Liquidity score calculation per instrument
    - Bid-ask spread monitoring
    - Volume profile tracking
    - Liquidation cost estimation
    - Time to liquidation calculation
    - Market depth monitoring
    - Liquidity warning generation

    Attributes:
        min_liquidity_score (float): Minimum acceptable liquidity score
        max_spread_bps (float): Maximum acceptable spread in basis points
        min_daily_volume (float): Minimum acceptable daily volume
        liquidation_participation_rate (float): Max participation in daily volume
    """

    def __init__(
        self,
        min_liquidity_score: float = 0.5,
        max_spread_bps: float = 50.0,
        min_daily_volume: float = 1_000_000,
        liquidation_participation_rate: float = 0.10,
        critical_spread_bps: float = 100.0,
        depth_levels: int = 5
    ):
        """
        Initialize Liquidity Risk Monitor

        Args:
            min_liquidity_score: Minimum acceptable liquidity score (0-1)
            max_spread_bps: Maximum acceptable spread in basis points
            min_daily_volume: Minimum acceptable daily volume in dollars
            liquidation_participation_rate: Max participation rate (e.g., 0.10 = 10% of daily volume)
            critical_spread_bps: Critical spread threshold for alerts
            depth_levels: Number of order book levels to analyze
        """
        self.min_liquidity_score = min_liquidity_score
        self.max_spread_bps = max_spread_bps
        self.min_daily_volume = min_daily_volume
        self.liquidation_participation_rate = liquidation_participation_rate
        self.critical_spread_bps = critical_spread_bps
        self.depth_levels = depth_levels

        # Historical data storage
        self.liquidity_history: List[Tuple[datetime, float]] = []
        self.spread_history: Dict[str, List[Tuple[datetime, float]]] = {}

        logger.info(f"LiquidityRisk initialized with min_score={min_liquidity_score}")

    def calculate_bid_ask_spread(
        self,
        bid_price: float,
        ask_price: float,
        method: str = 'relative'
    ) -> float:
        """
        Calculate bid-ask spread

        Args:
            bid_price: Best bid price
            ask_price: Best ask price
            method: 'absolute' or 'relative' (basis points)

        Returns:
            Spread value
        """
        if method == 'absolute':
            spread = ask_price - bid_price
        else:  # relative (basis points)
            mid_price = (bid_price + ask_price) / 2
            if mid_price > 0:
                spread = ((ask_price - bid_price) / mid_price) * 10000  # bps
            else:
                spread = 0.0

        return float(spread)

    def calculate_volume_profile(
        self,
        volume_data: pd.DataFrame,
        lookback_days: int = 20
    ) -> Dict[str, float]:
        """
        Calculate volume profile metrics

        Args:
            volume_data: DataFrame with 'volume' and 'price' columns
            lookback_days: Number of days to analyze

        Returns:
            Dict with volume profile metrics
        """
        if len(volume_data) == 0:
            return {
                'avg_daily_volume': 0.0,
                'avg_daily_dollar_volume': 0.0,
                'volume_volatility': 0.0,
                'volume_trend': 0.0
            }

        recent_data = volume_data.tail(lookback_days)

        # Calculate metrics
        avg_daily_volume = float(recent_data['volume'].mean())
        avg_daily_dollar_volume = float((recent_data['volume'] * recent_data['price']).mean())
        volume_volatility = float(recent_data['volume'].std() / avg_daily_volume if avg_daily_volume > 0 else 0)

        # Volume trend (positive = increasing, negative = decreasing)
        if len(recent_data) > 1:
            volume_trend = float(np.polyfit(range(len(recent_data)), recent_data['volume'].values, 1)[0])
            volume_trend = volume_trend / avg_daily_volume if avg_daily_volume > 0 else 0
        else:
            volume_trend = 0.0

        return {
            'avg_daily_volume': avg_daily_volume,
            'avg_daily_dollar_volume': avg_daily_dollar_volume,
            'volume_volatility': volume_volatility,
            'volume_trend': volume_trend
        }

    def calculate_liquidity_score(
        self,
        spread_bps: float,
        daily_dollar_volume: float,
        market_cap: Optional[float] = None,
        volume_volatility: Optional[float] = None
    ) -> float:
        """
        Calculate comprehensive liquidity score for an asset

        Combines multiple liquidity indicators into single score
        Range: [0, 1] where 1 = highly liquid, 0 = illiquid

        Args:
            spread_bps: Bid-ask spread in basis points
            daily_dollar_volume: Average daily dollar volume
            market_cap: Optional market capitalization
            volume_volatility: Optional volume volatility metric

        Returns:
            Liquidity score (0-1)
        """
        scores = []
        weights = []

        # Spread score (inverted - lower spread is better)
        if spread_bps <= self.max_spread_bps:
            spread_score = 1.0 - (spread_bps / self.max_spread_bps) * 0.5
        else:
            spread_score = 0.5 * (self.critical_spread_bps - spread_bps) / self.critical_spread_bps
            spread_score = max(0.0, spread_score)

        scores.append(spread_score)
        weights.append(0.4)

        # Volume score
        if daily_dollar_volume >= self.min_daily_volume:
            volume_score = min(1.0, daily_dollar_volume / (self.min_daily_volume * 10))
        else:
            volume_score = daily_dollar_volume / self.min_daily_volume

        scores.append(volume_score)
        weights.append(0.4)

        # Market cap score (if available)
        if market_cap is not None:
            if market_cap >= 10_000_000_000:  # $10B+
                cap_score = 1.0
            elif market_cap >= 1_000_000_000:  # $1B+
                cap_score = 0.8
            elif market_cap >= 100_000_000:  # $100M+
                cap_score = 0.6
            else:
                cap_score = 0.4

            scores.append(cap_score)
            weights.append(0.1)

        # Volume volatility score (if available)
        if volume_volatility is not None:
            # Lower volatility is better
            if volume_volatility <= 0.5:
                vol_vol_score = 1.0
            elif volume_volatility <= 1.0:
                vol_vol_score = 0.7
            else:
                vol_vol_score = max(0.0, 1.0 - (volume_volatility - 1.0) / 2.0)

            scores.append(vol_vol_score)
            weights.append(0.1)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Calculate weighted score
        liquidity_score = sum(s * w for s, w in zip(scores, weights))

        return float(np.clip(liquidity_score, 0.0, 1.0))

    def estimate_liquidation_cost(
        self,
        position_size: float,
        spread_bps: float,
        market_depth: Optional[MarketDepthData] = None,
        daily_volume: Optional[float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Estimate cost of liquidating a position

        Args:
            position_size: Size of position in dollars
            spread_bps: Current bid-ask spread in basis points
            market_depth: Optional market depth data
            daily_volume: Optional average daily volume

        Returns:
            Tuple of (total_cost, cost_breakdown_dict)
        """
        costs = {}

        # Base spread cost
        spread_cost = position_size * (spread_bps / 10000)
        costs['spread_cost'] = spread_cost

        # Market impact cost
        if daily_volume is not None and daily_volume > 0:
            # Estimate impact using square root law
            participation_rate = position_size / daily_volume
            impact_bps = spread_bps * np.sqrt(participation_rate / self.liquidation_participation_rate)
            impact_cost = position_size * (impact_bps / 10000)
        else:
            # Conservative estimate
            impact_cost = position_size * 0.005  # 50 bps
        costs['impact_cost'] = impact_cost

        # Opportunity cost (time risk)
        if daily_volume is not None and daily_volume > 0:
            # Days to liquidate at participation rate
            days_to_liquidate = position_size / (daily_volume * self.liquidation_participation_rate)
            # Assume 10 bps per day opportunity cost
            opportunity_cost = position_size * 0.001 * days_to_liquidate
        else:
            opportunity_cost = position_size * 0.002  # 20 bps conservative
        costs['opportunity_cost'] = opportunity_cost

        # Depth-based slippage (if available)
        if market_depth is not None:
            slippage_cost = self._calculate_depth_slippage(position_size, market_depth)
            costs['slippage_cost'] = slippage_cost
        else:
            costs['slippage_cost'] = 0.0

        total_cost = sum(costs.values())

        return float(total_cost), costs

    def _calculate_depth_slippage(
        self,
        position_size: float,
        market_depth: MarketDepthData
    ) -> float:
        """
        Calculate slippage based on order book depth

        Args:
            position_size: Size of position to liquidate
            market_depth: Market depth data

        Returns:
            Estimated slippage cost
        """
        # Assuming we're selling (using bid side)
        remaining_size = position_size / market_depth.mid_price
        total_cost = 0.0

        for price, volume in market_depth.bid_levels:
            if remaining_size <= 0:
                break

            filled_size = min(remaining_size, volume)
            slippage_per_unit = market_depth.mid_price - price
            total_cost += filled_size * slippage_per_unit
            remaining_size -= filled_size

        # If we couldn't fill from depth data, add conservative estimate
        if remaining_size > 0:
            total_cost += remaining_size * market_depth.mid_price * 0.01  # 1% slippage

        return float(total_cost)

    def calculate_time_to_liquidate(
        self,
        position_size: float,
        daily_volume: float,
        urgency: str = 'normal'
    ) -> float:
        """
        Calculate estimated time to liquidate position

        Args:
            position_size: Size of position in dollars
            daily_volume: Average daily volume in dollars
            urgency: 'low', 'normal', 'high', 'emergency'

        Returns:
            Time to liquidate in days
        """
        if daily_volume <= 0:
            return float('inf')

        # Adjust participation rate based on urgency
        urgency_rates = {
            'low': 0.05,      # 5% of daily volume
            'normal': 0.10,   # 10% of daily volume
            'high': 0.20,     # 20% of daily volume
            'emergency': 0.50  # 50% of daily volume
        }

        participation_rate = urgency_rates.get(urgency, self.liquidation_participation_rate)

        # Calculate days
        days_to_liquidate = position_size / (daily_volume * participation_rate)

        return float(days_to_liquidate)

    def calculate_market_depth_score(
        self,
        market_depth: MarketDepthData,
        position_size: float
    ) -> float:
        """
        Calculate market depth score for a position

        Measures how well the order book can absorb the position
        Range: [0, 1] where 1 = excellent depth, 0 = poor depth

        Args:
            market_depth: Market depth data
            position_size: Position size in dollars

        Returns:
            Market depth score (0-1)
        """
        # Calculate total depth in dollars
        bid_depth_dollars = sum(price * volume for price, volume in market_depth.bid_levels)
        ask_depth_dollars = sum(price * volume for price, volume in market_depth.ask_levels)
        total_depth = (bid_depth_dollars + ask_depth_dollars) / 2

        # Compare to position size
        if total_depth <= 0:
            return 0.0

        depth_ratio = total_depth / position_size

        # Score based on ratio
        if depth_ratio >= 10:
            score = 1.0
        elif depth_ratio >= 5:
            score = 0.9
        elif depth_ratio >= 2:
            score = 0.7
        elif depth_ratio >= 1:
            score = 0.5
        else:
            score = 0.3 * depth_ratio

        return float(np.clip(score, 0.0, 1.0))

    def identify_illiquid_positions(
        self,
        asset_liquidity_scores: Dict[str, float],
        threshold: Optional[float] = None
    ) -> List[str]:
        """
        Identify positions with low liquidity

        Args:
            asset_liquidity_scores: Dict mapping assets to liquidity scores
            threshold: Liquidity score threshold (uses class default if None)

        Returns:
            List of illiquid asset symbols
        """
        threshold = threshold or self.min_liquidity_score

        illiquid_assets = [
            asset for asset, score in asset_liquidity_scores.items()
            if score < threshold
        ]

        return illiquid_assets

    def calculate_liquidity_concentration(
        self,
        positions: Dict[str, float],
        liquidity_scores: Dict[str, float]
    ) -> float:
        """
        Calculate concentration in illiquid positions

        Args:
            positions: Dict mapping assets to position sizes (dollars)
            liquidity_scores: Dict mapping assets to liquidity scores

        Returns:
            Percentage of portfolio in illiquid positions
        """
        total_value = sum(abs(pos) for pos in positions.values())

        if total_value == 0:
            return 0.0

        illiquid_value = sum(
            abs(positions.get(asset, 0))
            for asset, score in liquidity_scores.items()
            if score < self.min_liquidity_score
        )

        concentration = illiquid_value / total_value

        return float(concentration)

    def generate_liquidity_alerts(
        self,
        metrics: LiquidityMetrics
    ) -> List[LiquidityAlert]:
        """
        Generate alerts based on liquidity metrics

        Args:
            metrics: Liquidity metrics

        Returns:
            List of liquidity alerts
        """
        alerts = []
        timestamp = metrics.timestamp

        # Low portfolio liquidity alert
        if metrics.portfolio_liquidity_score < self.min_liquidity_score:
            severity = 'critical' if metrics.portfolio_liquidity_score < 0.3 else 'high'
            alerts.append(LiquidityAlert(
                timestamp=timestamp,
                alert_type='low_liquidity',
                severity=severity,
                message=f"Low portfolio liquidity score: {metrics.portfolio_liquidity_score:.2f}",
                asset='portfolio',
                liquidity_score=metrics.portfolio_liquidity_score,
                threshold=self.min_liquidity_score,
                details={'illiquid_positions': len(metrics.illiquid_positions)}
            ))

        # Individual asset liquidity alerts
        for asset, score in metrics.asset_liquidity_scores.items():
            if score < self.min_liquidity_score:
                alerts.append(LiquidityAlert(
                    timestamp=timestamp,
                    alert_type='low_liquidity',
                    severity='medium',
                    message=f"Low liquidity score for {asset}",
                    asset=asset,
                    liquidity_score=score,
                    threshold=self.min_liquidity_score,
                    details={'spread_bps': metrics.bid_ask_spreads.get(asset, 0)}
                ))

        # High spread alerts
        for asset, spread in metrics.bid_ask_spreads.items():
            if spread > self.critical_spread_bps:
                alerts.append(LiquidityAlert(
                    timestamp=timestamp,
                    alert_type='high_spread',
                    severity='high',
                    message=f"Critical spread for {asset}: {spread:.1f} bps",
                    asset=asset,
                    liquidity_score=metrics.asset_liquidity_scores.get(asset, 0),
                    threshold=self.critical_spread_bps,
                    details={'spread_bps': spread}
                ))
            elif spread > self.max_spread_bps:
                alerts.append(LiquidityAlert(
                    timestamp=timestamp,
                    alert_type='high_spread',
                    severity='medium',
                    message=f"High spread for {asset}: {spread:.1f} bps",
                    asset=asset,
                    liquidity_score=metrics.asset_liquidity_scores.get(asset, 0),
                    threshold=self.max_spread_bps,
                    details={'spread_bps': spread}
                ))

        # Liquidation time alerts
        for asset, days in metrics.time_to_liquidate.items():
            if days > 10:
                alerts.append(LiquidityAlert(
                    timestamp=timestamp,
                    alert_type='liquidation_risk',
                    severity='high',
                    message=f"Long liquidation time for {asset}: {days:.1f} days",
                    asset=asset,
                    liquidity_score=metrics.asset_liquidity_scores.get(asset, 0),
                    threshold=10.0,
                    details={'days': days, 'cost': metrics.liquidation_costs.get(asset, 0)}
                ))

        # High liquidation cost alert
        if metrics.total_liquidation_cost > 0:
            total_value = sum(abs(cost) for cost in metrics.liquidation_costs.values())
            cost_pct = metrics.total_liquidation_cost / total_value if total_value > 0 else 0

            if cost_pct > 0.02:  # 2% liquidation cost
                alerts.append(LiquidityAlert(
                    timestamp=timestamp,
                    alert_type='liquidation_risk',
                    severity='high',
                    message=f"High total liquidation cost: {cost_pct:.1%}",
                    asset='portfolio',
                    liquidity_score=metrics.portfolio_liquidity_score,
                    threshold=0.02,
                    details={'total_cost': metrics.total_liquidation_cost}
                ))

        # Liquidity concentration alert
        if metrics.liquidity_concentration > 0.3:
            alerts.append(LiquidityAlert(
                timestamp=timestamp,
                alert_type='concentration',
                severity='medium',
                message=f"High concentration in illiquid positions: {metrics.liquidity_concentration:.1%}",
                asset='portfolio',
                liquidity_score=metrics.portfolio_liquidity_score,
                threshold=0.3,
                details={'illiquid_count': len(metrics.illiquid_positions)}
            ))

        return alerts

    def calculate_liquidity_risk(
        self,
        positions: Dict[str, float],
        spreads: Dict[str, Tuple[float, float]],  # (bid, ask)
        volume_data: Dict[str, pd.DataFrame],
        market_depths: Optional[Dict[str, MarketDepthData]] = None,
        market_caps: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None
    ) -> LiquidityMetrics:
        """
        Calculate comprehensive liquidity risk metrics

        Args:
            positions: Dict mapping assets to position sizes (dollars)
            spreads: Dict mapping assets to (bid, ask) prices
            volume_data: Dict mapping assets to DataFrames with volume/price data
            market_depths: Optional dict of market depth data
            market_caps: Optional dict of market capitalizations
            timestamp: Current timestamp (uses now if None)

        Returns:
            LiquidityMetrics with comprehensive analysis
        """
        timestamp = timestamp or datetime.now()

        asset_liquidity_scores = {}
        bid_ask_spreads = {}
        volume_profiles = {}
        liquidation_costs = {}
        time_to_liquidate = {}
        market_depth_scores = {}

        # Calculate metrics for each asset
        for asset in positions.keys():
            # Calculate spread
            if asset in spreads:
                bid, ask = spreads[asset]
                spread_bps = self.calculate_bid_ask_spread(bid, ask, method='relative')
                bid_ask_spreads[asset] = spread_bps
            else:
                spread_bps = self.max_spread_bps
                bid_ask_spreads[asset] = spread_bps

            # Calculate volume profile
            if asset in volume_data:
                profile = self.calculate_volume_profile(volume_data[asset])
                volume_profiles[asset] = profile
                daily_volume = profile['avg_daily_dollar_volume']
                volume_volatility = profile['volume_volatility']
            else:
                daily_volume = 0.0
                volume_volatility = None
                volume_profiles[asset] = {
                    'avg_daily_volume': 0.0,
                    'avg_daily_dollar_volume': 0.0,
                    'volume_volatility': 0.0,
                    'volume_trend': 0.0
                }

            # Calculate liquidity score
            market_cap = market_caps.get(asset) if market_caps else None
            liquidity_score = self.calculate_liquidity_score(
                spread_bps,
                daily_volume,
                market_cap,
                volume_volatility
            )
            asset_liquidity_scores[asset] = liquidity_score

            # Calculate liquidation cost
            position_size = abs(positions[asset])
            market_depth = market_depths.get(asset) if market_depths else None
            total_cost, _ = self.estimate_liquidation_cost(
                position_size,
                spread_bps,
                market_depth,
                daily_volume
            )
            liquidation_costs[asset] = total_cost

            # Calculate time to liquidate
            if daily_volume > 0:
                days = self.calculate_time_to_liquidate(position_size, daily_volume)
                time_to_liquidate[asset] = days
            else:
                time_to_liquidate[asset] = float('inf')

            # Calculate market depth score
            if market_depth is not None:
                depth_score = self.calculate_market_depth_score(market_depth, position_size)
                market_depth_scores[asset] = depth_score

        # Calculate portfolio-level metrics
        total_value = sum(abs(pos) for pos in positions.values())

        # Weighted average liquidity score
        if total_value > 0:
            weighted_liquidity_score = sum(
                asset_liquidity_scores[asset] * abs(positions[asset]) / total_value
                for asset in positions.keys()
            )
        else:
            weighted_liquidity_score = 0.0

        # Simple average liquidity score
        portfolio_liquidity_score = np.mean(list(asset_liquidity_scores.values()))

        # Average spread
        avg_spread = np.mean(list(bid_ask_spreads.values())) if bid_ask_spreads else 0.0
        max_spread = max(bid_ask_spreads.values()) if bid_ask_spreads else 0.0

        # Total liquidation cost
        total_liquidation_cost = sum(liquidation_costs.values())

        # Max liquidation time
        finite_times = [t for t in time_to_liquidate.values() if not np.isinf(t)]
        max_liquidation_time = max(finite_times) if finite_times else float('inf')

        # Identify illiquid positions
        illiquid_positions = self.identify_illiquid_positions(asset_liquidity_scores)

        # Calculate liquidity concentration
        liquidity_concentration = self.calculate_liquidity_concentration(
            positions,
            asset_liquidity_scores
        )

        # Store in history
        self.liquidity_history.append((timestamp, portfolio_liquidity_score))
        if len(self.liquidity_history) > 1000:
            self.liquidity_history.pop(0)

        # Create metrics object
        metrics = LiquidityMetrics(
            timestamp=timestamp,
            asset_liquidity_scores=asset_liquidity_scores,
            portfolio_liquidity_score=portfolio_liquidity_score,
            weighted_liquidity_score=weighted_liquidity_score,
            bid_ask_spreads=bid_ask_spreads,
            avg_spread=avg_spread,
            max_spread=max_spread,
            volume_profiles=volume_profiles,
            liquidation_costs=liquidation_costs,
            total_liquidation_cost=total_liquidation_cost,
            time_to_liquidate=time_to_liquidate,
            max_liquidation_time=max_liquidation_time,
            market_depth_scores=market_depth_scores,
            illiquid_positions=illiquid_positions,
            liquidity_concentration=liquidity_concentration,
            alerts=[]
        )

        # Generate alerts
        metrics.alerts = self.generate_liquidity_alerts(metrics)

        logger.info(
            f"Liquidity risk calculated: score={portfolio_liquidity_score:.2f}, "
            f"avg_spread={avg_spread:.1f}bps, illiquid={len(illiquid_positions)}"
        )

        return metrics

    def get_liquidity_report(self, metrics: LiquidityMetrics) -> str:
        """
        Generate human-readable liquidity risk report

        Args:
            metrics: Liquidity metrics

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("LIQUIDITY RISK REPORT")
        report.append("=" * 60)
        report.append(f"Timestamp: {metrics.timestamp}")
        report.append("")

        report.append("Portfolio Liquidity:")
        report.append(f"  Portfolio Liquidity Score:   {metrics.portfolio_liquidity_score:.2f}")
        report.append(f"  Weighted Liquidity Score:    {metrics.weighted_liquidity_score:.2f}")
        report.append(f"  Liquidity Concentration:     {metrics.liquidity_concentration:.1%}")
        report.append("")

        report.append("Spread Metrics:")
        report.append(f"  Average Spread:              {metrics.avg_spread:.1f} bps")
        report.append(f"  Maximum Spread:              {metrics.max_spread:.1f} bps")
        report.append("")

        report.append("Liquidation Analysis:")
        report.append(f"  Total Liquidation Cost:      ${metrics.total_liquidation_cost:,.0f}")
        if not np.isinf(metrics.max_liquidation_time):
            report.append(f"  Max Liquidation Time:        {metrics.max_liquidation_time:.1f} days")
        else:
            report.append(f"  Max Liquidation Time:        Infinite (no volume data)")
        report.append("")

        if metrics.illiquid_positions:
            report.append(f"Illiquid Positions ({len(metrics.illiquid_positions)}):")
            for asset in metrics.illiquid_positions:
                score = metrics.asset_liquidity_scores[asset]
                spread = metrics.bid_ask_spreads.get(asset, 0)
                report.append(f"  {asset:20s} Score: {score:.2f}, Spread: {spread:.1f} bps")
            report.append("")

        # Top positions by liquidity
        sorted_by_liquidity = sorted(
            metrics.asset_liquidity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

        report.append("Top 10 Most Liquid Positions:")
        for asset, score in sorted_by_liquidity:
            spread = metrics.bid_ask_spreads.get(asset, 0)
            report.append(f"  {asset:20s} Score: {score:.2f}, Spread: {spread:.1f} bps")
        report.append("")

        if metrics.alerts:
            report.append(f"ALERTS ({len(metrics.alerts)}):")
            for alert in metrics.alerts:
                report.append(f"  [{alert.severity.upper()}] {alert.alert_type}: {alert.message}")
            report.append("")

        report.append("=" * 60)
        return "\n".join(report)
