"""
Correlation management for portfolio construction.

Manages correlations between assets and strategies to improve diversification
and adjust position sizing. Includes rolling correlations, clustering,
dynamic penalties, and regime-dependent adjustments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from loguru import logger


class CorrelationManager:
    """
    Correlation manager for portfolio optimization.

    Features:
    - Rolling correlation calculation
    - Correlation clustering for grouping
    - Dynamic correlation penalties
    - Regime-dependent correlation adjustments
    - Cross-asset correlation monitoring

    Attributes:
        lookback_window (int): Lookback period for correlation calculation
        min_periods (int): Minimum periods required for correlation
        high_correlation_threshold (float): Threshold for high correlation warning
        cluster_threshold (float): Distance threshold for correlation clustering
    """

    def __init__(
        self,
        lookback_window: int = 60,
        min_periods: int = 30,
        high_correlation_threshold: float = 0.7,
        cluster_threshold: float = 0.5,
    ):
        """
        Initialize correlation manager.

        Args:
            lookback_window: Days to lookback for correlation calculation
            min_periods: Minimum periods required for valid correlation
            high_correlation_threshold: Correlation above this is considered high
            cluster_threshold: Distance threshold for hierarchical clustering
        """
        self.lookback_window = lookback_window
        self.min_periods = min_periods
        self.high_correlation_threshold = high_correlation_threshold
        self.cluster_threshold = cluster_threshold

        logger.info(
            f"CorrelationManager initialized: lookback={lookback_window}, "
            f"high_corr_threshold={high_correlation_threshold}"
        )

    def calculate_rolling_correlation(
        self,
        returns: pd.DataFrame,
        window: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Calculate rolling correlation matrix.

        Args:
            returns: Returns DataFrame (time × assets)
            window: Rolling window size (uses self.lookback_window if None)

        Returns:
            Latest correlation matrix
        """
        if window is None:
            window = self.lookback_window

        if len(returns) < self.min_periods:
            logger.warning(
                f"Insufficient data for correlation: {len(returns)} < {self.min_periods}"
            )
            return pd.DataFrame()

        # Calculate rolling correlation
        rolling_corr = returns.rolling(window=window, min_periods=self.min_periods).corr()

        # Extract latest correlation matrix
        latest_corr = rolling_corr.iloc[-len(returns.columns):]

        # Ensure diagonal is 1 and symmetric
        np.fill_diagonal(latest_corr.values, 1.0)
        latest_corr = (latest_corr + latest_corr.T) / 2

        logger.debug(
            f"Rolling correlation calculated: {len(latest_corr)}x{len(latest_corr)} matrix, "
            f"window={window}"
        )

        return latest_corr

    def calculate_exponential_correlation(
        self,
        returns: pd.DataFrame,
        halflife: int = 30,
    ) -> pd.DataFrame:
        """
        Calculate exponentially-weighted correlation matrix.

        Recent data gets more weight than older data. Useful for
        adapting quickly to changing market regimes.

        Args:
            returns: Returns DataFrame (time × assets)
            halflife: Halflife for exponential weighting (in days)

        Returns:
            Exponentially-weighted correlation matrix
        """
        if len(returns) < self.min_periods:
            logger.warning("Insufficient data for exponential correlation")
            return pd.DataFrame()

        # Calculate exponentially-weighted covariance
        ewm_cov = returns.ewm(halflife=halflife, min_periods=self.min_periods).cov()

        # Extract latest covariance matrix
        latest_cov = ewm_cov.iloc[-len(returns.columns):]

        # Convert to correlation
        std_devs = np.sqrt(np.diag(latest_cov))
        correlation = latest_cov.values / np.outer(std_devs, std_devs)

        # Clean up numerical errors
        correlation = np.clip(correlation, -1, 1)
        np.fill_diagonal(correlation, 1.0)

        corr_df = pd.DataFrame(
            correlation,
            index=returns.columns,
            columns=returns.columns
        )

        logger.debug(
            f"Exponential correlation calculated with halflife={halflife}"
        )

        return corr_df

    def cluster_by_correlation(
        self,
        correlation_matrix: pd.DataFrame,
        n_clusters: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Cluster assets by correlation using hierarchical clustering.

        Groups highly correlated assets together. Useful for:
        - Identifying redundant positions
        - Diversification analysis
        - Risk grouping

        Args:
            correlation_matrix: Correlation matrix
            n_clusters: Number of clusters (auto-determined if None)

        Returns:
            Dictionary mapping asset names to cluster IDs
        """
        if correlation_matrix.empty or len(correlation_matrix) < 2:
            logger.warning("Cannot cluster: insufficient assets")
            return {}

        try:
            # Convert correlation to distance
            # distance = sqrt(0.5 * (1 - correlation))
            distance_matrix = np.sqrt(0.5 * (1 - correlation_matrix.values))
            np.fill_diagonal(distance_matrix, 0)

            # Hierarchical clustering
            condensed_dist = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_dist, method='average')

            # Determine number of clusters
            if n_clusters is None:
                # Auto-determine using distance threshold
                clusters = fcluster(
                    linkage_matrix,
                    self.cluster_threshold,
                    criterion='distance'
                )
            else:
                clusters = fcluster(
                    linkage_matrix,
                    n_clusters,
                    criterion='maxclust'
                )

            # Create mapping
            cluster_map = dict(zip(correlation_matrix.index, clusters))

            unique_clusters = len(set(clusters))
            logger.info(
                f"Correlation clustering: {len(correlation_matrix)} assets "
                f"→ {unique_clusters} clusters"
            )

            return cluster_map

        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return {}

    def calculate_correlation_penalty(
        self,
        correlation_matrix: pd.DataFrame,
        weights: pd.Series,
    ) -> float:
        """
        Calculate portfolio correlation penalty.

        High correlations reduce diversification benefits.
        This penalty quantifies the diversification loss.

        Penalty = sqrt(w' * C * w)
        where C is correlation matrix and w is weights

        Args:
            correlation_matrix: Correlation matrix
            weights: Portfolio weights

        Returns:
            Correlation penalty (0 to 1, lower is better)
        """
        # Align weights with correlation matrix
        weights = weights.reindex(correlation_matrix.index, fill_value=0)
        weights = weights / weights.sum() if weights.sum() > 0 else weights

        # Calculate weighted correlation
        penalty = np.sqrt(
            weights.values @ correlation_matrix.values @ weights.values
        )

        logger.debug(f"Correlation penalty: {penalty:.3f}")

        return penalty

    def adjust_weights_for_correlation(
        self,
        weights: pd.Series,
        correlation_matrix: pd.DataFrame,
        penalty_factor: float = 0.5,
    ) -> pd.Series:
        """
        Adjust portfolio weights to penalize high correlations.

        Reduces weights of highly correlated assets to improve diversification.

        Args:
            weights: Original portfolio weights
            correlation_matrix: Correlation matrix
            penalty_factor: How aggressively to penalize correlations (0 to 1)

        Returns:
            Adjusted weights with correlation penalty applied
        """
        if penalty_factor <= 0:
            return weights

        # Get clusters
        clusters = self.cluster_by_correlation(correlation_matrix)

        if not clusters:
            return weights

        # Calculate average correlation within each cluster
        adjusted_weights = weights.copy()

        for cluster_id in set(clusters.values()):
            # Get assets in this cluster
            cluster_assets = [
                asset for asset, cid in clusters.items()
                if cid == cluster_id and asset in weights.index
            ]

            if len(cluster_assets) <= 1:
                continue

            # Calculate average pairwise correlation in cluster
            cluster_corr = correlation_matrix.loc[cluster_assets, cluster_assets]
            avg_corr = (cluster_corr.values.sum() - len(cluster_assets)) / (
                len(cluster_assets) * (len(cluster_assets) - 1)
            )

            # Apply penalty if correlation is high
            if avg_corr > self.high_correlation_threshold:
                penalty = 1 - penalty_factor * (avg_corr - self.high_correlation_threshold)
                penalty = max(penalty, 0.5)  # Don't reduce by more than 50%

                # Reduce weights for all assets in this cluster
                for asset in cluster_assets:
                    adjusted_weights[asset] *= penalty

                logger.debug(
                    f"Cluster {cluster_id}: {len(cluster_assets)} assets, "
                    f"avg_corr={avg_corr:.2f}, penalty={penalty:.2f}"
                )

        # Renormalize
        if adjusted_weights.sum() > 0:
            adjusted_weights = adjusted_weights / adjusted_weights.sum()

        logger.info(
            f"Correlation-adjusted weights: max_change="
            f"{(weights - adjusted_weights).abs().max():.3f}"
        )

        return adjusted_weights

    def detect_correlation_regime(
        self,
        returns: pd.DataFrame,
        short_window: int = 20,
        long_window: int = 60,
    ) -> str:
        """
        Detect correlation regime (rising, falling, high, low).

        Useful for adjusting portfolio strategy based on correlation environment.

        Args:
            returns: Historical returns DataFrame
            short_window: Short-term correlation window
            long_window: Long-term correlation window

        Returns:
            Regime string: 'high_rising', 'high_falling', 'low_rising', 'low_falling'
        """
        if len(returns) < long_window:
            logger.warning("Insufficient data for regime detection")
            return 'unknown'

        # Calculate short and long-term average correlations
        short_corr = returns.iloc[-short_window:].corr()
        long_corr = returns.iloc[-long_window:].corr()

        # Average off-diagonal correlations
        n = len(short_corr)
        short_avg = (short_corr.values.sum() - n) / (n * (n - 1))
        long_avg = (long_corr.values.sum() - n) / (n * (n - 1))

        # Determine regime
        is_high = short_avg > self.high_correlation_threshold
        is_rising = short_avg > long_avg

        if is_high and is_rising:
            regime = 'high_rising'
        elif is_high and not is_rising:
            regime = 'high_falling'
        elif not is_high and is_rising:
            regime = 'low_rising'
        else:
            regime = 'low_falling'

        logger.info(
            f"Correlation regime: {regime} "
            f"(short_avg={short_avg:.2f}, long_avg={long_avg:.2f})"
        )

        return regime

    def calculate_diversification_ratio(
        self,
        weights: pd.Series,
        volatilities: pd.Series,
        correlation_matrix: pd.DataFrame,
    ) -> float:
        """
        Calculate portfolio diversification ratio.

        DR = (weighted average volatility) / (portfolio volatility)
        Higher values indicate better diversification.

        Args:
            weights: Portfolio weights
            volatilities: Individual asset volatilities
            correlation_matrix: Correlation matrix

        Returns:
            Diversification ratio (≥ 1, higher is better)
        """
        # Align inputs
        weights = weights.reindex(volatilities.index, fill_value=0)
        weights = weights / weights.sum() if weights.sum() > 0 else weights

        # Weighted average volatility
        weighted_avg_vol = (weights * volatilities).sum()

        # Portfolio volatility
        covariance_matrix = (
            correlation_matrix.values *
            np.outer(volatilities.values, volatilities.values)
        )
        portfolio_variance = weights.values @ covariance_matrix @ weights.values
        portfolio_vol = np.sqrt(portfolio_variance)

        if portfolio_vol <= 0:
            logger.warning("Non-positive portfolio volatility")
            return 1.0

        # Diversification ratio
        div_ratio = weighted_avg_vol / portfolio_vol

        logger.debug(
            f"Diversification ratio: {div_ratio:.2f} "
            f"(weighted_vol={weighted_avg_vol:.3f}, portfolio_vol={portfolio_vol:.3f})"
        )

        return div_ratio

    def monitor_cross_asset_correlations(
        self,
        returns_dict: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Monitor correlations across different asset classes.

        Useful for multi-asset portfolio management and regime detection.

        Args:
            returns_dict: Dictionary of {asset_class: returns_df}
                         e.g., {'equities': equity_returns, 'bonds': bond_returns}

        Returns:
            Cross-asset correlation matrix
        """
        if not returns_dict:
            logger.warning("Empty returns dictionary")
            return pd.DataFrame()

        # Combine all returns into single DataFrame
        all_returns = pd.DataFrame()

        for asset_class, returns_df in returns_dict.items():
            # Use first column or aggregate if multiple
            if isinstance(returns_df, pd.DataFrame):
                if len(returns_df.columns) == 1:
                    all_returns[asset_class] = returns_df.iloc[:, 0]
                else:
                    # Use equal-weighted average
                    all_returns[asset_class] = returns_df.mean(axis=1)
            else:
                all_returns[asset_class] = returns_df

        # Calculate correlation matrix
        cross_corr = all_returns.corr()

        logger.info(
            f"Cross-asset correlations calculated for {len(cross_corr)} asset classes"
        )

        return cross_corr

    def identify_concentration_risk(
        self,
        weights: pd.Series,
        correlation_matrix: pd.DataFrame,
        threshold: float = 0.3,
    ) -> List[Tuple[str, str, float]]:
        """
        Identify concentration risks from high correlations.

        Returns pairs of assets with high correlation and significant combined weight.

        Args:
            weights: Portfolio weights
            correlation_matrix: Correlation matrix
            threshold: Combined weight threshold for risk identification

        Returns:
            List of (asset1, asset2, correlation) tuples indicating concentration risks
        """
        risks = []

        assets = correlation_matrix.index.tolist()

        for i, asset1 in enumerate(assets):
            for asset2 in assets[i + 1:]:
                corr = correlation_matrix.loc[asset1, asset2]

                # High correlation and significant combined weight
                if corr > self.high_correlation_threshold:
                    weight1 = weights.get(asset1, 0)
                    weight2 = weights.get(asset2, 0)
                    combined_weight = weight1 + weight2

                    if combined_weight > threshold:
                        risks.append((asset1, asset2, corr))

        if risks:
            logger.warning(
                f"Identified {len(risks)} concentration risks from high correlations"
            )
        else:
            logger.info("No significant concentration risks identified")

        return risks

    def calculate_regime_adjusted_correlation(
        self,
        returns: pd.DataFrame,
        regime: str,
        crisis_multiplier: float = 1.5,
    ) -> pd.DataFrame:
        """
        Adjust correlation matrix based on market regime.

        In crisis regimes, correlations tend to increase (correlation breakdown).
        This method adjusts historical correlations to account for regime effects.

        Args:
            returns: Historical returns
            regime: Market regime ('normal', 'stress', 'crisis')
            crisis_multiplier: How much to increase correlations in crisis

        Returns:
            Regime-adjusted correlation matrix
        """
        # Calculate base correlation
        correlation_matrix = returns.corr()

        if regime == 'normal':
            return correlation_matrix

        # Adjust correlations for stress/crisis
        adjusted_corr = correlation_matrix.copy()

        if regime == 'stress':
            multiplier = 1.2
        elif regime == 'crisis':
            multiplier = crisis_multiplier
        else:
            multiplier = 1.0

        # Increase off-diagonal correlations
        for i in range(len(adjusted_corr)):
            for j in range(i + 1, len(adjusted_corr)):
                original_corr = adjusted_corr.iloc[i, j]
                # Move correlation towards 1 (or -1 for negative correlations)
                adjusted_corr.iloc[i, j] = original_corr * multiplier
                adjusted_corr.iloc[j, i] = adjusted_corr.iloc[i, j]

        # Clip to valid range
        adjusted_corr = adjusted_corr.clip(-1, 1)
        np.fill_diagonal(adjusted_corr.values, 1.0)

        logger.info(
            f"Regime-adjusted correlation: {regime} regime with {multiplier}x multiplier"
        )

        return adjusted_corr
