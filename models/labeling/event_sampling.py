"""
Event sampling methods
Based on López de Prado's "Advances in Financial Machine Learning", Chapter 2
"""
import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger


class EventSampling:
    """
    CUSUM event sampling (López de Prado Ch. 2)

    Event-driven sampling methods that detect structural breaks
    and significant market events rather than sampling at fixed intervals.
    """

    def __init__(self, prices: pd.Series):
        """
        Initialize with price series

        Args:
            prices: Price series
        """
        self.prices = prices
        self.returns = prices.pct_change().dropna()
        logger.info(f"Initialized EventSampling with {len(prices)} prices")

    def cusum_filter(self, threshold: float = 0.01) -> pd.DatetimeIndex:
        """
        CUSUM filter for event detection

        The CUSUM filter detects events when cumulative sum of deviations
        exceeds a threshold, indicating a significant directional move.

        Args:
            threshold: Threshold for event detection (as fraction of volatility)

        Returns:
            DatetimeIndex of detected events
        """
        logger.info(f"Running CUSUM filter with threshold={threshold}")

        # Initialize
        events = []
        s_pos = 0
        s_neg = 0

        # Standardize returns
        returns_std = self.returns.dropna()
        returns_mean = returns_std.mean()
        returns_vol = returns_std.std()

        if returns_vol == 0:
            logger.warning("Zero volatility detected, cannot apply CUSUM filter")
            return pd.DatetimeIndex([])

        for idx, ret in returns_std.items():
            # Update positive and negative CUSUM
            s_pos = max(0, s_pos + ret - returns_mean)
            s_neg = min(0, s_neg + ret - returns_mean)

            # Check if threshold is breached
            if s_pos > threshold * returns_vol:
                events.append(idx)
                s_pos = 0
            elif abs(s_neg) > threshold * returns_vol:
                events.append(idx)
                s_neg = 0

        logger.info(f"CUSUM filter detected {len(events)} events")

        return pd.DatetimeIndex(events)

    def symmetric_cusum(self, threshold_up: float = 0.01,
                       threshold_down: float = 0.01) -> pd.DatetimeIndex:
        """
        Symmetric CUSUM filter with separate up/down thresholds

        Args:
            threshold_up: Threshold for upward events
            threshold_down: Threshold for downward events

        Returns:
            DatetimeIndex of detected events
        """
        logger.info(f"Running symmetric CUSUM with thresholds up={threshold_up}, down={threshold_down}")

        events = []
        s_pos = 0
        s_neg = 0

        returns_vol = self.returns.std()

        for idx, ret in self.returns.items():
            s_pos = max(0, s_pos + ret)
            s_neg = min(0, s_neg + ret)

            if s_pos > threshold_up * returns_vol:
                events.append(idx)
                s_pos = 0
            elif abs(s_neg) > threshold_down * returns_vol:
                events.append(idx)
                s_neg = 0

        logger.info(f"Symmetric CUSUM detected {len(events)} events")

        return pd.DatetimeIndex(events)

    def entropy_filter(self, lookback: int = 100,
                      threshold: float = 0.95) -> pd.DatetimeIndex:
        """
        Entropy-based event sampling

        Detects periods of high uncertainty/entropy in the return distribution.

        Args:
            lookback: Lookback period for entropy calculation
            threshold: Percentile threshold for high entropy

        Returns:
            DatetimeIndex of high-entropy events
        """
        logger.info(f"Running entropy filter with lookback={lookback}, threshold={threshold}")

        # Calculate rolling entropy
        def entropy(x):
            # Discretize returns into bins
            hist, _ = np.histogram(x.dropna(), bins=10)
            probs = hist / len(x.dropna())
            probs = probs[probs > 0]
            if len(probs) == 0:
                return 0
            return -np.sum(probs * np.log(probs))

        rolling_entropy = self.returns.rolling(lookback).apply(entropy, raw=False)

        # Select high entropy periods
        threshold_value = rolling_entropy.quantile(threshold)
        events = rolling_entropy[rolling_entropy > threshold_value].index

        logger.info(f"Entropy filter detected {len(events)} high-entropy events")

        return pd.DatetimeIndex(events)

    def volatility_filter(self, lookback: int = 20,
                         threshold: float = 2.0) -> pd.DatetimeIndex:
        """
        Detect high volatility events

        Args:
            lookback: Lookback period for volatility calculation
            threshold: Multiple of rolling mean volatility

        Returns:
            DatetimeIndex of high volatility events
        """
        logger.info(f"Running volatility filter with lookback={lookback}, threshold={threshold}")

        # Calculate rolling volatility
        rolling_vol = self.returns.rolling(lookback).std()
        vol_mean = rolling_vol.mean()

        # Detect high volatility events
        events = rolling_vol[rolling_vol > threshold * vol_mean].index

        logger.info(f"Volatility filter detected {len(events)} high-volatility events")

        return pd.DatetimeIndex(events)

    def structural_break_filter(self, lookback: int = 50,
                               threshold: float = 2.0) -> pd.DatetimeIndex:
        """
        Detect structural breaks using rolling statistics

        Args:
            lookback: Lookback period
            threshold: Z-score threshold for detecting breaks

        Returns:
            DatetimeIndex of structural break events
        """
        logger.info(f"Running structural break filter with lookback={lookback}")

        # Calculate rolling mean and std
        rolling_mean = self.returns.rolling(lookback).mean()
        rolling_std = self.returns.rolling(lookback).std()

        # Calculate z-score of current return vs historical
        z_score = (self.returns - rolling_mean) / rolling_std

        # Detect significant deviations
        events = z_score[abs(z_score) > threshold].index

        logger.info(f"Structural break filter detected {len(events)} events")

        return pd.DatetimeIndex(events)

    def combine_filters(self, methods: Optional[list] = None,
                       min_votes: int = 2) -> pd.DatetimeIndex:
        """
        Combine multiple event detection methods

        Args:
            methods: List of filter methods to combine
            min_votes: Minimum number of methods that must agree

        Returns:
            DatetimeIndex of events with sufficient votes
        """
        if methods is None:
            methods = ['cusum', 'entropy', 'volatility']

        logger.info(f"Combining filters: {methods} with min_votes={min_votes}")

        all_events = []

        if 'cusum' in methods:
            all_events.append(self.cusum_filter())
        if 'entropy' in methods:
            all_events.append(self.entropy_filter())
        if 'volatility' in methods:
            all_events.append(self.volatility_filter())
        if 'structural' in methods:
            all_events.append(self.structural_break_filter())

        # Count votes for each timestamp
        vote_counts = pd.Series(dtype=int)
        for events in all_events:
            for event in events:
                if event in vote_counts:
                    vote_counts[event] += 1
                else:
                    vote_counts[event] = 1

        # Select events with sufficient votes
        selected_events = vote_counts[vote_counts >= min_votes].index

        logger.info(f"Combined filter detected {len(selected_events)} events")

        return pd.DatetimeIndex(selected_events).sort_values()


def detect_events(prices: pd.Series,
                 method: str = 'cusum',
                 **kwargs) -> pd.DatetimeIndex:
    """
    Convenience function for event detection

    Args:
        prices: Price series
        method: Detection method ('cusum', 'entropy', 'volatility', 'combined')
        **kwargs: Additional arguments for the filter

    Returns:
        DatetimeIndex of detected events
    """
    sampler = EventSampling(prices)

    if method == 'cusum':
        return sampler.cusum_filter(**kwargs)
    elif method == 'entropy':
        return sampler.entropy_filter(**kwargs)
    elif method == 'volatility':
        return sampler.volatility_filter(**kwargs)
    elif method == 'structural':
        return sampler.structural_break_filter(**kwargs)
    elif method == 'combined':
        return sampler.combine_filters(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
