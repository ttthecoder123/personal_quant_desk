"""
Triple-barrier labeling implementation
Based on López de Prado's "Advances in Financial Machine Learning", Chapter 3
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from loguru import logger


class TripleBarrierLabeling:
    """
    López de Prado's triple-barrier labeling method
    Chapter 3 of Advances in Financial Machine Learning

    The triple-barrier method labels observations based on which barrier
    is touched first:
    1. Upper barrier (profit taking)
    2. Lower barrier (stop loss)
    3. Vertical barrier (time limit)
    """

    def __init__(self, prices: pd.Series, events: pd.DataFrame):
        """
        Initialize with price series and events DataFrame

        Args:
            prices: Price series (typically Close prices)
            events: DataFrame with columns:
                - t1: Vertical barrier (max holding period)
                - trgt: Target price move (volatility units)
                - side: Side of trade (1: long, -1: short, 0: both)
        """
        self.prices = prices
        self.events = events
        logger.info(f"Initialized TripleBarrierLabeling with {len(events)} events")

    def apply_triple_barrier(self, pt_sl: List[float] = [1, 1]) -> pd.DataFrame:
        """
        Apply triple-barrier method to label events

        Args:
            pt_sl: [profit_taking, stop_loss] as multipliers of target
                   pt_sl[0]: profit taking barrier width
                   pt_sl[1]: stop loss barrier width

        Returns:
            DataFrame with barrier touches and labels
        """
        events = self.events.copy()
        touches = pd.DataFrame(index=events.index)

        logger.info(f"Applying triple-barrier with pt_sl={pt_sl}")

        # Calculate barriers for each event
        for loc, event in events.iterrows():
            # Get price path from event to max holding period
            t1 = event['t1']
            target = event['trgt']
            side = event.get('side', 0)

            # Price path
            price_path = self.prices[loc:t1]
            if len(price_path) < 2:
                continue

            # Calculate returns from entry
            returns = (price_path / self.prices[loc] - 1)

            # Define barriers
            if side == 0:  # No side info, symmetric barriers
                upper_barrier = target * pt_sl[0]
                lower_barrier = -target * pt_sl[1]
            elif side == 1:  # Long position
                upper_barrier = target * pt_sl[0]
                lower_barrier = -target * pt_sl[1]
            else:  # Short position
                upper_barrier = target * pt_sl[1]
                lower_barrier = -target * pt_sl[0]

            # Find first barrier touch
            upper_touch = returns[returns > upper_barrier]
            lower_touch = returns[returns < lower_barrier]

            touches.loc[loc, 'pt'] = upper_touch.index.min() if len(upper_touch) > 0 else pd.NaT
            touches.loc[loc, 'sl'] = lower_touch.index.min() if len(lower_touch) > 0 else pd.NaT
            touches.loc[loc, 't1'] = t1

        # Determine which barrier was touched first
        touches['first_touch'] = touches[['pt', 'sl', 't1']].min(axis=1)

        # Generate labels
        labels = pd.Series(index=events.index, dtype=float)
        for idx in touches.index:
            first = touches.loc[idx, 'first_touch']
            if pd.isna(first):
                continue

            # Return at barrier touch
            ret = self.prices[first] / self.prices[idx] - 1

            # Determine label based on side
            side = events.loc[idx, 'side'] if 'side' in events.columns else 0
            if side == 0:
                labels[idx] = np.sign(ret)  # Sign of return
            else:
                labels[idx] = np.sign(ret * side)  # Adjusted for side

        # Combine results
        result = pd.DataFrame({
            'first_touch_time': touches['first_touch'],
            'barrier_type': self._identify_barrier(touches),
            'return': labels.apply(lambda x: self.prices[touches.loc[x, 'first_touch']] /
                                   self.prices[x] - 1 if x in touches.index and
                                   not pd.isna(touches.loc[x, 'first_touch']) else np.nan),
            'label': labels
        }, index=events.index)

        logger.info(f"Generated {len(result.dropna())} valid labels")
        logger.info(f"Label distribution: {result['label'].value_counts().to_dict()}")

        return result

    def _identify_barrier(self, touches: pd.DataFrame) -> pd.Series:
        """Identify which barrier was touched first"""
        barrier_type = pd.Series(index=touches.index, dtype=str)

        for idx in touches.index:
            first = touches.loc[idx, 'first_touch']
            if pd.isna(first):
                barrier_type[idx] = 'none'
            elif first == touches.loc[idx, 'pt']:
                barrier_type[idx] = 'profit_take'
            elif first == touches.loc[idx, 'sl']:
                barrier_type[idx] = 'stop_loss'
            else:
                barrier_type[idx] = 'time_out'

        return barrier_type

    @staticmethod
    def get_daily_volatility(prices: pd.Series, span: int = 100) -> pd.Series:
        """
        Compute daily volatility using exponentially weighted moving average

        Args:
            prices: Price series
            span: Span for EWMA calculation

        Returns:
            Daily volatility series
        """
        returns = prices.pct_change()
        volatility = returns.ewm(span=span, min_periods=span).std()
        return volatility

    def add_vertical_barrier(self, timestamps: pd.DatetimeIndex,
                           num_days: int = 10) -> pd.Series:
        """
        Add vertical barrier (maximum holding period)

        Args:
            timestamps: Event timestamps
            num_days: Maximum holding period in days

        Returns:
            Series of vertical barrier timestamps
        """
        t1 = pd.Series(index=timestamps, dtype='datetime64[ns]')

        for timestamp in timestamps:
            # Find the timestamp num_days ahead
            try:
                future_idx = self.prices.index.get_indexer([timestamp], method='bfill')[0]
                max_idx = min(future_idx + num_days, len(self.prices) - 1)
                t1[timestamp] = self.prices.index[max_idx]
            except Exception as e:
                logger.warning(f"Could not set barrier for {timestamp}: {e}")
                continue

        return t1


def apply_triple_barrier_labeling(
    prices: pd.Series,
    events_idx: pd.DatetimeIndex,
    volatility: pd.Series,
    pt_sl: List[float] = [1, 1],
    num_days: int = 10,
    volatility_multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Convenience function to apply triple-barrier labeling

    Args:
        prices: Price series
        events_idx: Event timestamps
        volatility: Volatility series
        pt_sl: [profit_taking, stop_loss] multipliers
        num_days: Maximum holding period
        volatility_multiplier: Multiplier for target calculation

    Returns:
        DataFrame with labels
    """
    # Create events DataFrame
    events = pd.DataFrame(index=events_idx)
    events['trgt'] = volatility.loc[events_idx] * volatility_multiplier

    # Create labeler instance
    labeler = TripleBarrierLabeling(prices, events)

    # Add vertical barrier
    events['t1'] = labeler.add_vertical_barrier(events_idx, num_days=num_days)
    events['side'] = 0  # No side information

    # Apply labeling
    labels = labeler.apply_triple_barrier(pt_sl=pt_sl)

    return labels
