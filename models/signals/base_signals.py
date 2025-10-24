"""
Rule-based trading signals (primary model)
"""
import pandas as pd
import numpy as np
from typing import Optional, Dict
from loguru import logger


class BaseSignals:
    """
    Rule-based trading signals (primary model)

    Implements classical technical analysis and quantitative trading rules
    as baseline signals for the meta-labeling framework.
    """

    def __init__(self, features: pd.DataFrame, prices: pd.Series):
        """
        Initialize with features and prices

        Args:
            features: Feature DataFrame
            prices: Price series
        """
        self.features = features
        self.prices = prices
        logger.info(f"Initialized BaseSignals with {len(features)} observations")

    def momentum_signal(self, lookback: int = 20, threshold: float = 1.5) -> pd.Series:
        """
        Momentum-based signal using normalized returns

        Args:
            lookback: Lookback period for momentum calculation
            threshold: Z-score threshold for signal generation

        Returns:
            Signal series (-1, 0, 1)
        """
        logger.info(f"Generating momentum signal with lookback={lookback}")

        # Get momentum feature
        momentum_col = f'return_{lookback}d'
        if momentum_col not in self.features.columns:
            logger.warning(f"Feature {momentum_col} not found, using price returns")
            momentum = self.prices.pct_change(lookback)
        else:
            momentum = self.features[momentum_col]

        # Z-score normalization
        z_score = (momentum - momentum.rolling(252).mean()) / momentum.rolling(252).std()

        # Generate signals
        signals = pd.Series(0, index=self.features.index)
        signals[z_score > threshold] = 1  # Strong momentum up
        signals[z_score < -threshold] = -1  # Strong momentum down

        logger.info(f"Momentum signals: {signals.value_counts().to_dict()}")

        return signals

    def mean_reversion_signal(self, bb_threshold: float = 0.2,
                             rsi_threshold: float = 30) -> pd.Series:
        """
        Mean reversion signal using Bollinger Bands and RSI

        Args:
            bb_threshold: Bollinger Band position threshold
            rsi_threshold: RSI threshold

        Returns:
            Signal series (-1, 0, 1)
        """
        logger.info(f"Generating mean reversion signal")

        signals = pd.Series(0, index=self.features.index)

        # Check if required features exist
        has_bb = 'bb_position' in self.features.columns
        has_rsi = 'rsi_14' in self.features.columns

        if not has_bb or not has_rsi:
            logger.warning(f"Missing features (bb_position: {has_bb}, rsi_14: {has_rsi})")
            return signals

        bb_position = self.features['bb_position']
        rsi = self.features['rsi_14']

        # Oversold: Low BB position + Low RSI
        oversold = (bb_position < bb_threshold) & (rsi < rsi_threshold)
        signals[oversold] = 1

        # Overbought: High BB position + High RSI
        overbought = (bb_position > (1 - bb_threshold)) & (rsi > (100 - rsi_threshold))
        signals[overbought] = -1

        logger.info(f"Mean reversion signals: {signals.value_counts().to_dict()}")

        return signals

    def volatility_breakout_signal(self, vol_quantile: float = 0.2,
                                  return_threshold: float = 0.02) -> pd.Series:
        """
        Volatility breakout signal (squeeze and release)

        Args:
            vol_quantile: Quantile threshold for low volatility
            return_threshold: Return threshold for breakout

        Returns:
            Signal series (-1, 0, 1)
        """
        logger.info(f"Generating volatility breakout signal")

        signals = pd.Series(0, index=self.features.index)

        # Check for required features
        has_bb_bw = 'bb_bandwidth' in self.features.columns
        has_atr = 'atr_ratio' in self.features.columns
        has_returns = 'return_20d' in self.features.columns

        if not (has_bb_bw and has_atr):
            logger.warning(f"Missing volatility features")
            return signals

        bb_bandwidth = self.features['bb_bandwidth']
        atr_ratio = self.features['atr_ratio']

        # Low volatility squeeze
        squeeze = (bb_bandwidth < bb_bandwidth.quantile(vol_quantile)) & \
                 (atr_ratio < atr_ratio.quantile(vol_quantile * 1.5))

        # Price breakout
        if has_returns:
            returns_20d = self.features['return_20d']
        else:
            returns_20d = self.prices.pct_change(20)

        # Generate signals on breakout from squeeze
        signals[squeeze & (returns_20d > return_threshold)] = 1
        signals[squeeze & (returns_20d < -return_threshold)] = -1

        logger.info(f"Volatility breakout signals: {signals.value_counts().to_dict()}")

        return signals

    def trend_following_signal(self, fast_period: int = 50,
                              slow_period: int = 200) -> pd.Series:
        """
        Trend following signal using moving average crossover

        Args:
            fast_period: Fast MA period
            slow_period: Slow MA period

        Returns:
            Signal series (-1, 0, 1)
        """
        logger.info(f"Generating trend following signal")

        signals = pd.Series(0, index=self.features.index)

        # Calculate moving averages
        fast_ma_col = f'sma_{fast_period}'
        slow_ma_col = f'sma_{slow_period}'

        if fast_ma_col in self.features.columns and slow_ma_col in self.features.columns:
            fast_ma = self.features[fast_ma_col]
            slow_ma = self.features[slow_ma_col]
        else:
            logger.warning("SMA features not found, calculating from prices")
            fast_ma = self.prices.rolling(fast_period).mean()
            slow_ma = self.prices.rolling(slow_period).mean()

        # Generate signals
        signals[fast_ma > slow_ma] = 1
        signals[fast_ma < slow_ma] = -1

        logger.info(f"Trend following signals: {signals.value_counts().to_dict()}")

        return signals

    def regime_adaptive_signal(self) -> pd.Series:
        """
        Regime-adaptive signal that switches between momentum and mean reversion

        Returns:
            Signal series (-1, 0, 1)
        """
        logger.info("Generating regime-adaptive signal")

        signals = pd.Series(0, index=self.features.index)

        # Identify regime using volatility and trend strength
        if 'atr_ratio' in self.features.columns:
            volatility = self.features['atr_ratio']
            high_vol_regime = volatility > volatility.median()
        else:
            high_vol_regime = pd.Series(False, index=self.features.index)

        # In high volatility, use mean reversion
        # In low volatility, use momentum
        momentum_sig = self.momentum_signal()
        mean_rev_sig = self.mean_reversion_signal()

        signals[high_vol_regime] = mean_rev_sig[high_vol_regime]
        signals[~high_vol_regime] = momentum_sig[~high_vol_regime]

        logger.info(f"Regime-adaptive signals: {signals.value_counts().to_dict()}")

        return signals

    def composite_signal(self, weights: Optional[Dict[str, float]] = None) -> pd.Series:
        """
        Combine multiple signals with voting

        Args:
            weights: Dictionary of signal weights

        Returns:
            Composite signal series (-1, 0, 1)
        """
        logger.info("Generating composite signal")

        if weights is None:
            weights = {
                'momentum': 0.3,
                'mean_reversion': 0.25,
                'volatility': 0.25,
                'trend': 0.2
            }

        signals = pd.DataFrame(index=self.features.index)
        signals['momentum'] = self.momentum_signal()
        signals['mean_reversion'] = self.mean_reversion_signal()
        signals['volatility'] = self.volatility_breakout_signal()
        signals['trend'] = self.trend_following_signal()

        # Weighted voting
        composite = sum(signals[col] * weights[col] for col in signals.columns if col in weights)

        # Discretize with threshold
        final_signals = pd.Series(0, index=self.features.index)
        final_signals[composite > 0.5] = 1
        final_signals[composite < -0.5] = -1

        logger.info(f"Composite signals: {final_signals.value_counts().to_dict()}")

        return final_signals

    def evaluate_signals(self, signals: pd.Series,
                        forward_returns: Optional[pd.Series] = None) -> Dict:
        """
        Evaluate signal quality

        Args:
            signals: Signal series
            forward_returns: Forward returns for evaluation

        Returns:
            Dictionary of metrics
        """
        if forward_returns is None:
            forward_returns = self.prices.pct_change().shift(-1)

        # Align signals and returns
        common_idx = signals.index.intersection(forward_returns.index)
        sig = signals.loc[common_idx]
        ret = forward_returns.loc[common_idx]

        # Calculate metrics
        signal_returns = sig * ret

        metrics = {
            'n_signals': (sig != 0).sum(),
            'hit_rate': (signal_returns[sig != 0] > 0).mean(),
            'mean_return': signal_returns[sig != 0].mean(),
            'sharpe': signal_returns.mean() / signal_returns.std() * np.sqrt(252) if signal_returns.std() > 0 else 0,
            'long_short_ratio': (sig == 1).sum() / (sig == -1).sum() if (sig == -1).sum() > 0 else np.inf
        }

        logger.info(f"Signal metrics: {metrics}")

        return metrics
