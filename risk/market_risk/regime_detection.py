"""
Market Regime Detection using Hidden Markov Models and statistical methods.

Implements institutional-grade regime detection following:
- Hidden Markov Models for state inference
- Volatility regime classification
- Trend vs range-bound detection
- Regime transition analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from scipy import stats
from loguru import logger

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    logger.warning("hmmlearn not available - HMM features disabled")
    HMM_AVAILABLE = False


@dataclass
class RegimeState:
    """Container for regime state information."""
    regime_id: int
    regime_name: str
    volatility: float
    mean_return: float
    probability: float
    duration: int
    timestamp: datetime


@dataclass
class RegimeMetrics:
    """Container for regime metrics."""
    current_regime: RegimeState
    regime_probabilities: np.ndarray
    transition_matrix: np.ndarray
    expected_duration: float
    regime_stability: float
    regime_history: List[int]


class RegimeDetector:
    """
    Advanced market regime detection using Hidden Markov Models.

    Identifies market regimes including:
    - Volatility regimes (low/medium/high)
    - Trend vs range-bound markets
    - Risk-on vs risk-off environments
    - Regime transitions and persistence
    """

    def __init__(self,
                 n_regimes: int = 3,
                 volatility_window: int = 20,
                 transition_threshold: float = 0.7,
                 min_regime_duration: int = 5):
        """
        Initialize regime detector.

        Args:
            n_regimes: Number of hidden states (regimes)
            volatility_window: Window for volatility calculation
            transition_threshold: Probability threshold for regime change
            min_regime_duration: Minimum bars for stable regime
        """
        self.n_regimes = n_regimes
        self.volatility_window = volatility_window
        self.transition_threshold = transition_threshold
        self.min_regime_duration = min_regime_duration

        # HMM model
        self.hmm_model = None
        self.is_fitted = False

        # Regime tracking
        self.current_regime = None
        self.regime_history = []
        self.regime_start_idx = 0

        # Regime characteristics
        self.regime_names = {
            0: "Low Volatility / Trending",
            1: "Medium Volatility / Mixed",
            2: "High Volatility / Crisis"
        }

        logger.info(f"Initialized RegimeDetector with {n_regimes} regimes")

    def fit(self, returns: pd.Series, n_iter: int = 100) -> 'RegimeDetector':
        """
        Fit Hidden Markov Model to return series.

        Args:
            returns: Time series of returns
            n_iter: Number of EM iterations

        Returns:
            self
        """
        if not HMM_AVAILABLE:
            logger.error("HMM not available - cannot fit model")
            return self

        try:
            # Prepare features for HMM
            features = self._prepare_features(returns)

            # Initialize and fit Gaussian HMM
            self.hmm_model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=n_iter,
                random_state=42
            )

            self.hmm_model.fit(features)
            self.is_fitted = True

            logger.success(f"HMM model fitted with {n_iter} iterations")
            logger.info(f"Converged: {self.hmm_model.monitor_.converged}")

            # Log regime statistics
            self._log_regime_statistics(features)

        except Exception as e:
            logger.error(f"Error fitting HMM model: {str(e)}")

        return self

    def predict_regime(self, returns: pd.Series) -> RegimeMetrics:
        """
        Predict current market regime.

        Args:
            returns: Time series of returns

        Returns:
            RegimeMetrics with current regime information
        """
        if not self.is_fitted:
            logger.warning("Model not fitted - using statistical regime detection")
            return self._statistical_regime_detection(returns)

        try:
            # Prepare features
            features = self._prepare_features(returns)

            # Predict hidden states
            hidden_states = self.hmm_model.predict(features)

            # Get state probabilities
            posterior_probs = self.hmm_model.predict_proba(features)

            # Get current regime
            current_state = hidden_states[-1]
            current_prob = posterior_probs[-1, current_state]

            # Calculate regime metrics
            regime_duration = self._calculate_regime_duration(hidden_states)

            # Get volatility and return for current regime
            recent_returns = returns.iloc[-self.volatility_window:]
            current_vol = recent_returns.std() * np.sqrt(252)
            current_return = recent_returns.mean() * 252

            # Create regime state
            regime_state = RegimeState(
                regime_id=int(current_state),
                regime_name=self.regime_names.get(current_state, f"Regime {current_state}"),
                volatility=float(current_vol),
                mean_return=float(current_return),
                probability=float(current_prob),
                duration=regime_duration,
                timestamp=datetime.now()
            )

            # Calculate stability
            stability = self._calculate_regime_stability(posterior_probs[-self.volatility_window:])

            # Expected duration
            expected_duration = self._calculate_expected_duration(current_state)

            metrics = RegimeMetrics(
                current_regime=regime_state,
                regime_probabilities=posterior_probs[-1],
                transition_matrix=self.hmm_model.transmat_,
                expected_duration=expected_duration,
                regime_stability=stability,
                regime_history=hidden_states.tolist()
            )

            # Update tracking
            self.current_regime = current_state
            self.regime_history = hidden_states.tolist()

            return metrics

        except Exception as e:
            logger.error(f"Error predicting regime: {str(e)}")
            return self._statistical_regime_detection(returns)

    def detect_volatility_regime(self, returns: pd.Series) -> Dict[str, float]:
        """
        Classify volatility regime without HMM.

        Args:
            returns: Time series of returns

        Returns:
            Dictionary with regime classification
        """
        # Calculate realized volatility
        vol_short = returns.iloc[-self.volatility_window:].std() * np.sqrt(252)
        vol_long = returns.iloc[-252:].std() * np.sqrt(252) if len(returns) >= 252 else vol_short

        # Calculate historical percentiles
        rolling_vol = returns.rolling(self.volatility_window).std() * np.sqrt(252)
        vol_percentile = stats.percentileofscore(rolling_vol.dropna(), vol_short) / 100

        # Classify regime
        if vol_percentile < 0.33:
            regime = "low"
            regime_id = 0
        elif vol_percentile < 0.67:
            regime = "medium"
            regime_id = 1
        else:
            regime = "high"
            regime_id = 2

        # Volatility of volatility
        vol_of_vol = rolling_vol.iloc[-20:].std() if len(rolling_vol) >= 20 else 0

        return {
            'regime': regime,
            'regime_id': regime_id,
            'current_vol': vol_short,
            'long_vol': vol_long,
            'vol_percentile': vol_percentile,
            'vol_of_vol': vol_of_vol,
            'vol_regime_strength': abs(vol_percentile - 0.5) * 2  # 0-1 scale
        }

    def detect_trend_regime(self, prices: pd.Series) -> Dict[str, float]:
        """
        Detect trend vs range-bound regime.

        Args:
            prices: Time series of prices

        Returns:
            Dictionary with trend regime metrics
        """
        # Moving averages
        ma_20 = prices.rolling(20).mean()
        ma_50 = prices.rolling(50).mean()
        ma_200 = prices.rolling(200).mean() if len(prices) >= 200 else ma_50

        # Current price relative to MAs
        price_to_ma20 = (prices.iloc[-1] - ma_20.iloc[-1]) / ma_20.iloc[-1]
        price_to_ma50 = (prices.iloc[-1] - ma_50.iloc[-1]) / ma_50.iloc[-1]

        # Trend strength using linear regression
        recent_prices = prices.iloc[-20:].values
        x = np.arange(len(recent_prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_prices)
        trend_strength = r_value ** 2  # R-squared
        trend_direction = np.sign(slope)

        # ADX-like calculation for trend quality
        high_low_range = prices.rolling(20).max() - prices.rolling(20).min()
        trend_quality = abs(price_to_ma20) / (high_low_range.iloc[-1] / prices.iloc[-1])

        # Classify regime
        if trend_strength > 0.6 and abs(price_to_ma20) > 0.05:
            regime = "trending"
            regime_id = 1
        else:
            regime = "range_bound"
            regime_id = 0

        return {
            'regime': regime,
            'regime_id': regime_id,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'trend_quality': trend_quality,
            'price_to_ma20': price_to_ma20,
            'price_to_ma50': price_to_ma50,
            'ma_alignment': 1 if ma_20.iloc[-1] > ma_50.iloc[-1] else -1
        }

    def detect_regime_change(self, returns: pd.Series) -> Dict[str, any]:
        """
        Detect potential regime changes.

        Args:
            returns: Time series of returns

        Returns:
            Dictionary with regime change signals
        """
        if not self.is_fitted:
            return {'regime_change': False, 'confidence': 0.0}

        try:
            features = self._prepare_features(returns)
            posterior_probs = self.hmm_model.predict_proba(features)

            # Recent probability evolution
            recent_probs = posterior_probs[-10:]

            # Check for regime transition
            current_regime = np.argmax(recent_probs[-1])
            prev_regime = np.argmax(recent_probs[-2])

            regime_change = current_regime != prev_regime

            # Transition confidence
            confidence = recent_probs[-1, current_regime]

            # Probability volatility (instability indicator)
            prob_volatility = np.std(recent_probs, axis=0).mean()

            return {
                'regime_change': regime_change,
                'current_regime': int(current_regime),
                'previous_regime': int(prev_regime),
                'confidence': float(confidence),
                'probability_volatility': float(prob_volatility),
                'stable': confidence > self.transition_threshold and prob_volatility < 0.2
            }

        except Exception as e:
            logger.error(f"Error detecting regime change: {str(e)}")
            return {'regime_change': False, 'confidence': 0.0}

    def calculate_regime_persistence(self, hidden_states: np.ndarray) -> Dict[int, float]:
        """
        Calculate average persistence (duration) for each regime.

        Args:
            hidden_states: Array of regime states

        Returns:
            Dictionary mapping regime to average duration
        """
        persistence = {}

        for regime in range(self.n_regimes):
            # Find regime runs
            is_regime = (hidden_states == regime).astype(int)
            changes = np.diff(np.concatenate([[0], is_regime, [0]]))

            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]

            if len(starts) > 0:
                durations = ends - starts
                persistence[regime] = float(np.mean(durations))
            else:
                persistence[regime] = 0.0

        return persistence

    def generate_regime_report(self, returns: pd.Series, prices: pd.Series) -> Dict:
        """
        Generate comprehensive regime analysis report.

        Args:
            returns: Time series of returns
            prices: Time series of prices

        Returns:
            Dictionary with complete regime analysis
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'volatility_regime': self.detect_volatility_regime(returns),
            'trend_regime': self.detect_trend_regime(prices),
        }

        # Add HMM-based analysis if available
        if self.is_fitted:
            regime_metrics = self.predict_regime(returns)
            report['hmm_regime'] = {
                'regime_id': regime_metrics.current_regime.regime_id,
                'regime_name': regime_metrics.current_regime.regime_name,
                'probability': regime_metrics.current_regime.probability,
                'duration': regime_metrics.current_regime.duration,
                'stability': regime_metrics.regime_stability,
                'expected_duration': regime_metrics.expected_duration
            }

            report['regime_change_detection'] = self.detect_regime_change(returns)

            # Transition probabilities
            current_regime = regime_metrics.current_regime.regime_id
            report['transition_probabilities'] = {
                f'to_regime_{i}': float(regime_metrics.transition_matrix[current_regime, i])
                for i in range(self.n_regimes)
            }

        return report

    def get_regime_adjusted_risk(self, base_risk: float, returns: pd.Series) -> float:
        """
        Adjust risk allocation based on current regime.

        Args:
            base_risk: Base risk allocation
            returns: Time series of returns

        Returns:
            Regime-adjusted risk allocation
        """
        vol_regime = self.detect_volatility_regime(returns)

        # Risk multipliers by regime
        regime_multipliers = {
            0: 1.2,   # Low vol - increase risk
            1: 1.0,   # Medium vol - maintain
            2: 0.5    # High vol - reduce risk
        }

        regime_id = vol_regime['regime_id']
        multiplier = regime_multipliers.get(regime_id, 1.0)

        # Additional adjustment for trend regime
        trend_regime = self.detect_trend_regime(
            pd.Series(np.cumsum(returns), index=returns.index)
        )

        if trend_regime['regime'] == 'trending' and trend_regime['trend_strength'] > 0.7:
            multiplier *= 1.1  # Slight increase in trending markets

        adjusted_risk = base_risk * multiplier

        logger.debug(f"Risk adjusted: {base_risk:.3f} -> {adjusted_risk:.3f} "
                    f"(regime={regime_id}, multiplier={multiplier:.2f})")

        return adjusted_risk

    # Private helper methods

    def _prepare_features(self, returns: pd.Series) -> np.ndarray:
        """Prepare feature matrix for HMM."""
        returns_clean = returns.fillna(0)

        # Feature engineering
        features = pd.DataFrame(index=returns.index)
        features['returns'] = returns_clean
        features['abs_returns'] = returns_clean.abs()
        features['log_vol'] = np.log(returns_clean.rolling(5).std() + 1e-8)

        # Standardize features
        feature_matrix = features.fillna(0).values
        feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-8)

        return feature_matrix

    def _statistical_regime_detection(self, returns: pd.Series) -> RegimeMetrics:
        """Fallback statistical regime detection when HMM unavailable."""
        vol_regime = self.detect_volatility_regime(returns)

        regime_state = RegimeState(
            regime_id=vol_regime['regime_id'],
            regime_name=self.regime_names.get(vol_regime['regime_id'], "Unknown"),
            volatility=vol_regime['current_vol'],
            mean_return=returns.mean() * 252,
            probability=1.0,
            duration=0,
            timestamp=datetime.now()
        )

        # Dummy transition matrix
        transition_matrix = np.eye(self.n_regimes) * 0.9 + 0.1 / self.n_regimes

        return RegimeMetrics(
            current_regime=regime_state,
            regime_probabilities=np.array([1.0 if i == regime_state.regime_id else 0.0
                                          for i in range(self.n_regimes)]),
            transition_matrix=transition_matrix,
            expected_duration=10.0,
            regime_stability=0.9,
            regime_history=[regime_state.regime_id]
        )

    def _calculate_regime_duration(self, hidden_states: np.ndarray) -> int:
        """Calculate current regime duration."""
        if len(hidden_states) == 0:
            return 0

        current_regime = hidden_states[-1]
        duration = 1

        for i in range(len(hidden_states) - 2, -1, -1):
            if hidden_states[i] == current_regime:
                duration += 1
            else:
                break

        return duration

    def _calculate_regime_stability(self, posterior_probs: np.ndarray) -> float:
        """Calculate regime stability from posterior probabilities."""
        if len(posterior_probs) == 0:
            return 0.0

        # Maximum probability across time
        max_probs = np.max(posterior_probs, axis=1)

        # Stability is average of maximum probabilities
        stability = np.mean(max_probs)

        return float(stability)

    def _calculate_expected_duration(self, regime: int) -> float:
        """Calculate expected regime duration from transition matrix."""
        if not self.is_fitted:
            return 10.0

        # Expected duration = 1 / (1 - self_transition_prob)
        self_transition = self.hmm_model.transmat_[regime, regime]
        expected = 1.0 / (1.0 - self_transition + 1e-8)

        return float(expected)

    def _log_regime_statistics(self, features: np.ndarray):
        """Log regime statistics after fitting."""
        try:
            # Predict all regimes
            hidden_states = self.hmm_model.predict(features)

            # Regime frequencies
            unique, counts = np.unique(hidden_states, return_counts=True)
            for regime, count in zip(unique, counts):
                freq = count / len(hidden_states)
                logger.info(f"Regime {regime}: {freq:.1%} of observations")

            # Average regime durations
            persistence = self.calculate_regime_persistence(hidden_states)
            for regime, duration in persistence.items():
                logger.info(f"Regime {regime} avg duration: {duration:.1f} periods")

        except Exception as e:
            logger.warning(f"Could not log regime statistics: {str(e)}")
