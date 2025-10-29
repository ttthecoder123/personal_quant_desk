"""
Volatility Forecasting using GARCH models and advanced techniques.

Implements institutional-grade volatility forecasting:
- GARCH(1,1) and variations
- EWMA (Exponentially Weighted Moving Average)
- Realized volatility calculations
- Volatility term structure
- Intraday patterns and spike detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
from loguru import logger

try:
    from arch import arch_model
    from arch.univariate import GARCH, EGARCH, FIGARCH
    ARCH_AVAILABLE = True
except ImportError:
    logger.warning("arch package not available - GARCH features disabled")
    ARCH_AVAILABLE = False


@dataclass
class VolatilityForecast:
    """Container for volatility forecast."""
    current_vol: float
    forecast_vol: float
    forecast_horizon: int
    confidence_interval: Tuple[float, float]
    model_type: str
    timestamp: datetime


@dataclass
class VolatilityMetrics:
    """Container for comprehensive volatility metrics."""
    realized_vol: float
    garch_vol: float
    ewma_vol: float
    parkinson_vol: float
    garman_klass_vol: float
    yang_zhang_vol: float
    vol_ratio: float  # Short-term / long-term
    vol_regime: str
    spike_detected: bool


class VolatilityForecaster:
    """
    Advanced volatility forecasting using multiple approaches.

    Implements:
    - GARCH family models
    - EWMA volatility
    - Range-based estimators (Parkinson, Garman-Klass, Yang-Zhang)
    - Realized volatility
    - Volatility term structure
    - Intraday patterns
    """

    def __init__(self,
                 garch_p: int = 1,
                 garch_q: int = 1,
                 ewma_lambda: float = 0.94,
                 vol_window: int = 20,
                 spike_threshold: float = 2.5):
        """
        Initialize volatility forecaster.

        Args:
            garch_p: GARCH lag order for volatility
            garch_q: GARCH lag order for residuals
            ewma_lambda: Decay factor for EWMA (RiskMetrics uses 0.94)
            vol_window: Window for realized volatility
            spike_threshold: Standard deviations for spike detection
        """
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.ewma_lambda = ewma_lambda
        self.vol_window = vol_window
        self.spike_threshold = spike_threshold

        # Model storage
        self.garch_model = None
        self.garch_result = None
        self.is_fitted = False

        # Historical volatility tracking
        self.vol_history = []
        self.forecast_history = []

        logger.info(f"Initialized VolatilityForecaster with GARCH({garch_p},{garch_q})")

    def fit_garch(self, returns: pd.Series, model_type: str = 'GARCH') -> 'VolatilityForecaster':
        """
        Fit GARCH model to returns.

        Args:
            returns: Time series of returns
            model_type: Type of GARCH model ('GARCH', 'EGARCH', 'GJR-GARCH')

        Returns:
            self
        """
        if not ARCH_AVAILABLE:
            logger.error("arch package not available")
            return self

        try:
            returns_clean = returns.dropna() * 100  # Convert to percentage

            # Initialize model
            if model_type == 'EGARCH':
                self.garch_model = arch_model(
                    returns_clean,
                    vol='EGARCH',
                    p=self.garch_p,
                    q=self.garch_q
                )
            elif model_type == 'GJR-GARCH':
                self.garch_model = arch_model(
                    returns_clean,
                    vol='GARCH',
                    p=self.garch_p,
                    o=1,  # Asymmetric term
                    q=self.garch_q
                )
            else:  # Standard GARCH
                self.garch_model = arch_model(
                    returns_clean,
                    vol='GARCH',
                    p=self.garch_p,
                    q=self.garch_q
                )

            # Fit model
            self.garch_result = self.garch_model.fit(disp='off', show_warning=False)
            self.is_fitted = True

            logger.success(f"{model_type} model fitted successfully")
            logger.info(f"Log-likelihood: {self.garch_result.loglikelihood:.2f}")
            logger.info(f"AIC: {self.garch_result.aic:.2f}, BIC: {self.garch_result.bic:.2f}")

            # Log parameters
            self._log_garch_parameters()

        except Exception as e:
            logger.error(f"Error fitting GARCH model: {str(e)}")
            self.is_fitted = False

        return self

    def forecast_volatility(self,
                          returns: pd.Series,
                          horizon: int = 1,
                          method: str = 'GARCH') -> VolatilityForecast:
        """
        Forecast future volatility.

        Args:
            returns: Time series of returns
            horizon: Forecast horizon in periods
            method: Forecasting method ('GARCH', 'EWMA', 'REALIZED')

        Returns:
            VolatilityForecast object
        """
        if method == 'GARCH' and self.is_fitted:
            return self._forecast_garch(returns, horizon)
        elif method == 'EWMA':
            return self._forecast_ewma(returns, horizon)
        else:
            return self._forecast_realized(returns, horizon)

    def calculate_realized_volatility(self,
                                     returns: pd.Series,
                                     window: Optional[int] = None) -> pd.Series:
        """
        Calculate realized volatility.

        Args:
            returns: Time series of returns
            window: Rolling window (default: self.vol_window)

        Returns:
            Series of realized volatility (annualized)
        """
        window = window or self.vol_window

        # Simple realized volatility
        realized_vol = returns.rolling(window).std() * np.sqrt(252)

        return realized_vol

    def calculate_parkinson_volatility(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """
        Parkinson volatility estimator using high-low range.

        More efficient than close-to-close volatility.

        Args:
            high: High prices
            low: Low prices

        Returns:
            Series of Parkinson volatility (annualized)
        """
        # Parkinson's volatility
        hl_ratio = (high / low).apply(np.log)
        parkinson_vol = np.sqrt(1 / (4 * np.log(2)) * hl_ratio ** 2) * np.sqrt(252)

        return parkinson_vol.rolling(self.vol_window).mean()

    def calculate_garman_klass_volatility(self,
                                         open_: pd.Series,
                                         high: pd.Series,
                                         low: pd.Series,
                                         close: pd.Series) -> pd.Series:
        """
        Garman-Klass volatility estimator.

        More efficient than Parkinson, uses open and close as well.

        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Series of Garman-Klass volatility (annualized)
        """
        log_hl = (high / low).apply(np.log)
        log_co = (close / open_).apply(np.log)

        gk_vol = np.sqrt(
            0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        ) * np.sqrt(252)

        return gk_vol.rolling(self.vol_window).mean()

    def calculate_yang_zhang_volatility(self,
                                        open_: pd.Series,
                                        high: pd.Series,
                                        low: pd.Series,
                                        close: pd.Series) -> pd.Series:
        """
        Yang-Zhang volatility estimator.

        Handles opening jumps and drift, most efficient for small samples.

        Args:
            open_: Open prices
            high: High prices
            low: Low prices
            close: Close prices

        Returns:
            Series of Yang-Zhang volatility (annualized)
        """
        log_ho = (high / open_).apply(np.log)
        log_lo = (low / open_).apply(np.log)
        log_co = (close / open_).apply(np.log)

        log_oc = (open_ / close.shift(1)).apply(np.log)
        log_cc = (close / close.shift(1)).apply(np.log)

        # Overnight volatility
        overnight_vol = log_oc.rolling(self.vol_window).var()

        # Open-to-close volatility (Rogers-Satchell)
        rs_vol = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(self.vol_window).mean()

        # Close-to-close volatility
        close_vol = log_cc.rolling(self.vol_window).var()

        # Yang-Zhang estimator
        k = 0.34 / (1.34 + (self.vol_window + 1) / (self.vol_window - 1))
        yz_vol = np.sqrt(overnight_vol + k * close_vol + (1 - k) * rs_vol) * np.sqrt(252)

        return yz_vol

    def calculate_ewma_volatility(self, returns: pd.Series) -> pd.Series:
        """
        Calculate EWMA (Exponentially Weighted Moving Average) volatility.

        RiskMetrics approach with lambda = 0.94.

        Args:
            returns: Time series of returns

        Returns:
            Series of EWMA volatility (annualized)
        """
        # EWMA variance
        ewma_var = returns.ewm(alpha=1-self.ewma_lambda, adjust=False).var()

        # Annualized volatility
        ewma_vol = np.sqrt(ewma_var * 252)

        return ewma_vol

    def calculate_volatility_term_structure(self,
                                           returns: pd.Series,
                                           horizons: List[int] = None) -> pd.DataFrame:
        """
        Calculate volatility term structure across multiple horizons.

        Args:
            returns: Time series of returns
            horizons: List of forecast horizons in days

        Returns:
            DataFrame with volatility forecasts for each horizon
        """
        if horizons is None:
            horizons = [1, 5, 10, 21, 63, 126, 252]

        term_structure = pd.DataFrame(index=['volatility'], columns=horizons)

        for horizon in horizons:
            # Use GARCH if fitted, otherwise EWMA
            if self.is_fitted:
                forecast = self._forecast_garch(returns, horizon)
            else:
                forecast = self._forecast_ewma(returns, horizon)

            term_structure.loc['volatility', horizon] = forecast.forecast_vol

        return term_structure.T

    def detect_volatility_spike(self, returns: pd.Series) -> Dict[str, any]:
        """
        Detect volatility spikes.

        Args:
            returns: Time series of returns

        Returns:
            Dictionary with spike detection results
        """
        # Calculate rolling volatility
        rolling_vol = self.calculate_realized_volatility(returns, window=self.vol_window)

        # Current volatility
        current_vol = rolling_vol.iloc[-1]

        # Historical mean and std
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()

        # Z-score
        z_score = (current_vol - vol_mean) / (vol_std + 1e-8)

        # Spike detection
        spike_detected = abs(z_score) > self.spike_threshold

        # Rate of change
        vol_change = rolling_vol.pct_change().iloc[-1]

        return {
            'spike_detected': bool(spike_detected),
            'z_score': float(z_score),
            'current_vol': float(current_vol),
            'mean_vol': float(vol_mean),
            'vol_percentile': float(stats.percentileofscore(rolling_vol.dropna(), current_vol) / 100),
            'vol_change_pct': float(vol_change),
            'severity': 'high' if abs(z_score) > 3 else 'medium' if abs(z_score) > 2 else 'low'
        }

    def calculate_intraday_volatility_pattern(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate average intraday volatility pattern.

        Args:
            returns: DataFrame with DatetimeIndex and intraday returns

        Returns:
            Series with average volatility by time of day
        """
        if not isinstance(returns.index, pd.DatetimeIndex):
            logger.warning("Index must be DatetimeIndex for intraday analysis")
            return pd.Series()

        # Extract hour (or hour+minute)
        returns['hour'] = returns.index.hour

        # Calculate average volatility by hour
        intraday_pattern = returns.groupby('hour').apply(
            lambda x: x.iloc[:, 0].std() * np.sqrt(252 * 6.5)  # Assuming 6.5 hour trading day
        )

        return intraday_pattern

    def calculate_comprehensive_metrics(self,
                                       returns: pd.Series,
                                       high: Optional[pd.Series] = None,
                                       low: Optional[pd.Series] = None,
                                       open_: Optional[pd.Series] = None,
                                       close: Optional[pd.Series] = None) -> VolatilityMetrics:
        """
        Calculate comprehensive volatility metrics.

        Args:
            returns: Time series of returns
            high: High prices (optional)
            low: Low prices (optional)
            open_: Open prices (optional)
            close: Close prices (optional)

        Returns:
            VolatilityMetrics object
        """
        # Realized volatility
        realized_vol = self.calculate_realized_volatility(returns).iloc[-1]

        # EWMA volatility
        ewma_vol = self.calculate_ewma_volatility(returns).iloc[-1]

        # GARCH volatility
        if self.is_fitted:
            garch_forecast = self._forecast_garch(returns, horizon=1)
            garch_vol = garch_forecast.forecast_vol
        else:
            garch_vol = realized_vol

        # Range-based estimators if data available
        parkinson_vol = np.nan
        garman_klass_vol = np.nan
        yang_zhang_vol = np.nan

        if high is not None and low is not None:
            parkinson_vol = self.calculate_parkinson_volatility(high, low).iloc[-1]

            if open_ is not None and close is not None:
                garman_klass_vol = self.calculate_garman_klass_volatility(
                    open_, high, low, close
                ).iloc[-1]
                yang_zhang_vol = self.calculate_yang_zhang_volatility(
                    open_, high, low, close
                ).iloc[-1]

        # Volatility ratio (short-term / long-term)
        vol_short = returns.iloc[-10:].std() * np.sqrt(252)
        vol_long = returns.iloc[-60:].std() * np.sqrt(252) if len(returns) >= 60 else vol_short
        vol_ratio = vol_short / vol_long if vol_long > 0 else 1.0

        # Volatility regime
        vol_percentile = stats.percentileofscore(
            self.calculate_realized_volatility(returns).dropna(),
            realized_vol
        ) / 100

        if vol_percentile < 0.33:
            vol_regime = "low"
        elif vol_percentile < 0.67:
            vol_regime = "medium"
        else:
            vol_regime = "high"

        # Spike detection
        spike_info = self.detect_volatility_spike(returns)

        return VolatilityMetrics(
            realized_vol=float(realized_vol),
            garch_vol=float(garch_vol),
            ewma_vol=float(ewma_vol),
            parkinson_vol=float(parkinson_vol) if not np.isnan(parkinson_vol) else 0.0,
            garman_klass_vol=float(garman_klass_vol) if not np.isnan(garman_klass_vol) else 0.0,
            yang_zhang_vol=float(yang_zhang_vol) if not np.isnan(yang_zhang_vol) else 0.0,
            vol_ratio=float(vol_ratio),
            vol_regime=vol_regime,
            spike_detected=spike_info['spike_detected']
        )

    def generate_volatility_report(self,
                                  returns: pd.Series,
                                  high: Optional[pd.Series] = None,
                                  low: Optional[pd.Series] = None,
                                  open_: Optional[pd.Series] = None,
                                  close: Optional[pd.Series] = None) -> Dict:
        """
        Generate comprehensive volatility report.

        Args:
            returns: Time series of returns
            high: High prices (optional)
            low: Low prices (optional)
            open_: Open prices (optional)
            close: Close prices (optional)

        Returns:
            Dictionary with complete volatility analysis
        """
        metrics = self.calculate_comprehensive_metrics(
            returns, high, low, open_, close
        )

        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'realized_vol': metrics.realized_vol,
                'garch_vol': metrics.garch_vol,
                'ewma_vol': metrics.ewma_vol,
                'parkinson_vol': metrics.parkinson_vol,
                'garman_klass_vol': metrics.garman_klass_vol,
                'yang_zhang_vol': metrics.yang_zhang_vol,
                'vol_ratio': metrics.vol_ratio,
                'vol_regime': metrics.vol_regime
            },
            'spike_detection': self.detect_volatility_spike(returns),
        }

        # Add forecasts
        if self.is_fitted:
            forecasts = {}
            for horizon in [1, 5, 10, 21]:
                forecast = self._forecast_garch(returns, horizon)
                forecasts[f'day_{horizon}'] = {
                    'vol': forecast.forecast_vol,
                    'ci_lower': forecast.confidence_interval[0],
                    'ci_upper': forecast.confidence_interval[1]
                }
            report['forecasts'] = forecasts

        return report

    # Private helper methods

    def _forecast_garch(self, returns: pd.Series, horizon: int) -> VolatilityForecast:
        """Forecast using GARCH model."""
        if not self.is_fitted:
            logger.warning("GARCH model not fitted")
            return self._forecast_realized(returns, horizon)

        try:
            # Refit with latest data
            returns_clean = returns.dropna() * 100
            self.garch_result = self.garch_model.fit(disp='off', show_warning=False)

            # Forecast
            forecast = self.garch_result.forecast(horizon=horizon)

            # Extract forecast variance and convert to volatility
            forecast_var = forecast.variance.values[-1, :]
            forecast_vol = np.sqrt(forecast_var.mean() / 100) * np.sqrt(252)  # Annualized

            # Current volatility
            current_vol = np.sqrt(self.garch_result.conditional_volatility.iloc[-1] ** 2 / 100) * np.sqrt(252)

            # Confidence intervals (approximate)
            ci_lower = forecast_vol * 0.8
            ci_upper = forecast_vol * 1.2

            return VolatilityForecast(
                current_vol=float(current_vol),
                forecast_vol=float(forecast_vol),
                forecast_horizon=horizon,
                confidence_interval=(float(ci_lower), float(ci_upper)),
                model_type='GARCH',
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Error in GARCH forecast: {str(e)}")
            return self._forecast_realized(returns, horizon)

    def _forecast_ewma(self, returns: pd.Series, horizon: int) -> VolatilityForecast:
        """Forecast using EWMA."""
        ewma_vol = self.calculate_ewma_volatility(returns)
        current_vol = ewma_vol.iloc[-1]

        # EWMA forecast (constant for all horizons in simple version)
        forecast_vol = current_vol

        # Confidence intervals
        ci_lower = forecast_vol * 0.85
        ci_upper = forecast_vol * 1.15

        return VolatilityForecast(
            current_vol=float(current_vol),
            forecast_vol=float(forecast_vol),
            forecast_horizon=horizon,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            model_type='EWMA',
            timestamp=datetime.now()
        )

    def _forecast_realized(self, returns: pd.Series, horizon: int) -> VolatilityForecast:
        """Forecast using realized volatility."""
        realized_vol = self.calculate_realized_volatility(returns)
        current_vol = realized_vol.iloc[-1]

        # Simple persistence forecast
        forecast_vol = current_vol

        # Confidence intervals
        ci_lower = forecast_vol * 0.8
        ci_upper = forecast_vol * 1.2

        return VolatilityForecast(
            current_vol=float(current_vol),
            forecast_vol=float(forecast_vol),
            forecast_horizon=horizon,
            confidence_interval=(float(ci_lower), float(ci_upper)),
            model_type='REALIZED',
            timestamp=datetime.now()
        )

    def _log_garch_parameters(self):
        """Log GARCH model parameters."""
        if not self.is_fitted:
            return

        try:
            params = self.garch_result.params
            logger.info("GARCH Parameters:")
            for param, value in params.items():
                logger.info(f"  {param}: {value:.6f}")

            # Persistence
            alpha = params.get('alpha[1]', 0)
            beta = params.get('beta[1]', 0)
            persistence = alpha + beta
            logger.info(f"  Persistence (α+β): {persistence:.6f}")

            # Half-life
            if persistence < 1:
                half_life = np.log(0.5) / np.log(persistence)
                logger.info(f"  Half-life: {half_life:.1f} periods")

        except Exception as e:
            logger.warning(f"Could not log GARCH parameters: {str(e)}")
