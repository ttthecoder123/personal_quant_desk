"""
Factor Risk Analysis and Management.

Implements institutional-grade factor risk modeling:
- Multi-factor exposure calculation (Fama-French + momentum)
- Factor contribution to risk decomposition
- Factor limits and monitoring
- Factor VaR calculation
- Factor hedging recommendations
- Style drift detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
from loguru import logger

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    logger.warning("statsmodels not available - limited factor analysis")
    STATSMODELS_AVAILABLE = False


@dataclass
class FactorExposure:
    """Container for factor exposure."""
    factor_name: str
    exposure: float
    t_statistic: float
    p_value: float
    r_squared: float
    significant: bool


@dataclass
class FactorRiskMetrics:
    """Container for factor risk metrics."""
    factor_exposures: Dict[str, float]
    factor_var: float
    specific_var: float
    total_var: float
    factor_contribution_pct: float
    factor_limit_breaches: List[str]
    diversification_ratio: float


class FactorRisk:
    """
    Factor risk analysis and management.

    Implements:
    - Fama-French 3-factor and 5-factor models
    - Carhart 4-factor (adds momentum)
    - Factor exposure estimation
    - Risk decomposition (factor vs specific)
    - Factor VaR calculation
    - Factor limit monitoring
    - Factor hedging recommendations
    """

    def __init__(self,
                 model_type: str = 'fama_french_5',
                 lookback_window: int = 252,
                 factor_limits: Optional[Dict[str, float]] = None,
                 var_confidence: float = 0.95):
        """
        Initialize factor risk analyzer.

        Args:
            model_type: Factor model ('fama_french_3', 'fama_french_5', 'carhart_4')
            lookback_window: Window for factor exposure estimation
            factor_limits: Dictionary of factor exposure limits
            var_confidence: Confidence level for VaR calculation
        """
        self.model_type = model_type
        self.lookback_window = lookback_window
        self.var_confidence = var_confidence

        # Default factor limits (beta-equivalent)
        self.factor_limits = factor_limits or {
            'market': 1.5,      # Market beta
            'size': 0.5,        # SMB exposure
            'value': 0.5,       # HML exposure
            'momentum': 0.5,    # MOM exposure
            'profitability': 0.5,  # RMW exposure
            'investment': 0.5   # CMA exposure
        }

        # Factor data storage
        self.factor_returns = None
        self.factor_covariance = None

        # Exposure history
        self.exposure_history = []

        logger.info(f"Initialized FactorRisk with {model_type} model")

    def set_factor_returns(self, factor_returns: pd.DataFrame):
        """
        Set factor return data.

        Expected columns depend on model_type:
        - fama_french_3: ['Mkt-RF', 'SMB', 'HML', 'RF']
        - fama_french_5: ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        - carhart_4: ['Mkt-RF', 'SMB', 'HML', 'MOM', 'RF']

        Args:
            factor_returns: DataFrame with factor returns
        """
        self.factor_returns = factor_returns

        # Calculate factor covariance matrix
        factor_cols = [col for col in factor_returns.columns if col != 'RF']
        self.factor_covariance = factor_returns[factor_cols].cov()

        logger.info(f"Factor returns set with {len(factor_returns)} observations")

    def calculate_factor_exposures(self,
                                   asset_returns: pd.Series,
                                   factor_returns: Optional[pd.DataFrame] = None) -> Dict[str, FactorExposure]:
        """
        Calculate factor exposures using regression analysis.

        Regression: R_asset - R_f = α + β_factors * F + ε

        Args:
            asset_returns: Time series of asset returns
            factor_returns: DataFrame with factor returns (uses self.factor_returns if None)

        Returns:
            Dictionary of FactorExposure objects
        """
        if factor_returns is None:
            factor_returns = self.factor_returns

        if factor_returns is None:
            logger.error("Factor returns not set")
            return {}

        if not STATSMODELS_AVAILABLE:
            logger.error("statsmodels not available")
            return {}

        try:
            # Align data
            aligned_data = pd.concat([asset_returns, factor_returns], axis=1).dropna()

            if len(aligned_data) < 30:
                logger.warning("Insufficient data for factor regression")
                return {}

            # Excess returns
            excess_returns = aligned_data[asset_returns.name] - aligned_data['RF']

            # Factor columns (exclude RF)
            factor_cols = [col for col in factor_returns.columns if col != 'RF']

            # Regression
            X = aligned_data[factor_cols]
            X = sm.add_constant(X)  # Add intercept
            y = excess_returns

            model = sm.OLS(y, X)
            results = model.fit()

            # Extract exposures
            exposures = {}

            for i, factor in enumerate(['alpha'] + factor_cols):
                param = results.params.iloc[i]
                t_stat = results.tvalues.iloc[i]
                p_val = results.pvalues.iloc[i]

                exposures[factor] = FactorExposure(
                    factor_name=factor,
                    exposure=float(param),
                    t_statistic=float(t_stat),
                    p_value=float(p_val),
                    r_squared=float(results.rsquared),
                    significant=p_val < 0.05
                )

            logger.debug(f"Factor exposures calculated - R²: {results.rsquared:.3f}")

            return exposures

        except Exception as e:
            logger.error(f"Error calculating factor exposures: {str(e)}")
            return {}

    def calculate_portfolio_factor_exposure(self,
                                           positions: Dict[str, float],
                                           asset_returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        Calculate portfolio-level factor exposures.

        Portfolio exposure = weighted average of individual exposures.

        Args:
            positions: Dictionary of {asset: weight}
            asset_returns: Dictionary of {asset: returns_series}

        Returns:
            Dictionary of portfolio factor exposures
        """
        # Calculate exposures for each asset
        asset_exposures = {}

        for asset, weight in positions.items():
            if asset in asset_returns:
                exposures = self.calculate_factor_exposures(asset_returns[asset])
                asset_exposures[asset] = exposures

        # Aggregate to portfolio level
        portfolio_exposures = {}

        # Get all factor names
        all_factors = set()
        for exposures in asset_exposures.values():
            all_factors.update(exposures.keys())

        # Weighted average
        for factor in all_factors:
            total_exposure = 0.0
            for asset, weight in positions.items():
                if asset in asset_exposures and factor in asset_exposures[asset]:
                    total_exposure += weight * asset_exposures[asset][factor].exposure

            portfolio_exposures[factor] = total_exposure

        return portfolio_exposures

    def decompose_risk(self,
                      asset_returns: pd.Series,
                      factor_returns: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Decompose portfolio risk into factor and specific components.

        Total Var = Factor Var + Specific Var

        Args:
            asset_returns: Time series of asset returns
            factor_returns: DataFrame with factor returns

        Returns:
            Dictionary with risk decomposition
        """
        # Get factor exposures
        exposures = self.calculate_factor_exposures(asset_returns, factor_returns)

        if not exposures:
            return {}

        try:
            # Extract beta vector (exclude alpha)
            factor_names = [name for name in exposures.keys() if name != 'alpha']
            beta_vector = np.array([exposures[name].exposure for name in factor_names])

            # Factor covariance matrix
            if factor_returns is None:
                factor_returns = self.factor_returns

            factor_cols = [col for col in factor_returns.columns if col != 'RF']
            factor_cov = factor_returns[factor_cols].cov().values

            # Factor variance contribution
            factor_var = beta_vector @ factor_cov @ beta_vector

            # Total variance
            total_var = asset_returns.var()

            # Specific variance (residual)
            specific_var = total_var - factor_var

            # Handle negative specific variance (numerical issues)
            if specific_var < 0:
                specific_var = 0
                factor_var = total_var

            # Volatility (annualized)
            factor_vol = np.sqrt(factor_var * 252)
            specific_vol = np.sqrt(specific_var * 252)
            total_vol = np.sqrt(total_var * 252)

            # Contribution percentages
            factor_contribution = factor_var / total_var if total_var > 0 else 0
            specific_contribution = specific_var / total_var if total_var > 0 else 0

            return {
                'factor_variance': float(factor_var),
                'specific_variance': float(specific_var),
                'total_variance': float(total_var),
                'factor_volatility': float(factor_vol),
                'specific_volatility': float(specific_vol),
                'total_volatility': float(total_vol),
                'factor_contribution_pct': float(factor_contribution * 100),
                'specific_contribution_pct': float(specific_contribution * 100)
            }

        except Exception as e:
            logger.error(f"Error decomposing risk: {str(e)}")
            return {}

    def calculate_factor_var(self,
                            asset_returns: pd.Series,
                            factor_returns: Optional[pd.DataFrame] = None,
                            horizon: int = 1) -> Dict[str, float]:
        """
        Calculate Value at Risk decomposed by factors.

        Args:
            asset_returns: Time series of asset returns
            factor_returns: DataFrame with factor returns
            horizon: VaR horizon in days

        Returns:
            Dictionary with VaR metrics
        """
        # Get factor exposures and risk decomposition
        exposures = self.calculate_factor_exposures(asset_returns, factor_returns)
        risk_decomp = self.decompose_risk(asset_returns, factor_returns)

        if not exposures or not risk_decomp:
            return {}

        # Total VaR (parametric)
        total_vol = risk_decomp['total_volatility'] / np.sqrt(252)  # Daily vol
        z_score = stats.norm.ppf(1 - self.var_confidence)
        total_var = -z_score * total_vol * np.sqrt(horizon)

        # Factor VaR
        factor_vol = risk_decomp['factor_volatility'] / np.sqrt(252)
        factor_var = -z_score * factor_vol * np.sqrt(horizon)

        # Specific VaR
        specific_vol = risk_decomp['specific_volatility'] / np.sqrt(252)
        specific_var = -z_score * specific_vol * np.sqrt(horizon)

        # Individual factor contributions
        factor_names = [name for name in exposures.keys() if name != 'alpha']
        beta_vector = np.array([exposures[name].exposure for name in factor_names])

        if factor_returns is None:
            factor_returns = self.factor_returns

        factor_cols = [col for col in factor_returns.columns if col != 'RF']
        factor_vols = factor_returns[factor_cols].std().values * np.sqrt(252)

        # Marginal VaR contribution by factor
        factor_contributions = {}
        for i, factor in enumerate(factor_names):
            marginal_var = abs(beta_vector[i]) * factor_vols[i] / np.sqrt(252) * z_score * np.sqrt(horizon)
            factor_contributions[factor] = float(marginal_var)

        return {
            'total_var': float(total_var),
            'factor_var': float(factor_var),
            'specific_var': float(specific_var),
            'factor_var_pct': float(factor_var / total_var * 100) if total_var != 0 else 0,
            'specific_var_pct': float(specific_var / total_var * 100) if total_var != 0 else 0,
            'factor_contributions': factor_contributions,
            'confidence_level': self.var_confidence,
            'horizon_days': horizon
        }

    def check_factor_limits(self, exposures: Dict[str, FactorExposure]) -> Dict[str, any]:
        """
        Check if factor exposures exceed limits.

        Args:
            exposures: Dictionary of FactorExposure objects

        Returns:
            Dictionary with limit check results
        """
        breaches = []
        factor_status = {}

        # Map factor names
        factor_mapping = {
            'Mkt-RF': 'market',
            'SMB': 'size',
            'HML': 'value',
            'MOM': 'momentum',
            'RMW': 'profitability',
            'CMA': 'investment'
        }

        for factor_name, exposure_obj in exposures.items():
            if factor_name == 'alpha':
                continue

            # Map to standard name
            standard_name = factor_mapping.get(factor_name, factor_name.lower())

            exposure_value = abs(exposure_obj.exposure)
            limit = self.factor_limits.get(standard_name, float('inf'))

            breach = exposure_value > limit

            if breach:
                breaches.append({
                    'factor': factor_name,
                    'exposure': float(exposure_value),
                    'limit': float(limit),
                    'excess': float(exposure_value - limit)
                })

            factor_status[factor_name] = {
                'exposure': float(exposure_obj.exposure),
                'limit': float(limit),
                'utilization_pct': float(exposure_value / limit * 100) if limit != float('inf') else 0,
                'breach': breach
            }

        return {
            'has_breaches': len(breaches) > 0,
            'n_breaches': len(breaches),
            'breaches': breaches,
            'factor_status': factor_status,
            'recommendation': 'Reduce factor exposures' if breaches else 'Within limits'
        }

    def calculate_factor_diversification(self, exposures: Dict[str, FactorExposure]) -> float:
        """
        Calculate factor diversification ratio.

        Measures how diversified the factor exposures are.

        Args:
            exposures: Dictionary of FactorExposure objects

        Returns:
            Diversification ratio (higher is more diversified)
        """
        factor_exposures = [abs(exp.exposure) for name, exp in exposures.items() if name != 'alpha']

        if not factor_exposures:
            return 0.0

        # Sum of absolute exposures
        sum_exposures = sum(factor_exposures)

        # Sum of squared exposures (Herfindahl index)
        sum_squared = sum(exp ** 2 for exp in factor_exposures)

        # Diversification ratio (1/HHI normalized)
        if sum_squared > 0:
            hhi = sum_squared / (sum_exposures ** 2)
            diversification = 1.0 / hhi if hhi > 0 else 0.0
        else:
            diversification = 0.0

        return float(diversification)

    def recommend_factor_hedges(self,
                               exposures: Dict[str, FactorExposure],
                               target_exposures: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Recommend factor hedging positions.

        Args:
            exposures: Current factor exposures
            target_exposures: Target exposures (default: neutral = 0)

        Returns:
            Dictionary of recommended hedge amounts by factor
        """
        if target_exposures is None:
            # Default to neutral (zero) exposures
            target_exposures = {name: 0.0 for name in exposures.keys() if name != 'alpha'}

        hedges = {}

        for factor_name, exposure_obj in exposures.items():
            if factor_name == 'alpha':
                continue

            current = exposure_obj.exposure
            target = target_exposures.get(factor_name, 0.0)

            # Required hedge
            hedge_amount = target - current

            if abs(hedge_amount) > 0.05:  # Materiality threshold
                hedges[factor_name] = float(hedge_amount)

        return hedges

    def detect_style_drift(self,
                          exposures_history: List[Dict[str, FactorExposure]],
                          threshold: float = 0.5) -> Dict[str, any]:
        """
        Detect style drift in factor exposures over time.

        Args:
            exposures_history: List of historical factor exposures
            threshold: Threshold for significant drift

        Returns:
            Dictionary with drift detection results
        """
        if len(exposures_history) < 2:
            return {'drift_detected': False}

        # Extract exposure time series for each factor
        factor_series = {}

        for exposures in exposures_history:
            for factor_name, exposure_obj in exposures.items():
                if factor_name not in factor_series:
                    factor_series[factor_name] = []
                factor_series[factor_name].append(exposure_obj.exposure)

        # Analyze drift for each factor
        drift_analysis = {}
        significant_drifts = []

        for factor_name, values in factor_series.items():
            if factor_name == 'alpha':
                continue

            values_array = np.array(values)

            # Trend analysis
            x = np.arange(len(values_array))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values_array)

            # Total change
            total_change = values_array[-1] - values_array[0]

            # Drift detected if significant trend or large change
            drift_detected = abs(total_change) > threshold or (p_value < 0.05 and abs(slope) > 0.01)

            drift_analysis[factor_name] = {
                'initial_exposure': float(values_array[0]),
                'current_exposure': float(values_array[-1]),
                'total_change': float(total_change),
                'trend_slope': float(slope),
                'trend_pvalue': float(p_value),
                'drift_detected': drift_detected
            }

            if drift_detected:
                significant_drifts.append(factor_name)

        return {
            'drift_detected': len(significant_drifts) > 0,
            'factors_with_drift': significant_drifts,
            'n_factors_drifted': len(significant_drifts),
            'drift_analysis': drift_analysis,
            'recommendation': 'Review factor exposures' if significant_drifts else 'Stable exposures'
        }

    def calculate_comprehensive_metrics(self,
                                       asset_returns: pd.Series,
                                       factor_returns: Optional[pd.DataFrame] = None) -> FactorRiskMetrics:
        """
        Calculate comprehensive factor risk metrics.

        Args:
            asset_returns: Time series of asset returns
            factor_returns: DataFrame with factor returns

        Returns:
            FactorRiskMetrics object
        """
        # Factor exposures
        exposures = self.calculate_factor_exposures(asset_returns, factor_returns)

        # Risk decomposition
        risk_decomp = self.decompose_risk(asset_returns, factor_returns)

        # Factor VaR
        factor_var_metrics = self.calculate_factor_var(asset_returns, factor_returns)

        # Factor limits
        limit_check = self.check_factor_limits(exposures)

        # Diversification
        diversification = self.calculate_factor_diversification(exposures)

        # Extract exposures as dictionary
        factor_exposures_dict = {
            name: exp.exposure for name, exp in exposures.items()
        }

        return FactorRiskMetrics(
            factor_exposures=factor_exposures_dict,
            factor_var=risk_decomp.get('factor_variance', 0.0),
            specific_var=risk_decomp.get('specific_variance', 0.0),
            total_var=risk_decomp.get('total_variance', 0.0),
            factor_contribution_pct=risk_decomp.get('factor_contribution_pct', 0.0),
            factor_limit_breaches=[b['factor'] for b in limit_check['breaches']],
            diversification_ratio=diversification
        )

    def generate_factor_report(self,
                              asset_returns: pd.Series,
                              factor_returns: Optional[pd.DataFrame] = None) -> Dict:
        """
        Generate comprehensive factor risk report.

        Args:
            asset_returns: Time series of asset returns
            factor_returns: DataFrame with factor returns

        Returns:
            Dictionary with complete factor analysis
        """
        # Comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(asset_returns, factor_returns)

        # Factor exposures
        exposures = self.calculate_factor_exposures(asset_returns, factor_returns)

        # Risk decomposition
        risk_decomp = self.decompose_risk(asset_returns, factor_returns)

        # Factor VaR
        factor_var = self.calculate_factor_var(asset_returns, factor_returns)

        # Limit checks
        limit_check = self.check_factor_limits(exposures)

        # Hedging recommendations
        hedges = self.recommend_factor_hedges(exposures)

        report = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'factor_exposures': {
                name: {
                    'exposure': exp.exposure,
                    't_statistic': exp.t_statistic,
                    'p_value': exp.p_value,
                    'significant': exp.significant
                }
                for name, exp in exposures.items()
            },
            'risk_decomposition': risk_decomp,
            'factor_var': factor_var,
            'limit_checks': limit_check,
            'diversification_ratio': metrics.diversification_ratio,
            'hedging_recommendations': hedges
        }

        return report

    def create_factor_neutral_portfolio(self,
                                       asset_returns: Dict[str, pd.Series],
                                       target_vol: float = 0.10) -> Dict[str, float]:
        """
        Create factor-neutral portfolio using optimization.

        Args:
            asset_returns: Dictionary of asset returns
            target_vol: Target portfolio volatility

        Returns:
            Dictionary of optimal weights
        """
        if not STATSMODELS_AVAILABLE:
            logger.error("statsmodels required for optimization")
            return {}

        try:
            # Calculate factor exposures for all assets
            asset_exposures = {}
            for asset, returns in asset_returns.items():
                exposures = self.calculate_factor_exposures(returns)
                asset_exposures[asset] = exposures

            # Setup optimization
            n_assets = len(asset_returns)
            assets = list(asset_returns.keys())

            # Initial guess (equal weight)
            x0 = np.ones(n_assets) / n_assets

            # Objective: minimize tracking error from factor-neutral
            def objective(weights):
                portfolio_exposures = {}
                all_factors = set()

                for asset, exp_dict in asset_exposures.items():
                    all_factors.update(exp_dict.keys())

                for factor in all_factors:
                    if factor == 'alpha':
                        continue
                    exp_sum = sum(
                        weights[i] * asset_exposures[assets[i]].get(factor, FactorExposure(
                            factor, 0, 0, 1, 0, False
                        )).exposure
                        for i in range(n_assets)
                    )
                    portfolio_exposures[factor] = exp_sum

                # Minimize sum of squared exposures (factor neutrality)
                return sum(exp ** 2 for exp in portfolio_exposures.values())

            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
            ]

            # Bounds
            bounds = [(0, 0.3) for _ in range(n_assets)]  # Max 30% per asset

            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                weights = {asset: float(w) for asset, w in zip(assets, result.x)}
                logger.success("Factor-neutral portfolio created")
                return weights
            else:
                logger.error("Optimization failed")
                return {}

        except Exception as e:
            logger.error(f"Error creating factor-neutral portfolio: {str(e)}")
            return {}
