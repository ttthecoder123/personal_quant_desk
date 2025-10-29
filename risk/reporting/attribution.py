"""
Risk Factor Attribution Analysis

Decomposes portfolio risk into factor contributions:
- Factor-based risk attribution
- Asset class attribution
- Systematic vs idiosyncratic risk
- Factor contribution to VaR
- Time-varying factor exposure analysis
- Marginal risk contribution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings

try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - limited attribution functionality")


class AttributionMethod(Enum):
    """Risk attribution methodology"""
    FACTOR_MODEL = "factor_model"  # Factor-based decomposition
    BRINSON = "brinson"  # Brinson attribution
    MARGINAL_VAR = "marginal_var"  # Marginal VaR contribution
    COMPONENT_VAR = "component_var"  # Component VaR
    EULER = "euler"  # Euler decomposition


class RiskFactorType(Enum):
    """Types of risk factors"""
    MARKET = "market"
    SIZE = "size"
    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    CURRENCY = "currency"
    SECTOR = "sector"
    COUNTRY = "country"


@dataclass
class FactorContribution:
    """Factor contribution to portfolio risk"""
    factor_name: str
    factor_type: RiskFactorType
    exposure: float
    volatility: float
    risk_contribution: float
    risk_contribution_pct: float
    marginal_risk: float
    diversification_benefit: float

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'factor_name': self.factor_name,
            'factor_type': self.factor_type.value,
            'exposure': self.exposure,
            'volatility': self.volatility,
            'risk_contribution': self.risk_contribution,
            'risk_contribution_pct': self.risk_contribution_pct,
            'marginal_risk': self.marginal_risk,
            'diversification_benefit': self.diversification_benefit
        }


@dataclass
class AttributionResult:
    """Complete risk attribution result"""
    timestamp: datetime
    method: AttributionMethod
    total_risk: float
    factor_contributions: List[FactorContribution]
    specific_risk: float
    specific_risk_pct: float
    diversification_ratio: float
    concentration_index: float
    explained_variance: float

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'method': self.method.value,
            'total_risk': self.total_risk,
            'factor_contributions': [fc.to_dict() for fc in self.factor_contributions],
            'specific_risk': self.specific_risk,
            'specific_risk_pct': self.specific_risk_pct,
            'diversification_ratio': self.diversification_ratio,
            'concentration_index': self.concentration_index,
            'explained_variance': self.explained_variance
        }


@dataclass
class AssetClassAttribution:
    """Risk attribution by asset class"""
    asset_class: str
    weight: float
    risk_contribution: float
    risk_contribution_pct: float
    return_contribution: float
    sharpe_contribution: float

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'asset_class': self.asset_class,
            'weight': self.weight,
            'risk_contribution': self.risk_contribution,
            'risk_contribution_pct': self.risk_contribution_pct,
            'return_contribution': self.return_contribution,
            'sharpe_contribution': self.sharpe_contribution
        }


class RiskAttribution:
    """
    Comprehensive risk attribution analysis

    Decomposes portfolio risk into:
    - Factor contributions
    - Asset class contributions
    - Systematic vs idiosyncratic risk
    - Marginal risk contributions
    - Time-varying exposures
    """

    def __init__(
        self,
        method: AttributionMethod = AttributionMethod.COMPONENT_VAR,
        annualization_factor: int = 252
    ):
        """
        Initialize risk attribution analyzer

        Args:
            method: Attribution methodology
            annualization_factor: Days per year for annualization
        """
        self.method = method
        self.annualization_factor = annualization_factor

        # Attribution history
        self.attribution_history: List[AttributionResult] = []

    def calculate_factor_attribution(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        weights: Optional[np.ndarray] = None
    ) -> AttributionResult:
        """
        Calculate factor-based risk attribution

        Args:
            returns: Asset returns DataFrame (assets x time)
            factor_returns: Factor returns DataFrame (factors x time)
            weights: Optional portfolio weights (defaults to equal weight)

        Returns:
            AttributionResult with factor contributions
        """
        if weights is None:
            weights = np.ones(len(returns.columns)) / len(returns.columns)

        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)

        # Estimate factor exposures (betas)
        factor_exposures = self._estimate_factor_exposures(
            portfolio_returns, factor_returns
        )

        # Calculate factor covariance matrix
        factor_cov = factor_returns.cov() * self.annualization_factor

        # Calculate portfolio risk
        portfolio_variance = portfolio_returns.var() * self.annualization_factor
        portfolio_risk = np.sqrt(portfolio_variance)

        # Calculate factor contributions
        factor_contributions = []
        total_factor_risk = 0

        for i, factor_name in enumerate(factor_returns.columns):
            exposure = factor_exposures[i]
            factor_vol = np.sqrt(factor_cov.iloc[i, i])

            # Marginal contribution to risk
            marginal_risk = exposure * factor_vol ** 2 / portfolio_risk

            # Component contribution
            component_risk = exposure * marginal_risk

            total_factor_risk += component_risk ** 2

            factor_contribution = FactorContribution(
                factor_name=factor_name,
                factor_type=self._infer_factor_type(factor_name),
                exposure=exposure,
                volatility=factor_vol,
                risk_contribution=component_risk,
                risk_contribution_pct=(component_risk / portfolio_risk * 100) if portfolio_risk > 0 else 0,
                marginal_risk=marginal_risk,
                diversification_benefit=0.0  # Calculated below
            )
            factor_contributions.append(factor_contribution)

        # Calculate specific (idiosyncratic) risk
        total_factor_risk = np.sqrt(total_factor_risk)
        specific_variance = max(0, portfolio_variance - total_factor_risk ** 2)
        specific_risk = np.sqrt(specific_variance)
        specific_risk_pct = (specific_risk / portfolio_risk * 100) if portfolio_risk > 0 else 0

        # Calculate diversification ratio
        sum_component_risks = sum(abs(fc.risk_contribution) for fc in factor_contributions)
        diversification_ratio = (sum_component_risks / portfolio_risk) if portfolio_risk > 0 else 1.0

        # Calculate concentration index (Herfindahl)
        risk_contributions_squared = [(fc.risk_contribution_pct / 100) ** 2 for fc in factor_contributions]
        concentration_index = sum(risk_contributions_squared)

        # Calculate explained variance (R-squared from factor model)
        explained_variance = 1 - (specific_variance / portfolio_variance) if portfolio_variance > 0 else 0

        # Create attribution result
        result = AttributionResult(
            timestamp=datetime.now(),
            method=self.method,
            total_risk=portfolio_risk,
            factor_contributions=factor_contributions,
            specific_risk=specific_risk,
            specific_risk_pct=specific_risk_pct,
            diversification_ratio=diversification_ratio,
            concentration_index=concentration_index,
            explained_variance=explained_variance
        )

        self.attribution_history.append(result)

        return result

    def calculate_component_var_attribution(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate Component VaR attribution

        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights
            confidence_level: VaR confidence level

        Returns:
            Dictionary of asset VaR contributions
        """
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)

        # Calculate portfolio VaR
        if SCIPY_AVAILABLE:
            portfolio_var = -stats.norm.ppf(1 - confidence_level) * portfolio_returns.std() * np.sqrt(self.annualization_factor)
        else:
            portfolio_var = -np.percentile(portfolio_returns, (1 - confidence_level) * 100)

        # Calculate covariance matrix
        cov_matrix = returns.cov() * self.annualization_factor

        # Calculate marginal VaR for each asset
        portfolio_std = portfolio_returns.std() * np.sqrt(self.annualization_factor)

        var_contributions = {}

        for i, asset in enumerate(returns.columns):
            # Marginal VaR = (weight * covariance with portfolio) / portfolio_std
            cov_with_portfolio = sum(
                weights[j] * cov_matrix.iloc[i, j]
                for j in range(len(weights))
            )

            marginal_var = cov_with_portfolio / portfolio_std

            # Component VaR
            component_var = weights[i] * marginal_var

            var_contributions[asset] = component_var

        return var_contributions

    def calculate_marginal_risk_contribution(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate marginal risk contribution (MRC) for each asset

        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights

        Returns:
            Dictionary of marginal risk contributions
        """
        # Calculate covariance matrix
        cov_matrix = returns.cov() * self.annualization_factor

        # Calculate portfolio variance
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)

        # Calculate marginal risk contributions
        marginal_contributions = {}

        for i, asset in enumerate(returns.columns):
            # MRC = (covariance with portfolio) / portfolio_std
            cov_with_portfolio = np.dot(cov_matrix.iloc[i, :], weights)
            mrc = cov_with_portfolio / portfolio_std

            marginal_contributions[asset] = mrc

        return marginal_contributions

    def calculate_asset_class_attribution(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        asset_classes: Dict[str, str]
    ) -> List[AssetClassAttribution]:
        """
        Calculate risk attribution by asset class

        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights
            asset_classes: Mapping of asset to asset class

        Returns:
            List of asset class attributions
        """
        # Group assets by class
        class_weights = {}
        class_returns = {}

        for i, asset in enumerate(returns.columns):
            asset_class = asset_classes.get(asset, "Unknown")

            if asset_class not in class_weights:
                class_weights[asset_class] = 0
                class_returns[asset_class] = []

            class_weights[asset_class] += weights[i]
            class_returns[asset_class].append(returns[asset] * weights[i])

        # Calculate portfolio metrics
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_risk = portfolio_returns.std() * np.sqrt(self.annualization_factor)

        # Calculate attribution for each class
        attributions = []

        for asset_class in class_weights:
            # Aggregate class returns
            class_return_series = pd.concat(class_returns[asset_class], axis=1).sum(axis=1)

            # Calculate risk contribution
            class_risk = class_return_series.std() * np.sqrt(self.annualization_factor)

            # Calculate correlation with portfolio
            correlation = class_return_series.corr(portfolio_returns)

            # Risk contribution
            risk_contribution = class_weights[asset_class] * class_risk * correlation

            # Return contribution
            return_contribution = class_return_series.mean() * self.annualization_factor

            # Sharpe contribution (simplified)
            sharpe_contribution = (return_contribution / risk_contribution) if risk_contribution > 0 else 0

            attribution = AssetClassAttribution(
                asset_class=asset_class,
                weight=class_weights[asset_class],
                risk_contribution=risk_contribution,
                risk_contribution_pct=(risk_contribution / portfolio_risk * 100) if portfolio_risk > 0 else 0,
                return_contribution=return_contribution,
                sharpe_contribution=sharpe_contribution
            )
            attributions.append(attribution)

        # Sort by risk contribution
        attributions.sort(key=lambda x: abs(x.risk_contribution), reverse=True)

        return attributions

    def calculate_time_varying_attribution(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        weights: np.ndarray,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Calculate rolling factor attribution over time

        Args:
            returns: Asset returns DataFrame
            factor_returns: Factor returns DataFrame
            weights: Portfolio weights
            window: Rolling window size

        Returns:
            DataFrame with time-varying attributions
        """
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)

        # Rolling factor exposures
        rolling_exposures = pd.DataFrame(
            index=returns.index[window:],
            columns=factor_returns.columns
        )

        for i in range(window, len(returns)):
            window_portfolio = portfolio_returns.iloc[i-window:i]
            window_factors = factor_returns.iloc[i-window:i]

            exposures = self._estimate_factor_exposures(
                window_portfolio,
                window_factors
            )

            rolling_exposures.iloc[i-window] = exposures

        return rolling_exposures

    def _estimate_factor_exposures(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> np.ndarray:
        """
        Estimate factor exposures using regression

        Args:
            portfolio_returns: Portfolio return series
            factor_returns: Factor return DataFrame

        Returns:
            Array of factor exposures (betas)
        """
        # Align data
        aligned_data = pd.concat([portfolio_returns, factor_returns], axis=1).dropna()

        if len(aligned_data) < 2:
            return np.zeros(len(factor_returns.columns))

        y = aligned_data.iloc[:, 0].values
        X = aligned_data.iloc[:, 1:].values

        # Add intercept
        X = np.column_stack([np.ones(len(X)), X])

        # OLS regression
        try:
            betas = np.linalg.lstsq(X, y, rcond=None)[0]
            return betas[1:]  # Exclude intercept
        except np.linalg.LinAlgError:
            return np.zeros(len(factor_returns.columns))

    def _infer_factor_type(self, factor_name: str) -> RiskFactorType:
        """Infer factor type from factor name"""
        name_lower = factor_name.lower()

        if 'market' in name_lower or 'mkt' in name_lower:
            return RiskFactorType.MARKET
        elif 'size' in name_lower or 'smb' in name_lower:
            return RiskFactorType.SIZE
        elif 'value' in name_lower or 'hml' in name_lower:
            return RiskFactorType.VALUE
        elif 'momentum' in name_lower or 'mom' in name_lower or 'umd' in name_lower:
            return RiskFactorType.MOMENTUM
        elif 'quality' in name_lower or 'rmw' in name_lower:
            return RiskFactorType.QUALITY
        elif 'volatility' in name_lower or 'vol' in name_lower:
            return RiskFactorType.VOLATILITY
        elif 'liquidity' in name_lower or 'liq' in name_lower:
            return RiskFactorType.LIQUIDITY
        elif 'currency' in name_lower or 'fx' in name_lower:
            return RiskFactorType.CURRENCY
        elif 'sector' in name_lower:
            return RiskFactorType.SECTOR
        elif 'country' in name_lower:
            return RiskFactorType.COUNTRY
        else:
            return RiskFactorType.MARKET

    def get_attribution_summary(
        self,
        result: AttributionResult,
        top_n: int = 10
    ) -> str:
        """
        Generate text summary of attribution results

        Args:
            result: Attribution result
            top_n: Number of top contributors to show

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("RISK ATTRIBUTION ANALYSIS")
        lines.append("=" * 60)
        lines.append(f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Method: {result.method.value}")
        lines.append("")

        lines.append(f"Total Portfolio Risk: {result.total_risk:.2%}")
        lines.append(f"Specific Risk: {result.specific_risk:.2%} ({result.specific_risk_pct:.1f}%)")
        lines.append(f"Diversification Ratio: {result.diversification_ratio:.2f}")
        lines.append(f"Concentration Index: {result.concentration_index:.4f}")
        lines.append(f"Explained Variance: {result.explained_variance:.2%}")
        lines.append("")

        lines.append(f"Top {top_n} Risk Contributors:")
        lines.append(f"{'Factor':<20} {'Exposure':<12} {'Risk %':<12} {'Marginal':<12}")
        lines.append("-" * 60)

        # Sort by absolute risk contribution
        sorted_factors = sorted(
            result.factor_contributions,
            key=lambda x: abs(x.risk_contribution_pct),
            reverse=True
        )

        for factor in sorted_factors[:top_n]:
            lines.append(
                f"{factor.factor_name:<20} "
                f"{factor.exposure:>11.3f} "
                f"{factor.risk_contribution_pct:>11.2f}% "
                f"{factor.marginal_risk:>11.4f}"
            )

        lines.append("=" * 60)

        return "\n".join(lines)

    def export_attribution_to_dataframe(
        self,
        result: AttributionResult
    ) -> pd.DataFrame:
        """
        Export attribution results to DataFrame

        Args:
            result: Attribution result

        Returns:
            DataFrame with attribution data
        """
        data = []

        for factor in result.factor_contributions:
            data.append({
                'Factor': factor.factor_name,
                'Type': factor.factor_type.value,
                'Exposure': factor.exposure,
                'Volatility': factor.volatility,
                'Risk Contribution': factor.risk_contribution,
                'Risk Contribution %': factor.risk_contribution_pct,
                'Marginal Risk': factor.marginal_risk,
                'Diversification Benefit': factor.diversification_benefit
            })

        df = pd.DataFrame(data)

        # Add summary row
        summary = {
            'Factor': 'TOTAL',
            'Type': '',
            'Exposure': '',
            'Volatility': '',
            'Risk Contribution': result.total_risk,
            'Risk Contribution %': 100.0,
            'Marginal Risk': '',
            'Diversification Benefit': ''
        }
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

        # Add specific risk row
        specific = {
            'Factor': 'Specific Risk',
            'Type': '',
            'Exposure': '',
            'Volatility': result.specific_risk,
            'Risk Contribution': result.specific_risk,
            'Risk Contribution %': result.specific_risk_pct,
            'Marginal Risk': '',
            'Diversification Benefit': ''
        }
        df = pd.concat([df, pd.DataFrame([specific])], ignore_index=True)

        return df

    def calculate_risk_parity_weights(
        self,
        returns: pd.DataFrame,
        target_risk: Optional[float] = None
    ) -> np.ndarray:
        """
        Calculate risk parity weights (equal risk contribution)

        Args:
            returns: Asset returns DataFrame
            target_risk: Optional target portfolio risk

        Returns:
            Array of risk parity weights
        """
        if not SCIPY_AVAILABLE:
            # Fallback to inverse volatility
            vols = returns.std() * np.sqrt(self.annualization_factor)
            weights = (1 / vols) / (1 / vols).sum()
            return weights.values

        # Calculate covariance matrix
        cov_matrix = returns.cov() * self.annualization_factor

        n_assets = len(returns.columns)

        def risk_parity_objective(weights):
            """Objective: minimize variance of risk contributions"""
            portfolio_std = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

            # Calculate risk contributions
            marginal_contrib = np.dot(cov_matrix, weights) / portfolio_std
            risk_contrib = weights * marginal_contrib

            # Target equal risk contribution
            target_contrib = portfolio_std / n_assets

            # Sum of squared deviations from equal contribution
            return np.sum((risk_contrib - target_contrib) ** 2)

        # Constraints: weights sum to 1, all positive
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Initial guess: equal weights
        x0 = np.ones(n_assets) / n_assets

        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            return result.x
        else:
            # Fallback to equal weights
            return x0
