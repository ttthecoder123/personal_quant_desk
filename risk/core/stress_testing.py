"""
Stress Testing Framework

Implements comprehensive stress testing:
- Historical scenario analysis (2008, 2020, etc.)
- Hypothetical scenarios (rate shocks, crashes, etc.)
- Sensitivity analysis
- Reverse stress testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class ScenarioType(Enum):
    """Types of stress scenarios"""
    HISTORICAL = "historical"
    HYPOTHETICAL = "hypothetical"
    SENSITIVITY = "sensitivity"
    REVERSE = "reverse"


@dataclass
class StressScenario:
    """Definition of a stress scenario"""
    name: str
    scenario_type: ScenarioType
    description: str
    shocks: Dict[str, float]  # Asset -> % change
    probability: Optional[float] = None


@dataclass
class StressTestResult:
    """Results of stress test"""
    scenario: StressScenario
    timestamp: datetime
    portfolio_return: float
    portfolio_value: float
    position_impacts: Dict[str, float]
    var_change: float
    volatility_change: float


class StressTester:
    """Comprehensive stress testing system"""

    def __init__(self):
        """Initialize stress tester with predefined scenarios"""
        self.scenarios = self._define_scenarios()
        self.results_history = []

    def _define_scenarios(self) -> List[StressScenario]:
        """Define standard stress scenarios"""
        scenarios = []

        # Historical Scenarios
        scenarios.append(StressScenario(
            name="2008_financial_crisis",
            scenario_type=ScenarioType.HISTORICAL,
            description="2008 Financial Crisis (Lehman collapse)",
            shocks={
                'SPY': -0.40,  # Equities down 40%
                'QQQ': -0.45,  # Tech down 45%
                'GC=F': 0.10,  # Gold up 10%
                'CL=F': -0.50,  # Oil down 50%
                'AUDUSD=X': -0.20,  # AUD down 20%
                'USDJPY=X': 0.15,  # JPY strengthens (USD up)
                'EURUSD=X': -0.10  # EUR weakens
            },
            probability=0.01
        ))

        scenarios.append(StressScenario(
            name="2020_covid_crash",
            scenario_type=ScenarioType.HISTORICAL,
            description="2020 COVID-19 Market Crash",
            shocks={
                'SPY': -0.30,
                'QQQ': -0.25,
                'GC=F': 0.15,
                'CL=F': -0.65,  # Oil crash
                'AUDUSD=X': -0.15,
                'USDJPY=X': 0.10,
                'EURUSD=X': -0.05
            },
            probability=0.02
        ))

        scenarios.append(StressScenario(
            name="2022_rate_shock",
            scenario_type=ScenarioType.HISTORICAL,
            description="2022 Fed Rate Hiking Cycle",
            shocks={
                'SPY': -0.20,
                'QQQ': -0.30,  # Growth stocks hit harder
                'GC=F': -0.05,
                'CL=F': 0.10,
                'AUDUSD=X': -0.08,
                'USDJPY=X': 0.20,  # USD strengthens significantly
                'EURUSD=X': -0.15
            },
            probability=0.05
        ))

        # Flash Crash
        scenarios.append(StressScenario(
            name="flash_crash",
            scenario_type=ScenarioType.HISTORICAL,
            description="Flash Crash Event (2010-style)",
            shocks={
                'SPY': -0.10,
                'QQQ': -0.12,
                'GC=F': 0.05,
                'CL=F': -0.08,
                'AUDUSD=X': -0.05,
                'USDJPY=X': 0.03,
                'EURUSD=X': -0.03
            },
            probability=0.03
        ))

        # Hypothetical Scenarios
        scenarios.append(StressScenario(
            name="interest_rate_shock_up",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description="200bps parallel rate increase",
            shocks={
                'SPY': -0.15,
                'QQQ': -0.20,
                'GC=F': -0.10,
                'CL=F': -0.05,
                'AUDUSD=X': -0.10,
                'USDJPY=X': 0.15,
                'EURUSD=X': -0.08
            },
            probability=0.10
        ))

        scenarios.append(StressScenario(
            name="interest_rate_shock_down",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description="200bps parallel rate decrease",
            shocks={
                'SPY': 0.10,
                'QQQ': 0.15,
                'GC=F': 0.20,
                'CL=F': 0.05,
                'AUDUSD=X': 0.08,
                'USDJPY=X': -0.10,
                'EURUSD=X': 0.05
            },
            probability=0.05
        ))

        scenarios.append(StressScenario(
            name="equity_crash_30",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description="30% equity market crash",
            shocks={
                'SPY': -0.30,
                'QQQ': -0.35,
                'GC=F': 0.15,
                'CL=F': -0.20,
                'AUDUSD=X': -0.15,
                'USDJPY=X': 0.12,
                'EURUSD=X': -0.08
            },
            probability=0.02
        ))

        scenarios.append(StressScenario(
            name="equity_crash_40",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description="40% equity market crash",
            shocks={
                'SPY': -0.40,
                'QQQ': -0.45,
                'GC=F': 0.20,
                'CL=F': -0.30,
                'AUDUSD=X': -0.20,
                'USDJPY=X': 0.15,
                'EURUSD=X': -0.12
            },
            probability=0.01
        ))

        scenarios.append(StressScenario(
            name="oil_price_shock_up",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description="50% oil price increase",
            shocks={
                'SPY': -0.08,
                'QQQ': -0.10,
                'GC=F': 0.05,
                'CL=F': 0.50,
                'AUDUSD=X': 0.05,
                'USDJPY=X': -0.03,
                'EURUSD=X': -0.02
            },
            probability=0.08
        ))

        scenarios.append(StressScenario(
            name="oil_price_shock_down",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description="50% oil price decrease",
            shocks={
                'SPY': 0.05,
                'QQQ': 0.08,
                'GC=F': -0.03,
                'CL=F': -0.50,
                'AUDUSD=X': -0.08,
                'USDJPY=X': 0.02,
                'EURUSD=X': 0.03
            },
            probability=0.10
        ))

        scenarios.append(StressScenario(
            name="currency_crisis",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description="Major currency devaluation",
            shocks={
                'SPY': -0.10,
                'QQQ': -0.12,
                'GC=F': 0.15,
                'CL=F': 0.05,
                'AUDUSD=X': -0.25,
                'USDJPY=X': 0.20,
                'EURUSD=X': -0.15
            },
            probability=0.05
        ))

        scenarios.append(StressScenario(
            name="correlation_breakdown",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description="All correlations → 1 (diversification fails)",
            shocks={
                'SPY': -0.25,
                'QQQ': -0.25,
                'GC=F': -0.25,  # Gold fails as hedge
                'CL=F': -0.25,
                'AUDUSD=X': -0.25,
                'USDJPY=X': -0.25,
                'EURUSD=X': -0.25
            },
            probability=0.02
        ))

        scenarios.append(StressScenario(
            name="liquidity_freeze",
            scenario_type=ScenarioType.HYPOTHETICAL,
            description="Market liquidity freeze (2x normal spreads)",
            shocks={
                'SPY': -0.15,
                'QQQ': -0.18,
                'GC=F': -0.08,
                'CL=F': -0.20,
                'AUDUSD=X': -0.12,
                'USDJPY=X': -0.10,
                'EURUSD=X': -0.10
            },
            probability=0.03
        ))

        return scenarios

    def run_scenario(
        self,
        scenario: StressScenario,
        positions: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float
    ) -> StressTestResult:
        """
        Run a single stress scenario

        Args:
            scenario: Stress scenario to run
            positions: Current positions (symbol -> quantity)
            current_prices: Current prices (symbol -> price)
            portfolio_value: Current portfolio value

        Returns:
            StressTestResult
        """
        position_impacts = {}
        total_impact = 0.0

        # Calculate impact on each position
        for symbol, quantity in positions.items():
            if quantity == 0:
                position_impacts[symbol] = 0.0
                continue

            current_price = current_prices.get(symbol, 0)
            current_position_value = quantity * current_price

            # Get shock for this symbol
            shock = scenario.shocks.get(symbol, 0.0)

            # Calculate P&L impact
            position_pnl = current_position_value * shock
            position_impacts[symbol] = position_pnl
            total_impact += position_pnl

        # Calculate portfolio return and new value
        portfolio_return = total_impact / portfolio_value if portfolio_value > 0 else 0.0
        new_portfolio_value = portfolio_value + total_impact

        return StressTestResult(
            scenario=scenario,
            timestamp=datetime.now(),
            portfolio_return=portfolio_return,
            portfolio_value=new_portfolio_value,
            position_impacts=position_impacts,
            var_change=0.0,  # Will be calculated separately if needed
            volatility_change=0.0  # Will be calculated separately if needed
        )

    def run_all_scenarios(
        self,
        positions: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float
    ) -> List[StressTestResult]:
        """
        Run all predefined stress scenarios

        Args:
            positions: Current positions
            current_prices: Current prices
            portfolio_value: Current portfolio value

        Returns:
            List of stress test results
        """
        results = []

        for scenario in self.scenarios:
            result = self.run_scenario(
                scenario,
                positions,
                current_prices,
                portfolio_value
            )
            results.append(result)

        # Store in history
        self.results_history.extend(results)

        return results

    def sensitivity_analysis(
        self,
        symbol: str,
        shock_range: np.ndarray,
        positions: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis for a single asset

        Args:
            symbol: Symbol to stress
            shock_range: Array of shocks to test (e.g., np.linspace(-0.5, 0.5, 21))
            positions: Current positions
            current_prices: Current prices
            portfolio_value: Current portfolio value

        Returns:
            DataFrame with shock -> portfolio impact
        """
        results = []

        for shock in shock_range:
            scenario = StressScenario(
                name=f"sensitivity_{symbol}_{shock:.2%}",
                scenario_type=ScenarioType.SENSITIVITY,
                description=f"{symbol} {shock:.2%} shock",
                shocks={symbol: shock}
            )

            result = self.run_scenario(
                scenario,
                positions,
                current_prices,
                portfolio_value
            )

            results.append({
                'symbol': symbol,
                'shock': shock,
                'portfolio_return': result.portfolio_return,
                'portfolio_value': result.portfolio_value,
                'position_impact': result.position_impacts.get(symbol, 0.0)
            })

        return pd.DataFrame(results)

    def multi_factor_scenario(
        self,
        shocks: Dict[str, float],
        positions: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float,
        scenario_name: str = "custom"
    ) -> StressTestResult:
        """
        Create and run a custom multi-factor scenario

        Args:
            shocks: Dictionary of symbol -> shock
            positions: Current positions
            current_prices: Current prices
            portfolio_value: Current portfolio value
            scenario_name: Name for the scenario

        Returns:
            StressTestResult
        """
        scenario = StressScenario(
            name=scenario_name,
            scenario_type=ScenarioType.HYPOTHETICAL,
            description=f"Custom scenario: {scenario_name}",
            shocks=shocks
        )

        return self.run_scenario(
            scenario,
            positions,
            current_prices,
            portfolio_value
        )

    def reverse_stress_test(
        self,
        target_loss: float,
        positions: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float,
        max_shock: float = 0.50
    ) -> Dict[str, float]:
        """
        Reverse stress test: find shocks that would cause target loss

        Args:
            target_loss: Target portfolio loss (positive number)
            positions: Current positions
            current_prices: Current prices
            portfolio_value: Current portfolio value
            max_shock: Maximum allowed shock magnitude

        Returns:
            Dictionary of shocks that would cause the target loss
        """
        from scipy.optimize import minimize

        # Get position values
        position_values = {}
        for symbol, quantity in positions.items():
            position_values[symbol] = quantity * current_prices.get(symbol, 0)

        symbols = list(positions.keys())
        n_symbols = len(symbols)

        def objective(shocks_array):
            """Minimize difference from target loss"""
            # Calculate portfolio loss from shocks
            total_loss = sum(
                position_values[symbol] * shock
                for symbol, shock in zip(symbols, shocks_array)
            )
            # We want negative loss (positive return means loss for short positions)
            return (total_loss + target_loss) ** 2

        # Initial guess: equal shocks
        x0 = np.ones(n_symbols) * (-target_loss / portfolio_value)

        # Constraints: shocks bounded
        bounds = [(-max_shock, max_shock) for _ in range(n_symbols)]

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds
        )

        # Convert to dictionary
        reverse_shocks = dict(zip(symbols, result.x))

        return reverse_shocks

    def get_worst_scenarios(
        self,
        n_scenarios: int = 5
    ) -> List[StressTestResult]:
        """
        Get the worst N scenarios from history

        Args:
            n_scenarios: Number of worst scenarios to return

        Returns:
            List of worst scenario results
        """
        if not self.results_history:
            return []

        # Sort by portfolio return (most negative first)
        sorted_results = sorted(
            self.results_history,
            key=lambda x: x.portfolio_return
        )

        return sorted_results[:n_scenarios]

    def expected_loss(
        self,
        results: List[StressTestResult]
    ) -> float:
        """
        Calculate probability-weighted expected loss from scenarios

        Args:
            results: List of stress test results with probabilities

        Returns:
            Expected loss
        """
        total_prob = sum(
            r.scenario.probability
            for r in results
            if r.scenario.probability is not None
        )

        if total_prob == 0:
            return 0.0

        expected_loss = sum(
            r.portfolio_return * (r.scenario.probability / total_prob)
            for r in results
            if r.scenario.probability is not None
        )

        return expected_loss
