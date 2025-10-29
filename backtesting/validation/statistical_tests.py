"""
Statistical Tests Module

Implements comprehensive statistical tests for backtesting validation including:
- Sharpe ratio significance tests (Johnson's formula)
- Information ratio tests
- Maximum drawdown distribution analysis
- Win rate confidence intervals
- Profit factor significance
- Parametric and non-parametric tests
- Distribution comparison tests
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from scipy.stats import norm, t as t_dist


@dataclass
class StatisticalTestResult:
    """Container for statistical test results."""

    test_name: str
    statistic: float
    p_value: float
    confidence_level: float
    null_hypothesis: str
    conclusion: str
    additional_info: Dict = None

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant."""
        return self.p_value < alpha

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'confidence_level': self.confidence_level,
            'null_hypothesis': self.null_hypothesis,
            'conclusion': self.conclusion,
            'is_significant': self.is_significant(),
            'additional_info': self.additional_info or {}
        }


class StatisticalTests:
    """
    Comprehensive statistical tests for backtesting validation.

    Implements rigorous statistical methods to validate strategy performance
    and detect statistically significant results.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical tests.

        Args:
            confidence_level: Confidence level for intervals (default 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        logger.info(f"Initialized StatisticalTests with {confidence_level:.1%} confidence")

    def sharpe_ratio_significance(
        self,
        returns: np.ndarray,
        sharpe_ratio: float,
        risk_free_rate: float = 0.0,
        annualization_factor: int = 252
    ) -> StatisticalTestResult:
        """
        Test Sharpe ratio significance using Johnson's formula.

        Tests whether the Sharpe ratio is significantly different from zero
        accounting for autocorrelation and higher moments.

        Args:
            returns: Array of returns
            sharpe_ratio: Calculated Sharpe ratio
            risk_free_rate: Risk-free rate
            annualization_factor: Annualization factor (252 for daily)

        Returns:
            StatisticalTestResult with test details
        """
        logger.debug("Calculating Sharpe ratio significance")

        n = len(returns)
        excess_returns = returns - risk_free_rate / annualization_factor

        # Calculate higher moments
        skewness = stats.skew(excess_returns)
        kurtosis = stats.kurtosis(excess_returns, fisher=True)  # Excess kurtosis

        # Johnson's formula for Sharpe ratio standard error
        # SE(SR) = sqrt((1 + SR^2/2 - skew*SR + (kurt-1)*SR^2/4) / n)
        sr_variance = (
            1 +
            sharpe_ratio**2 / 2 -
            skewness * sharpe_ratio +
            (kurtosis - 1) * sharpe_ratio**2 / 4
        ) / n

        sr_std_error = np.sqrt(sr_variance)

        # t-statistic
        t_stat = sharpe_ratio / sr_std_error if sr_std_error > 0 else 0

        # p-value (two-tailed)
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=n-1))

        conclusion = (
            f"Sharpe ratio of {sharpe_ratio:.3f} is "
            f"{'statistically significant' if p_value < self.alpha else 'not statistically significant'} "
            f"at {self.confidence_level:.1%} confidence"
        )

        return StatisticalTestResult(
            test_name="Sharpe Ratio Significance (Johnson's Formula)",
            statistic=t_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            null_hypothesis="Sharpe ratio equals zero",
            conclusion=conclusion,
            additional_info={
                'sharpe_ratio': sharpe_ratio,
                'standard_error': sr_std_error,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'n_observations': n
            }
        )

    def information_ratio_test(
        self,
        strategy_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> StatisticalTestResult:
        """
        Test information ratio significance.

        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns

        Returns:
            StatisticalTestResult with test details
        """
        logger.debug("Calculating information ratio significance")

        # Active returns
        active_returns = strategy_returns - benchmark_returns

        # Information ratio
        ir = np.mean(active_returns) / np.std(active_returns, ddof=1) if np.std(active_returns) > 0 else 0

        # t-test for non-zero mean
        n = len(active_returns)
        t_stat = ir * np.sqrt(n)
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df=n-1))

        conclusion = (
            f"Information ratio of {ir:.3f} is "
            f"{'statistically significant' if p_value < self.alpha else 'not statistically significant'} "
            f"at {self.confidence_level:.1%} confidence"
        )

        return StatisticalTestResult(
            test_name="Information Ratio Significance Test",
            statistic=t_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            null_hypothesis="Information ratio equals zero (no alpha)",
            conclusion=conclusion,
            additional_info={
                'information_ratio': ir,
                'mean_active_return': np.mean(active_returns),
                'tracking_error': np.std(active_returns, ddof=1),
                'n_observations': n
            }
        )

    def maximum_drawdown_distribution(
        self,
        returns: np.ndarray,
        observed_mdd: float,
        n_simulations: int = 10000
    ) -> StatisticalTestResult:
        """
        Analyze maximum drawdown distribution via Monte Carlo.

        Tests whether observed MDD is consistent with random returns
        having the same distribution.

        Args:
            returns: Array of returns
            observed_mdd: Observed maximum drawdown
            n_simulations: Number of Monte Carlo simulations

        Returns:
            StatisticalTestResult with test details
        """
        logger.debug(f"Running MDD distribution analysis ({n_simulations} simulations)")

        n = len(returns)
        simulated_mdds = []

        for _ in range(n_simulations):
            # Randomly permute returns
            shuffled_returns = np.random.permutation(returns)

            # Calculate cumulative returns and drawdown
            cum_returns = np.cumprod(1 + shuffled_returns)
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - running_max) / running_max
            mdd = np.min(drawdown)

            simulated_mdds.append(mdd)

        simulated_mdds = np.array(simulated_mdds)

        # Calculate p-value: proportion of simulations with worse MDD
        p_value = np.mean(simulated_mdds <= observed_mdd)

        # Percentile rank
        percentile = np.percentile(simulated_mdds, [5, 50, 95])

        conclusion = (
            f"Observed MDD of {observed_mdd:.2%} is at the "
            f"{(1-p_value)*100:.1f}th percentile of the simulated distribution "
            f"({'worse than expected' if p_value < 0.5 else 'better than expected'})"
        )

        return StatisticalTestResult(
            test_name="Maximum Drawdown Distribution Analysis",
            statistic=observed_mdd,
            p_value=p_value,
            confidence_level=self.confidence_level,
            null_hypothesis="MDD is consistent with random ordering of returns",
            conclusion=conclusion,
            additional_info={
                'observed_mdd': observed_mdd,
                'simulated_median': percentile[1],
                'simulated_5th': percentile[0],
                'simulated_95th': percentile[2],
                'n_simulations': n_simulations
            }
        )

    def win_rate_confidence_interval(
        self,
        n_wins: int,
        n_trades: int,
        test_value: float = 0.5
    ) -> StatisticalTestResult:
        """
        Calculate confidence interval for win rate using binomial distribution.

        Args:
            n_wins: Number of winning trades
            n_trades: Total number of trades
            test_value: Value to test against (default 0.5)

        Returns:
            StatisticalTestResult with test details
        """
        logger.debug(f"Calculating win rate CI for {n_wins}/{n_trades} wins")

        if n_trades == 0:
            logger.warning("No trades to analyze")
            return StatisticalTestResult(
                test_name="Win Rate Confidence Interval",
                statistic=0,
                p_value=1.0,
                confidence_level=self.confidence_level,
                null_hypothesis=f"Win rate equals {test_value}",
                conclusion="No trades to analyze"
            )

        win_rate = n_wins / n_trades

        # Binomial test
        binom_test = stats.binomtest(n_wins, n_trades, test_value, alternative='two-sided')
        p_value = binom_test.pvalue

        # Wilson score interval (better than normal approximation)
        z = norm.ppf(1 - self.alpha / 2)
        denominator = 1 + z**2 / n_trades
        center = (win_rate + z**2 / (2 * n_trades)) / denominator
        margin = z * np.sqrt(win_rate * (1 - win_rate) / n_trades + z**2 / (4 * n_trades**2)) / denominator

        ci_lower = center - margin
        ci_upper = center + margin

        conclusion = (
            f"Win rate of {win_rate:.2%} with {self.confidence_level:.0%} CI "
            f"[{ci_lower:.2%}, {ci_upper:.2%}] is "
            f"{'significantly different from' if p_value < self.alpha else 'not significantly different from'} "
            f"{test_value:.0%}"
        )

        return StatisticalTestResult(
            test_name="Win Rate Confidence Interval (Binomial)",
            statistic=win_rate,
            p_value=p_value,
            confidence_level=self.confidence_level,
            null_hypothesis=f"Win rate equals {test_value}",
            conclusion=conclusion,
            additional_info={
                'win_rate': win_rate,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_wins': n_wins,
                'n_trades': n_trades
            }
        )

    def profit_factor_significance(
        self,
        winning_trades: np.ndarray,
        losing_trades: np.ndarray
    ) -> StatisticalTestResult:
        """
        Test profit factor significance.

        Args:
            winning_trades: Array of winning trade P&L
            losing_trades: Array of losing trade P&L

        Returns:
            StatisticalTestResult with test details
        """
        logger.debug("Calculating profit factor significance")

        if len(losing_trades) == 0:
            logger.warning("No losing trades")
            return StatisticalTestResult(
                test_name="Profit Factor Significance",
                statistic=np.inf,
                p_value=0.0,
                confidence_level=self.confidence_level,
                null_hypothesis="Profit factor equals 1",
                conclusion="No losing trades - infinite profit factor"
            )

        gross_profit = np.sum(winning_trades)
        gross_loss = abs(np.sum(losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Bootstrap confidence interval
        n_bootstrap = 10000
        bootstrap_pfs = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            win_sample = np.random.choice(winning_trades, size=len(winning_trades), replace=True)
            loss_sample = np.random.choice(losing_trades, size=len(losing_trades), replace=True)

            boot_profit = np.sum(win_sample)
            boot_loss = abs(np.sum(loss_sample))
            boot_pf = boot_profit / boot_loss if boot_loss > 0 else np.inf

            if not np.isinf(boot_pf):
                bootstrap_pfs.append(boot_pf)

        bootstrap_pfs = np.array(bootstrap_pfs)

        # p-value: proportion of bootstrap samples with PF <= 1
        p_value = np.mean(bootstrap_pfs <= 1.0) if len(bootstrap_pfs) > 0 else 1.0

        # Confidence interval
        ci_lower, ci_upper = np.percentile(bootstrap_pfs, [self.alpha/2 * 100, (1-self.alpha/2) * 100])

        conclusion = (
            f"Profit factor of {profit_factor:.2f} with {self.confidence_level:.0%} CI "
            f"[{ci_lower:.2f}, {ci_upper:.2f}] is "
            f"{'significantly greater than 1' if p_value < self.alpha else 'not significantly greater than 1'}"
        )

        return StatisticalTestResult(
            test_name="Profit Factor Significance (Bootstrap)",
            statistic=profit_factor,
            p_value=p_value,
            confidence_level=self.confidence_level,
            null_hypothesis="Profit factor equals 1",
            conclusion=conclusion,
            additional_info={
                'profit_factor': profit_factor,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n_bootstrap': n_bootstrap
            }
        )

    def return_difference_test(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray,
        paired: bool = True
    ) -> StatisticalTestResult:
        """
        T-test for difference in returns between two strategies.

        Args:
            returns1: First strategy returns
            returns2: Second strategy returns
            paired: Whether returns are paired (same time periods)

        Returns:
            StatisticalTestResult with test details
        """
        logger.debug(f"Running {'paired' if paired else 'independent'} t-test")

        if paired:
            # Paired t-test
            if len(returns1) != len(returns2):
                raise ValueError("Paired test requires equal length arrays")

            differences = returns1 - returns2
            t_stat, p_value = stats.ttest_1samp(differences, 0)
            test_type = "Paired T-Test"
            mean_diff = np.mean(differences)

        else:
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(returns1, returns2)
            test_type = "Independent T-Test"
            mean_diff = np.mean(returns1) - np.mean(returns2)

        conclusion = (
            f"Mean return difference of {mean_diff:.4f} is "
            f"{'statistically significant' if p_value < self.alpha else 'not statistically significant'} "
            f"at {self.confidence_level:.1%} confidence"
        )

        return StatisticalTestResult(
            test_name=test_type,
            statistic=t_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            null_hypothesis="No difference in mean returns",
            conclusion=conclusion,
            additional_info={
                'mean_difference': mean_diff,
                'mean_returns1': np.mean(returns1),
                'mean_returns2': np.mean(returns2),
                'n_observations1': len(returns1),
                'n_observations2': len(returns2)
            }
        )

    def mann_whitney_u_test(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray
    ) -> StatisticalTestResult:
        """
        Mann-Whitney U test (non-parametric) for return differences.

        Tests whether returns from two strategies come from the same distribution.
        Does not assume normality.

        Args:
            returns1: First strategy returns
            returns2: Second strategy returns

        Returns:
            StatisticalTestResult with test details
        """
        logger.debug("Running Mann-Whitney U test")

        u_stat, p_value = stats.mannwhitneyu(returns1, returns2, alternative='two-sided')

        median_diff = np.median(returns1) - np.median(returns2)

        conclusion = (
            f"Median return difference of {median_diff:.4f} is "
            f"{'statistically significant' if p_value < self.alpha else 'not statistically significant'} "
            f"at {self.confidence_level:.1%} confidence (non-parametric test)"
        )

        return StatisticalTestResult(
            test_name="Mann-Whitney U Test",
            statistic=u_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            null_hypothesis="Returns come from same distribution",
            conclusion=conclusion,
            additional_info={
                'median_difference': median_diff,
                'median_returns1': np.median(returns1),
                'median_returns2': np.median(returns2),
                'n_observations1': len(returns1),
                'n_observations2': len(returns2)
            }
        )

    def runs_test(self, returns: np.ndarray) -> StatisticalTestResult:
        """
        Runs test for randomness in returns.

        Tests whether positive/negative returns are randomly distributed
        or show patterns.

        Args:
            returns: Array of returns

        Returns:
            StatisticalTestResult with test details
        """
        logger.debug("Running runs test for randomness")

        # Convert to binary: positive (1) or negative (0)
        binary = (returns > 0).astype(int)

        # Count runs
        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                runs += 1

        # Count positives and negatives
        n_pos = np.sum(binary)
        n_neg = len(binary) - n_pos

        # Expected runs and variance under randomness
        n = len(binary)
        expected_runs = 1 + (2 * n_pos * n_neg) / n
        variance_runs = (2 * n_pos * n_neg * (2 * n_pos * n_neg - n)) / (n**2 * (n - 1))

        # Z-statistic
        z_stat = (runs - expected_runs) / np.sqrt(variance_runs) if variance_runs > 0 else 0

        # p-value (two-tailed)
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))

        conclusion = (
            f"Observed {runs} runs vs expected {expected_runs:.1f}: "
            f"{'evidence of non-randomness' if p_value < self.alpha else 'consistent with randomness'} "
            f"at {self.confidence_level:.1%} confidence"
        )

        return StatisticalTestResult(
            test_name="Runs Test for Randomness",
            statistic=z_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            null_hypothesis="Returns are randomly distributed",
            conclusion=conclusion,
            additional_info={
                'observed_runs': runs,
                'expected_runs': expected_runs,
                'n_positive': n_pos,
                'n_negative': n_neg,
                'n_observations': n
            }
        )

    def kolmogorov_smirnov_test(
        self,
        returns1: np.ndarray,
        returns2: Optional[np.ndarray] = None,
        distribution: str = 'norm'
    ) -> StatisticalTestResult:
        """
        Kolmogorov-Smirnov test for distribution comparison.

        Can test against theoretical distribution or another sample.

        Args:
            returns1: First array of returns
            returns2: Second array (if comparing two samples) or None
            distribution: Theoretical distribution name ('norm', 'uniform', etc.)

        Returns:
            StatisticalTestResult with test details
        """
        logger.debug(f"Running KS test for {distribution if returns2 is None else 'two samples'}")

        if returns2 is None:
            # Test against theoretical distribution
            if distribution == 'norm':
                # Standardize
                standardized = (returns1 - np.mean(returns1)) / np.std(returns1, ddof=1)
                ks_stat, p_value = stats.kstest(standardized, 'norm')
                test_name = "KS Test (Normal Distribution)"
            else:
                ks_stat, p_value = stats.kstest(returns1, distribution)
                test_name = f"KS Test ({distribution} Distribution)"

            conclusion = (
                f"Returns are {'not consistent with' if p_value < self.alpha else 'consistent with'} "
                f"{distribution} distribution at {self.confidence_level:.1%} confidence"
            )
        else:
            # Two-sample test
            ks_stat, p_value = stats.ks_2samp(returns1, returns2)
            test_name = "KS Test (Two Samples)"

            conclusion = (
                f"Two return distributions are "
                f"{'significantly different' if p_value < self.alpha else 'not significantly different'} "
                f"at {self.confidence_level:.1%} confidence"
            )

        return StatisticalTestResult(
            test_name=test_name,
            statistic=ks_stat,
            p_value=p_value,
            confidence_level=self.confidence_level,
            null_hypothesis="Samples come from same distribution",
            conclusion=conclusion,
            additional_info={
                'ks_statistic': ks_stat,
                'distribution': distribution if returns2 is None else 'two-sample'
            }
        )

    def run_all_tests(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        trades_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, StatisticalTestResult]:
        """
        Run all applicable statistical tests.

        Args:
            returns: Strategy returns
            benchmark_returns: Optional benchmark returns
            trades_df: Optional DataFrame with trade information

        Returns:
            Dictionary of test results
        """
        logger.info("Running comprehensive statistical test suite")

        results = {}

        # Sharpe ratio test
        sharpe = np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252) if np.std(returns) > 0 else 0
        results['sharpe_significance'] = self.sharpe_ratio_significance(returns, sharpe)

        # Information ratio (if benchmark provided)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            results['information_ratio'] = self.information_ratio_test(returns, benchmark_returns)

        # Maximum drawdown
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        mdd = np.min(drawdown)
        results['mdd_distribution'] = self.maximum_drawdown_distribution(returns, mdd)

        # Win rate (if trades provided)
        if trades_df is not None and 'pnl' in trades_df.columns:
            n_wins = len(trades_df[trades_df['pnl'] > 0])
            n_trades = len(trades_df)
            results['win_rate'] = self.win_rate_confidence_interval(n_wins, n_trades)

            # Profit factor
            winning_trades = trades_df[trades_df['pnl'] > 0]['pnl'].values
            losing_trades = trades_df[trades_df['pnl'] < 0]['pnl'].values
            if len(winning_trades) > 0 and len(losing_trades) > 0:
                results['profit_factor'] = self.profit_factor_significance(winning_trades, losing_trades)

        # Runs test
        results['runs_test'] = self.runs_test(returns)

        # KS test for normality
        results['ks_normality'] = self.kolmogorov_smirnov_test(returns, distribution='norm')

        logger.info(f"Completed {len(results)} statistical tests")
        return results
