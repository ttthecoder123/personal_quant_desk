"""
Trade-level analysis module.

This module provides comprehensive analysis of individual trades including:
- Entry/exit analysis
- Hold time distribution
- P&L distribution
- Win/loss streaks
- Best/worst trades
- Trade clustering
- Correlation between trades
- Slippage and commission analysis
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, KMeans


@dataclass
class TradeStatistics:
    """Container for trade statistics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int
    win_rate: float
    loss_rate: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    largest_win: float
    largest_loss: float
    payoff_ratio: float
    profit_factor: float
    expectancy: float
    consecutive_wins: int
    consecutive_losses: int
    avg_hold_time: float
    median_hold_time: float


class TradeAnalyzer:
    """
    Comprehensive trade-level analysis.

    Analyzes individual trades to identify patterns, strengths, and weaknesses
    in trading strategy execution.
    """

    def __init__(self):
        """Initialize trade analyzer."""
        logger.info("TradeAnalyzer initialized")

    def analyze_trades(self, trades: pd.DataFrame) -> Dict:
        """
        Perform comprehensive trade analysis.

        Args:
            trades: DataFrame with columns: entry_date, exit_date, entry_price,
                   exit_price, quantity, pnl, symbol, etc.

        Returns:
            Dictionary containing all trade analysis results
        """
        logger.info(f"Analyzing {len(trades)} trades")

        if trades.empty:
            logger.warning("Empty trades DataFrame provided")
            return self._empty_analysis()

        results = {
            'statistics': self.calculate_statistics(trades),
            'entry_exit_analysis': self.analyze_entry_exit(trades),
            'hold_time_analysis': self.analyze_hold_times(trades),
            'pnl_distribution': self.analyze_pnl_distribution(trades),
            'streaks': self.detect_streaks(trades),
            'best_worst_trades': self.identify_best_worst_trades(trades),
            'slippage_commission': self.analyze_costs(trades),
        }

        logger.success("Trade analysis completed")
        return results

    def calculate_statistics(self, trades: pd.DataFrame) -> TradeStatistics:
        """
        Calculate comprehensive trade statistics.

        Args:
            trades: Trades DataFrame

        Returns:
            TradeStatistics object
        """
        if 'pnl' not in trades.columns:
            logger.warning("No 'pnl' column in trades DataFrame")
            return self._empty_statistics()

        pnl = trades['pnl'].dropna()

        winning_trades = pnl[pnl > 0]
        losing_trades = pnl[pnl < 0]
        breakeven_trades = pnl[pnl == 0]

        total_trades = len(pnl)
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        num_breakeven = len(breakeven_trades)

        win_rate = num_wins / total_trades if total_trades > 0 else 0
        loss_rate = num_losses / total_trades if total_trades > 0 else 0

        avg_win = winning_trades.mean() if num_wins > 0 else 0
        avg_loss = abs(losing_trades.mean()) if num_losses > 0 else 0
        avg_trade = pnl.mean()

        largest_win = winning_trades.max() if num_wins > 0 else 0
        largest_loss = abs(losing_trades.min()) if num_losses > 0 else 0

        payoff_ratio = avg_win / avg_loss if avg_loss != 0 else 0

        total_profit = winning_trades.sum() if num_wins > 0 else 0
        total_loss = abs(losing_trades.sum()) if num_losses > 0 else 0
        profit_factor = total_profit / total_loss if total_loss != 0 else 0

        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        # Calculate consecutive wins/losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive_streaks(pnl)

        # Calculate hold times if available
        avg_hold_time = 0
        median_hold_time = 0
        if 'entry_date' in trades.columns and 'exit_date' in trades.columns:
            hold_times = pd.to_datetime(trades['exit_date']) - pd.to_datetime(trades['entry_date'])
            avg_hold_time = hold_times.mean().total_seconds() / 3600  # hours
            median_hold_time = hold_times.median().total_seconds() / 3600

        return TradeStatistics(
            total_trades=total_trades,
            winning_trades=num_wins,
            losing_trades=num_losses,
            breakeven_trades=num_breakeven,
            win_rate=win_rate,
            loss_rate=loss_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            largest_win=largest_win,
            largest_loss=largest_loss,
            payoff_ratio=payoff_ratio,
            profit_factor=profit_factor,
            expectancy=expectancy,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            avg_hold_time=avg_hold_time,
            median_hold_time=median_hold_time,
        )

    def _calculate_consecutive_streaks(self, pnl: pd.Series) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses."""
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for value in pnl:
            if value > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif value < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
            else:
                current_wins = 0
                current_losses = 0

        return max_wins, max_losses

    def analyze_entry_exit(self, trades: pd.DataFrame) -> Dict:
        """
        Analyze entry and exit quality.

        Args:
            trades: Trades DataFrame

        Returns:
            Dictionary with entry/exit analysis
        """
        logger.debug("Analyzing entry/exit patterns")

        results = {}

        # Entry time analysis (if timestamp available)
        if 'entry_date' in trades.columns:
            entry_times = pd.to_datetime(trades['entry_date'])

            # Hour of day analysis
            entry_hours = entry_times.dt.hour
            results['entry_hour_distribution'] = entry_hours.value_counts().sort_index().to_dict()

            # Day of week analysis
            entry_days = entry_times.dt.dayofweek
            results['entry_day_distribution'] = entry_days.value_counts().sort_index().to_dict()

        # Exit time analysis
        if 'exit_date' in trades.columns:
            exit_times = pd.to_datetime(trades['exit_date'])

            exit_hours = exit_times.dt.hour
            results['exit_hour_distribution'] = exit_hours.value_counts().sort_index().to_dict()

            exit_days = exit_times.dt.dayofweek
            results['exit_day_distribution'] = exit_days.value_counts().sort_index().to_dict()

        # Entry/exit price analysis
        if 'entry_price' in trades.columns and 'exit_price' in trades.columns:
            price_change_pct = ((trades['exit_price'] - trades['entry_price']) /
                               trades['entry_price'] * 100)

            results['avg_price_change_pct'] = price_change_pct.mean()
            results['median_price_change_pct'] = price_change_pct.median()
            results['price_change_std'] = price_change_pct.std()

        return results

    def analyze_hold_times(self, trades: pd.DataFrame) -> Dict:
        """
        Analyze trade hold time distribution.

        Args:
            trades: Trades DataFrame

        Returns:
            Dictionary with hold time analysis
        """
        logger.debug("Analyzing hold times")

        if 'entry_date' not in trades.columns or 'exit_date' not in trades.columns:
            logger.warning("Missing entry_date or exit_date columns")
            return {}

        entry_times = pd.to_datetime(trades['entry_date'])
        exit_times = pd.to_datetime(trades['exit_date'])

        hold_times = (exit_times - entry_times).dt.total_seconds() / 3600  # hours

        results = {
            'mean_hold_time_hours': hold_times.mean(),
            'median_hold_time_hours': hold_times.median(),
            'std_hold_time_hours': hold_times.std(),
            'min_hold_time_hours': hold_times.min(),
            'max_hold_time_hours': hold_times.max(),
            'percentiles': {
                '25': hold_times.quantile(0.25),
                '50': hold_times.quantile(0.50),
                '75': hold_times.quantile(0.75),
                '90': hold_times.quantile(0.90),
                '95': hold_times.quantile(0.95),
            }
        }

        # Correlation between hold time and P&L
        if 'pnl' in trades.columns:
            correlation = hold_times.corr(trades['pnl'])
            results['hold_time_pnl_correlation'] = correlation

        return results

    def analyze_pnl_distribution(self, trades: pd.DataFrame) -> Dict:
        """
        Analyze P&L distribution characteristics.

        Args:
            trades: Trades DataFrame

        Returns:
            Dictionary with P&L distribution analysis
        """
        logger.debug("Analyzing P&L distribution")

        if 'pnl' not in trades.columns:
            return {}

        pnl = trades['pnl'].dropna()

        # Basic statistics
        results = {
            'mean': pnl.mean(),
            'median': pnl.median(),
            'std': pnl.std(),
            'skewness': stats.skew(pnl),
            'kurtosis': stats.kurtosis(pnl),
            'min': pnl.min(),
            'max': pnl.max(),
        }

        # Percentiles
        results['percentiles'] = {
            '5': pnl.quantile(0.05),
            '25': pnl.quantile(0.25),
            '50': pnl.quantile(0.50),
            '75': pnl.quantile(0.75),
            '95': pnl.quantile(0.95),
        }

        # Normality test
        _, p_value = stats.normaltest(pnl)
        results['normality_test_p_value'] = p_value
        results['is_normal_distribution'] = p_value > 0.05

        # Outlier detection (using IQR method)
        q1 = pnl.quantile(0.25)
        q3 = pnl.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = pnl[(pnl < lower_bound) | (pnl > upper_bound)]
        results['num_outliers'] = len(outliers)
        results['outlier_percentage'] = len(outliers) / len(pnl) * 100

        return results

    def detect_streaks(self, trades: pd.DataFrame) -> Dict:
        """
        Detect winning and losing streaks.

        Args:
            trades: Trades DataFrame

        Returns:
            Dictionary with streak information
        """
        logger.debug("Detecting win/loss streaks")

        if 'pnl' not in trades.columns:
            return {}

        pnl = trades['pnl'].dropna()

        # Identify streaks
        streaks = []
        current_streak = {'type': None, 'length': 0, 'total_pnl': 0}

        for value in pnl:
            if value > 0:
                if current_streak['type'] == 'win':
                    current_streak['length'] += 1
                    current_streak['total_pnl'] += value
                else:
                    if current_streak['type'] is not None:
                        streaks.append(current_streak.copy())
                    current_streak = {'type': 'win', 'length': 1, 'total_pnl': value}
            elif value < 0:
                if current_streak['type'] == 'loss':
                    current_streak['length'] += 1
                    current_streak['total_pnl'] += value
                else:
                    if current_streak['type'] is not None:
                        streaks.append(current_streak.copy())
                    current_streak = {'type': 'loss', 'length': 1, 'total_pnl': value}

        # Add final streak
        if current_streak['type'] is not None:
            streaks.append(current_streak)

        # Analyze streaks
        winning_streaks = [s for s in streaks if s['type'] == 'win']
        losing_streaks = [s for s in streaks if s['type'] == 'loss']

        results = {
            'total_streaks': len(streaks),
            'winning_streaks': len(winning_streaks),
            'losing_streaks': len(losing_streaks),
        }

        if winning_streaks:
            results['max_winning_streak'] = max(s['length'] for s in winning_streaks)
            results['avg_winning_streak'] = np.mean([s['length'] for s in winning_streaks])
            results['max_winning_streak_pnl'] = max(s['total_pnl'] for s in winning_streaks)

        if losing_streaks:
            results['max_losing_streak'] = max(s['length'] for s in losing_streaks)
            results['avg_losing_streak'] = np.mean([s['length'] for s in losing_streaks])
            results['max_losing_streak_pnl'] = min(s['total_pnl'] for s in losing_streaks)

        return results

    def identify_best_worst_trades(
        self,
        trades: pd.DataFrame,
        n: int = 10
    ) -> Dict:
        """
        Identify best and worst trades.

        Args:
            trades: Trades DataFrame
            n: Number of top/bottom trades to return

        Returns:
            Dictionary with best/worst trades
        """
        logger.debug(f"Identifying top {n} best and worst trades")

        if 'pnl' not in trades.columns:
            return {}

        # Sort by P&L
        sorted_trades = trades.sort_values('pnl', ascending=False)

        best_trades = sorted_trades.head(n)
        worst_trades = sorted_trades.tail(n)

        results = {
            'best_trades': best_trades[['symbol', 'entry_date', 'exit_date', 'pnl']].to_dict('records')
            if all(col in best_trades.columns for col in ['symbol', 'entry_date', 'exit_date', 'pnl'])
            else best_trades['pnl'].to_dict(),

            'worst_trades': worst_trades[['symbol', 'entry_date', 'exit_date', 'pnl']].to_dict('records')
            if all(col in worst_trades.columns for col in ['symbol', 'entry_date', 'exit_date', 'pnl'])
            else worst_trades['pnl'].to_dict(),
        }

        # Analyze characteristics of best/worst trades
        if 'symbol' in trades.columns:
            # Most profitable symbols
            symbol_pnl = trades.groupby('symbol')['pnl'].sum().sort_values(ascending=False)
            results['most_profitable_symbols'] = symbol_pnl.head(n).to_dict()
            results['least_profitable_symbols'] = symbol_pnl.tail(n).to_dict()

        return results

    def analyze_costs(self, trades: pd.DataFrame) -> Dict:
        """
        Analyze slippage and commission costs.

        Args:
            trades: Trades DataFrame

        Returns:
            Dictionary with cost analysis
        """
        logger.debug("Analyzing trading costs")

        results = {}

        if 'commission' in trades.columns:
            commission = trades['commission'].dropna()
            results['total_commission'] = commission.sum()
            results['avg_commission_per_trade'] = commission.mean()
            results['commission_std'] = commission.std()

        if 'slippage' in trades.columns:
            slippage = trades['slippage'].dropna()
            results['total_slippage'] = slippage.sum()
            results['avg_slippage_per_trade'] = slippage.mean()
            results['slippage_std'] = slippage.std()

        # Total costs
        if 'commission' in trades.columns and 'slippage' in trades.columns:
            total_costs = trades['commission'] + trades['slippage']
            results['total_costs'] = total_costs.sum()
            results['avg_costs_per_trade'] = total_costs.mean()

            # Cost as percentage of P&L
            if 'pnl' in trades.columns:
                gross_pnl = trades['pnl'] + total_costs
                cost_percentage = (total_costs.sum() / gross_pnl.sum() * 100
                                 if gross_pnl.sum() != 0 else 0)
                results['costs_as_pct_of_pnl'] = cost_percentage

        return results

    def plot_hold_time_distribution(
        self,
        trades: pd.DataFrame,
        bins: int = 30
    ) -> plt.Figure:
        """
        Plot hold time distribution histogram.

        Args:
            trades: Trades DataFrame
            bins: Number of histogram bins

        Returns:
            Matplotlib figure
        """
        if 'entry_date' not in trades.columns or 'exit_date' not in trades.columns:
            logger.warning("Cannot plot hold time: missing date columns")
            return plt.figure()

        entry_times = pd.to_datetime(trades['entry_date'])
        exit_times = pd.to_datetime(trades['exit_date'])
        hold_times = (exit_times - entry_times).dt.total_seconds() / 3600  # hours

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(hold_times, bins=bins, alpha=0.7, edgecolor='black')
        ax.set_title('Hold Time Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hold Time (hours)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add mean and median lines
        mean_hold = hold_times.mean()
        median_hold = hold_times.median()

        ax.axvline(mean_hold, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_hold:.1f}h')
        ax.axvline(median_hold, color='green', linestyle='--', linewidth=2,
                  label=f'Median: {median_hold:.1f}h')

        ax.legend()
        plt.tight_layout()

        return fig

    def plot_pnl_distribution(
        self,
        trades: pd.DataFrame,
        bins: int = 50
    ) -> plt.Figure:
        """
        Plot P&L distribution with histogram and Q-Q plot.

        Args:
            trades: Trades DataFrame
            bins: Number of histogram bins

        Returns:
            Matplotlib figure
        """
        if 'pnl' not in trades.columns:
            logger.warning("Cannot plot P&L: missing pnl column")
            return plt.figure()

        pnl = trades['pnl'].dropna()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(pnl, bins=bins, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Breakeven')
        ax1.axvline(pnl.mean(), color='green', linestyle='--', linewidth=2,
                   label=f'Mean: ${pnl.mean():.2f}')
        ax1.set_title('P&L Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('P&L ($)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q plot
        stats.probplot(pnl, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Test)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_cumulative_pnl(self, trades: pd.DataFrame) -> plt.Figure:
        """
        Plot cumulative P&L over time.

        Args:
            trades: Trades DataFrame

        Returns:
            Matplotlib figure
        """
        if 'pnl' not in trades.columns:
            logger.warning("Cannot plot cumulative P&L: missing pnl column")
            return plt.figure()

        cum_pnl = trades['pnl'].cumsum()

        fig, ax = plt.subplots(figsize=(12, 6))

        if 'exit_date' in trades.columns:
            dates = pd.to_datetime(trades['exit_date'])
            ax.plot(dates, cum_pnl, linewidth=2)
            ax.set_xlabel('Date', fontsize=12)
            plt.xticks(rotation=45)
        else:
            ax.plot(cum_pnl.values, linewidth=2)
            ax.set_xlabel('Trade Number', fontsize=12)

        ax.set_title('Cumulative P&L', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cumulative P&L ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        plt.tight_layout()
        return fig

    def plot_win_loss_streaks(self, trades: pd.DataFrame) -> plt.Figure:
        """
        Plot win/loss streak visualization.

        Args:
            trades: Trades DataFrame

        Returns:
            Matplotlib figure
        """
        if 'pnl' not in trades.columns:
            logger.warning("Cannot plot streaks: missing pnl column")
            return plt.figure()

        pnl = trades['pnl'].dropna()

        # Create win/loss indicator
        results = pnl.apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['green' if r > 0 else 'red' if r < 0 else 'gray' for r in results]
        ax.bar(range(len(results)), results, color=colors, alpha=0.7, edgecolor='black')

        ax.set_title('Win/Loss Streak Pattern', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trade Number', fontsize=12)
        ax.set_ylabel('Result (1=Win, -1=Loss)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linewidth=1)

        plt.tight_layout()
        return fig

    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure."""
        return {
            'statistics': self._empty_statistics(),
            'entry_exit_analysis': {},
            'hold_time_analysis': {},
            'pnl_distribution': {},
            'streaks': {},
            'best_worst_trades': {},
            'slippage_commission': {},
        }

    def _empty_statistics(self) -> TradeStatistics:
        """Return empty statistics object."""
        return TradeStatistics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            win_rate=0.0,
            loss_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            avg_trade=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            payoff_ratio=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            consecutive_wins=0,
            consecutive_losses=0,
            avg_hold_time=0.0,
            median_hold_time=0.0,
        )


class TradeClustering:
    """
    Clustering analysis for trades.

    Groups similar trades together to identify patterns and anomalies.
    """

    def __init__(self):
        """Initialize trade clustering."""
        logger.info("TradeClustering initialized")

    def cluster_trades(
        self,
        trades: pd.DataFrame,
        method: str = 'kmeans',
        n_clusters: int = 3,
        features: Optional[List[str]] = None
    ) -> Dict:
        """
        Cluster trades based on characteristics.

        Args:
            trades: Trades DataFrame
            method: Clustering method ('kmeans', 'dbscan', 'hierarchical')
            n_clusters: Number of clusters (for kmeans/hierarchical)
            features: List of features to use for clustering

        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Clustering trades using {method}")

        if features is None:
            features = self._select_default_features(trades)

        # Prepare feature matrix
        X = trades[features].dropna()

        if len(X) == 0:
            logger.warning("No valid features for clustering")
            return {}

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform clustering
        if method == 'kmeans':
            labels = self._kmeans_clustering(X_scaled, n_clusters)
        elif method == 'dbscan':
            labels = self._dbscan_clustering(X_scaled)
        elif method == 'hierarchical':
            labels = self._hierarchical_clustering(X_scaled, n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Analyze clusters
        results = self._analyze_clusters(trades, labels, features)

        return results

    def _select_default_features(self, trades: pd.DataFrame) -> List[str]:
        """Select default features for clustering."""
        available_features = []

        if 'pnl' in trades.columns:
            available_features.append('pnl')

        if 'entry_price' in trades.columns and 'exit_price' in trades.columns:
            trades['price_change_pct'] = ((trades['exit_price'] - trades['entry_price']) /
                                         trades['entry_price'] * 100)
            available_features.append('price_change_pct')

        if 'entry_date' in trades.columns and 'exit_date' in trades.columns:
            trades['hold_time_hours'] = ((pd.to_datetime(trades['exit_date']) -
                                         pd.to_datetime(trades['entry_date'])).dt.total_seconds() / 3600)
            available_features.append('hold_time_hours')

        if 'commission' in trades.columns:
            available_features.append('commission')

        return available_features

    def _kmeans_clustering(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform K-means clustering."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        return labels

    def _dbscan_clustering(self, X: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering."""
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(X)
        return labels

    def _hierarchical_clustering(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform hierarchical clustering."""
        from sklearn.cluster import AgglomerativeClustering

        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(X)
        return labels

    def _analyze_clusters(
        self,
        trades: pd.DataFrame,
        labels: np.ndarray,
        features: List[str]
    ) -> Dict:
        """Analyze clustering results."""
        trades_with_labels = trades.copy()
        trades_with_labels['cluster'] = labels

        results = {
            'n_clusters': len(np.unique(labels)),
            'cluster_sizes': {},
            'cluster_statistics': {},
        }

        for cluster_id in np.unique(labels):
            cluster_trades = trades_with_labels[trades_with_labels['cluster'] == cluster_id]

            results['cluster_sizes'][int(cluster_id)] = len(cluster_trades)

            cluster_stats = {}
            for feature in features:
                if feature in cluster_trades.columns:
                    cluster_stats[feature] = {
                        'mean': float(cluster_trades[feature].mean()),
                        'std': float(cluster_trades[feature].std()),
                        'min': float(cluster_trades[feature].min()),
                        'max': float(cluster_trades[feature].max()),
                    }

            results['cluster_statistics'][int(cluster_id)] = cluster_stats

        return results

    def plot_clusters(
        self,
        trades: pd.DataFrame,
        labels: np.ndarray,
        feature_x: str,
        feature_y: str
    ) -> plt.Figure:
        """
        Plot 2D visualization of clusters.

        Args:
            trades: Trades DataFrame
            labels: Cluster labels
            feature_x: Feature for x-axis
            feature_y: Feature for y-axis

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        scatter = ax.scatter(
            trades[feature_x],
            trades[feature_y],
            c=labels,
            cmap='viridis',
            alpha=0.6,
            edgecolors='black',
            s=50
        )

        ax.set_xlabel(feature_x, fontsize=12)
        ax.set_ylabel(feature_y, fontsize=12)
        ax.set_title('Trade Clusters', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax, label='Cluster')
        plt.tight_layout()

        return fig
