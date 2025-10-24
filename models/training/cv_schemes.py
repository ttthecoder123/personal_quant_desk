"""
Cross-validation schemes for time series
Based on López de Prado's "Advances in Financial Machine Learning", Chapter 7
"""
import pandas as pd
import numpy as np
from typing import Generator, Tuple, Optional
from loguru import logger


class PurgedKFold:
    """
    Purged K-fold cross-validation (López de Prado Ch. 7)

    Standard K-fold CV doesn't account for label leakage due to overlapping
    observations in time series. Purged K-fold removes observations from the
    training set that overlap with the test set.
    """

    def __init__(self, n_splits: int = 5, purge_gap: int = 10):
        """
        Args:
            n_splits: Number of folds
            purge_gap: Number of observations to exclude after test set
        """
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        logger.info(f"Initialized PurgedKFold with {n_splits} splits, purge_gap={purge_gap}")

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
             groups: Optional[pd.Series] = None) -> Generator:
        """
        Generate train/test indices with purging

        Args:
            X: Feature DataFrame
            y: Target series (unused, for sklearn compatibility)
            groups: Group labels (unused)

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits

        logger.info(f"Splitting {n_samples} samples into {self.n_splits} folds")

        for i in range(self.n_splits):
            # Test indices
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            test_idx = indices[test_start:test_end]

            # Train indices (with purging)
            train_idx = []

            # Before test set
            if test_start > 0:
                purge_before = max(0, test_start - self.purge_gap)
                train_idx.extend(indices[:purge_before])

            # After test set
            if test_end < n_samples:
                purge_after = min(n_samples, test_end + self.purge_gap)
                train_idx.extend(indices[purge_after:])

            logger.debug(f"Fold {i+1}: Train={len(train_idx)}, Test={len(test_idx)}")

            yield np.array(train_idx), test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits"""
        return self.n_splits


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged K-Fold (López de Prado Ch. 7)

    Generates multiple training paths to reduce variance in backtesting
    and provide more robust estimates of model performance.
    """

    def __init__(self, n_splits: int = 5, n_test_groups: int = 2, purge_gap: int = 10):
        """
        Args:
            n_splits: Number of groups to split data into
            n_test_groups: Number of groups to use for testing in each split
            purge_gap: Number of observations to purge
        """
        self.n_splits = n_splits
        self.n_test_groups = n_test_groups
        self.purge_gap = purge_gap
        logger.info(f"Initialized CombinatorialPurgedKFold: splits={n_splits}, test_groups={n_test_groups}")

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Generator:
        """
        Generate combinatorial purged splits

        Args:
            X: Feature DataFrame
            y: Target series

        Yields:
            Tuple of (train_indices, test_indices)
        """
        from itertools import combinations

        n_samples = len(X)
        indices = np.arange(n_samples)
        group_size = n_samples // self.n_splits

        # Create groups
        groups = []
        for i in range(self.n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_splits - 1 else n_samples
            groups.append(indices[start:end])

        # Generate all combinations of test groups
        test_combinations = list(combinations(range(self.n_splits), self.n_test_groups))

        logger.info(f"Generating {len(test_combinations)} combinatorial splits")

        for combo in test_combinations:
            # Test indices from selected groups
            test_idx = np.concatenate([groups[i] for i in combo])

            # Train indices from remaining groups (with purging)
            train_groups = [i for i in range(self.n_splits) if i not in combo]
            train_idx = []

            for train_group in train_groups:
                group_indices = groups[train_group]

                # Check if we need to purge this group
                needs_purging = False
                for test_group in combo:
                    if abs(train_group - test_group) == 1:
                        needs_purging = True
                        break

                if needs_purging:
                    # Purge edges of this group
                    if train_group < min(combo):
                        # Keep only first part (before test)
                        keep_until = max(0, len(group_indices) - self.purge_gap)
                        train_idx.extend(group_indices[:keep_until])
                    else:
                        # Keep only last part (after test)
                        keep_from = min(len(group_indices), self.purge_gap)
                        train_idx.extend(group_indices[keep_from:])
                else:
                    train_idx.extend(group_indices)

            yield np.array(train_idx), test_idx


class WalkForwardCV:
    """
    Walk-forward cross-validation for time series

    More realistic for trading strategies as it mimics the actual
    training and prediction process over time.
    """

    def __init__(self, n_splits: int = 5,
                train_period: int = 252,
                test_period: int = 63,
                gap: int = 10):
        """
        Args:
            n_splits: Number of splits
            train_period: Training period length (days)
            test_period: Test period length (days)
            gap: Gap between train and test (days)
        """
        self.n_splits = n_splits
        self.train_period = train_period
        self.test_period = test_period
        self.gap = gap
        logger.info(f"Initialized WalkForwardCV: {n_splits} splits, train={train_period}, test={test_period}")

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Generator:
        """
        Generate walk-forward train/test splits

        Args:
            X: Feature DataFrame
            y: Target series

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        step = self.test_period

        logger.info(f"Walk-forward CV with {n_samples} samples")

        for i in range(self.n_splits):
            # Calculate indices (moving forward in time)
            test_end = n_samples - i * step
            test_start = max(0, test_end - self.test_period)
            train_end = max(0, test_start - self.gap)
            train_start = max(0, train_end - self.train_period)

            if train_start >= train_end or test_start >= test_end:
                logger.debug(f"Stopping at split {i}: insufficient data")
                continue

            if train_end - train_start < 50:  # Minimum training size
                logger.debug(f"Stopping at split {i}: training set too small")
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            logger.debug(f"Split {i}: Train=[{train_start}:{train_end}], Test=[{test_start}:{test_end}]")

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits"""
        return self.n_splits


class ExpandingWindowCV:
    """
    Expanding window cross-validation

    Training set expands over time while test set moves forward.
    Useful when you want to use all available historical data.
    """

    def __init__(self, n_splits: int = 5,
                min_train_size: int = 252,
                test_size: int = 63,
                gap: int = 10):
        """
        Args:
            n_splits: Number of splits
            min_train_size: Minimum training period
            test_size: Test period size
            gap: Gap between train and test
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.gap = gap
        logger.info(f"Initialized ExpandingWindowCV: {n_splits} splits")

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Generator:
        """
        Generate expanding window splits

        Args:
            X: Feature DataFrame
            y: Target series

        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        step = self.test_size

        for i in range(self.n_splits):
            # Test set moves forward
            test_end = n_samples - i * step
            test_start = max(0, test_end - self.test_size)

            # Training set expands from beginning
            train_end = max(0, test_start - self.gap)
            train_start = 0

            if train_end - train_start < self.min_train_size:
                logger.debug(f"Stopping at split {i}: training set too small")
                continue

            if test_start >= test_end:
                logger.debug(f"Stopping at split {i}: no test data")
                continue

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            logger.debug(f"Split {i}: Train size={len(train_idx)}, Test size={len(test_idx)}")

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits"""
        return self.n_splits


def cross_validate_with_purging(model, X: pd.DataFrame, y: pd.Series,
                                cv_scheme: str = 'purged',
                                n_splits: int = 5,
                                purge_gap: int = 10) -> dict:
    """
    Convenience function for cross-validation with purging

    Args:
        model: Sklearn-compatible model
        X: Features
        y: Labels
        cv_scheme: 'purged', 'walk_forward', or 'expanding'
        n_splits: Number of splits
        purge_gap: Purge gap

    Returns:
        Dictionary with CV results
    """
    from sklearn.metrics import accuracy_score, f1_score

    logger.info(f"Running {cv_scheme} cross-validation with {n_splits} splits")

    # Select CV scheme
    if cv_scheme == 'purged':
        cv = PurgedKFold(n_splits=n_splits, purge_gap=purge_gap)
    elif cv_scheme == 'walk_forward':
        cv = WalkForwardCV(n_splits=n_splits, gap=purge_gap)
    elif cv_scheme == 'expanding':
        cv = ExpandingWindowCV(n_splits=n_splits, gap=purge_gap)
    else:
        raise ValueError(f"Unknown CV scheme: {cv_scheme}")

    # Run CV
    scores = {'accuracy': [], 'f1': []}

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        scores['accuracy'].append(accuracy_score(y_test, y_pred))
        scores['f1'].append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

    # Summary statistics
    results = {
        'mean_accuracy': np.mean(scores['accuracy']),
        'std_accuracy': np.std(scores['accuracy']),
        'mean_f1': np.mean(scores['f1']),
        'std_f1': np.std(scores['f1']),
        'scores': scores
    }

    logger.info(f"CV results: accuracy={results['mean_accuracy']:.4f}±{results['std_accuracy']:.4f}")

    return results
