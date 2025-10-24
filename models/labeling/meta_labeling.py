"""
Meta-labeling implementation
Based on López de Prado's "Advances in Financial Machine Learning", Chapter 3
"""
import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger


class MetaLabeling:
    """
    Meta-labeling for learning bet size (López de Prado Ch. 3)

    Meta-labeling is a secondary ML model that learns when to act on
    the primary model's signals. Instead of predicting direction,
    it predicts whether the primary model's prediction will be correct.
    """

    def __init__(self, primary_signals: pd.Series, prices: pd.Series):
        """
        Initialize with primary model signals

        Args:
            primary_signals: Series of primary model predictions (1, -1, 0)
            prices: Price series for calculating outcomes
        """
        self.primary_signals = primary_signals
        self.prices = prices
        logger.info(f"Initialized MetaLabeling with {len(primary_signals)} signals")

    def generate_meta_labels(self, triple_barrier_labels: pd.DataFrame) -> pd.DataFrame:
        """
        Generate meta-labels for sizing bets on primary signals

        Args:
            triple_barrier_labels: Output from triple-barrier labeling

        Returns:
            DataFrame with meta-labels and features
        """
        meta_labels = pd.DataFrame(index=self.primary_signals.index)

        # Binary classification: should we take the primary signal?
        # 1: Take the signal, 0: Skip the signal
        meta_labels['meta_label'] = 0

        for idx in self.primary_signals.index:
            if idx not in triple_barrier_labels.index:
                continue

            primary_side = self.primary_signals[idx]
            actual_return = triple_barrier_labels.loc[idx, 'return']

            # Skip if no primary signal or no actual return
            if primary_side == 0 or pd.isna(actual_return):
                continue

            # Meta-label is 1 if primary signal was correct
            meta_labels.loc[idx, 'meta_label'] = int(
                np.sign(actual_return) == np.sign(primary_side)
            )

        # Add probability calibration features
        meta_labels['primary_signal'] = self.primary_signals
        meta_labels['signal_strength'] = np.abs(self.primary_signals)

        logger.info(f"Generated {len(meta_labels.dropna())} meta-labels")
        logger.info(f"Meta-label distribution: {meta_labels['meta_label'].value_counts().to_dict()}")

        return meta_labels

    def train_meta_model(self, features: pd.DataFrame,
                        meta_labels: pd.Series,
                        calibration_method: str = 'isotonic') -> object:
        """
        Train a model to predict when primary signals will be successful

        Args:
            features: Feature matrix
            meta_labels: Meta-labels (0 or 1)
            calibration_method: Method for probability calibration ('isotonic' or 'sigmoid')

        Returns:
            Trained meta-model
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV

        # Align features and labels
        common_idx = features.index.intersection(meta_labels.index)
        X = features.loc[common_idx]
        y = meta_labels.loc[common_idx]

        # Remove NaN values
        valid_idx = y.dropna().index
        X = X.loc[valid_idx].dropna()
        y = y.loc[X.index]

        logger.info(f"Training meta-model on {len(X)} samples")

        # Train random forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=50,
            min_samples_leaf=25,
            random_state=42,
            n_jobs=-1
        )

        # Calibrate probabilities
        calibrated_clf = CalibratedClassifierCV(
            rf,
            method=calibration_method,
            cv=3
        )

        calibrated_clf.fit(X, y)

        logger.info(f'Meta-model trained successfully')
        logger.info(f'Training accuracy: {calibrated_clf.score(X, y):.4f}')

        return calibrated_clf

    def predict_bet_size(self, meta_model: object, features: pd.DataFrame) -> pd.Series:
        """
        Predict bet size using meta-model

        Args:
            meta_model: Trained meta-model
            features: Feature matrix

        Returns:
            Series of bet sizes (probabilities)
        """
        # Get probabilities for class 1 (take the bet)
        features_clean = features.dropna()
        bet_sizes = meta_model.predict_proba(features_clean)[:, 1]

        bet_size_series = pd.Series(bet_sizes, index=features_clean.index)

        logger.info(f"Generated {len(bet_size_series)} bet sizes")
        logger.info(f"Mean bet size: {bet_size_series.mean():.4f}")

        return bet_size_series

    def apply_meta_labels(self, primary_signals: pd.Series,
                         bet_sizes: pd.Series) -> pd.Series:
        """
        Apply meta-labels to scale primary signals

        Args:
            primary_signals: Primary model signals
            bet_sizes: Meta-model bet sizes

        Returns:
            Scaled signals
        """
        # Align indices
        common_idx = primary_signals.index.intersection(bet_sizes.index)

        # Scale signals by bet sizes
        scaled_signals = primary_signals.loc[common_idx] * bet_sizes.loc[common_idx]

        logger.info(f"Applied meta-labels to {len(scaled_signals)} signals")

        return scaled_signals


def create_meta_labeling_pipeline(
    primary_signals: pd.Series,
    prices: pd.Series,
    triple_barrier_labels: pd.DataFrame,
    features: pd.DataFrame,
    train: bool = True
) -> tuple:
    """
    Convenience function to create complete meta-labeling pipeline

    Args:
        primary_signals: Primary model signals
        prices: Price series
        triple_barrier_labels: Triple-barrier labels
        features: Feature matrix
        train: Whether to train the model

    Returns:
        Tuple of (meta_labeler, meta_model, bet_sizes)
    """
    # Create meta-labeler
    meta_labeler = MetaLabeling(primary_signals, prices)

    # Generate meta-labels
    meta_labels_df = meta_labeler.generate_meta_labels(triple_barrier_labels)

    meta_model = None
    bet_sizes = None

    if train:
        # Train meta-model
        meta_model = meta_labeler.train_meta_model(
            features,
            meta_labels_df['meta_label']
        )

        # Predict bet sizes
        bet_sizes = meta_labeler.predict_bet_size(meta_model, features)

    return meta_labeler, meta_model, bet_sizes
