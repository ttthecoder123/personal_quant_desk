"""
Machine learning based signal generation
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from loguru import logger


class MLSignals:
    """
    Machine learning based signal generation

    Trains classifiers on features to predict triple-barrier labels,
    generating signals based on ML predictions.
    """

    def __init__(self, features: pd.DataFrame, labels: pd.DataFrame):
        """
        Initialize with features and labels

        Args:
            features: Feature DataFrame
            labels: Labels from triple-barrier labeling
        """
        self.features = features
        self.labels = labels
        self.models = {}
        logger.info(f"Initialized MLSignals with {len(features)} features, {len(labels)} labels")

    def prepare_data(self, test_size: float = 0.2,
                    sample_weights: Optional[pd.Series] = None) -> Tuple:
        """
        Prepare data for ML training with proper time series split

        Args:
            test_size: Fraction of data for testing
            sample_weights: Optional sample weights

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, weights_train, weights_test)
        """
        logger.info(f"Preparing data with test_size={test_size}")

        # Remove NaN values
        common_idx = self.features.index.intersection(self.labels.index)
        X = self.features.loc[common_idx]
        y = self.labels.loc[common_idx, 'label']

        # Remove rows with NaN
        valid_idx = y.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        # Further remove features with NaN
        X = X.dropna()
        y = y.loc[X.index]

        logger.info(f"After cleaning: {len(X)} samples with {X.shape[1]} features")

        # Time series split (no shuffling!)
        split_idx = int(len(X) * (1 - test_size))

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        # Handle sample weights
        weights_train = None
        weights_test = None
        if sample_weights is not None:
            weights = sample_weights.loc[X.index]
            weights_train = weights.iloc[:split_idx]
            weights_test = weights.iloc[split_idx:]

        logger.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        logger.info(f"Train label distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test label distribution: {y_test.value_counts().to_dict()}")

        return X_train, X_test, y_train, y_test, weights_train, weights_test

    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series,
                           sample_weight: Optional[pd.Series] = None,
                           **kwargs) -> RandomForestClassifier:
        """
        Train Random Forest classifier

        Args:
            X_train: Training features
            y_train: Training labels
            sample_weight: Optional sample weights
            **kwargs: Additional parameters for RandomForest

        Returns:
            Trained RandomForestClassifier
        """
        logger.info("Training Random Forest")

        # Convert to classification problem (-1, 0, 1)
        y_class = y_train.copy()

        # Default parameters
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 50,
            'min_samples_leaf': 25,
            'max_features': 'sqrt',
            'random_state': 42,
            'n_jobs': -1
        }
        params.update(kwargs)

        rf = RandomForestClassifier(**params)

        if sample_weight is not None:
            rf.fit(X_train, y_class, sample_weight=sample_weight)
        else:
            rf.fit(X_train, y_class)

        # Calculate training accuracy
        train_acc = rf.score(X_train, y_class)
        logger.info(f"Random Forest trained. Training accuracy: {train_acc:.4f}")

        self.models['random_forest'] = rf
        return rf

    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                      sample_weight: Optional[pd.Series] = None,
                      **kwargs) -> object:
        """
        Train LightGBM classifier

        Args:
            X_train: Training features
            y_train: Training labels
            sample_weight: Optional sample weights
            **kwargs: Additional parameters for LightGBM

        Returns:
            Trained LGBMClassifier
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("LightGBM not installed. Install with: pip install lightgbm")
            return None

        logger.info("Training LightGBM")

        # Convert to classification
        y_class = y_train.copy()

        # Default parameters
        params = {
            'n_estimators': 100,
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        params.update(kwargs)

        lgbm = lgb.LGBMClassifier(**params)

        callbacks = [lgb.early_stopping(10, verbose=False), lgb.log_evaluation(0)]

        if sample_weight is not None:
            lgbm.fit(
                X_train, y_class,
                sample_weight=sample_weight,
                eval_set=[(X_train, y_class)],
                callbacks=callbacks
            )
        else:
            lgbm.fit(
                X_train, y_class,
                eval_set=[(X_train, y_class)],
                callbacks=callbacks
            )

        train_acc = lgbm.score(X_train, y_class)
        logger.info(f"LightGBM trained. Training accuracy: {train_acc:.4f}")

        self.models['lightgbm'] = lgbm
        return lgbm

    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                      sample_weight: Optional[pd.Series] = None) -> object:
        """
        Train ensemble of models

        Args:
            X_train: Training features
            y_train: Training labels
            sample_weight: Optional sample weights

        Returns:
            Trained ensemble model
        """
        from sklearn.ensemble import VotingClassifier

        logger.info("Training ensemble model")

        # Train individual models
        rf = self.train_random_forest(X_train, y_train, sample_weight)
        lgbm = self.train_lightgbm(X_train, y_train, sample_weight)

        if lgbm is None:
            logger.warning("LightGBM training failed, using Random Forest only")
            self.models['ensemble'] = rf
            return rf

        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('lgbm', lgbm)
            ],
            voting='soft'  # Use probabilities
        )

        # Convert to classification
        y_class = y_train.copy()

        if sample_weight is not None:
            ensemble.fit(X_train, y_class, sample_weight=sample_weight)
        else:
            ensemble.fit(X_train, y_class)

        train_acc = ensemble.score(X_train, y_class)
        logger.info(f"Ensemble trained. Training accuracy: {train_acc:.4f}")

        self.models['ensemble'] = ensemble
        return ensemble

    def generate_signals(self, model_name: str = 'ensemble',
                        X: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Generate trading signals from trained model

        Args:
            model_name: Name of the model to use
            X: Features to generate signals for (default: self.features)

        Returns:
            Signal series (-1, 0, 1)
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.models[model_name]

        if X is None:
            X = self.features

        logger.info(f"Generating signals using {model_name}")

        # Clean features
        X_clean = X.dropna()

        # Generate predictions
        predictions = model.predict(X_clean)

        signals = pd.Series(predictions, index=X_clean.index)

        logger.info(f"Generated {len(signals)} signals: {signals.value_counts().to_dict()}")

        return signals

    def generate_signal_probabilities(self, model_name: str = 'ensemble',
                                     X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate signal probabilities instead of hard predictions

        Args:
            model_name: Name of the model to use
            X: Features to generate signals for

        Returns:
            DataFrame with probabilities for each class
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        model = self.models[model_name]

        if X is None:
            X = self.features

        logger.info(f"Generating signal probabilities using {model_name}")

        # Clean features
        X_clean = X.dropna()

        # Generate probabilities
        probas = model.predict_proba(X_clean)

        # Get class labels
        classes = model.classes_

        # Create DataFrame
        proba_df = pd.DataFrame(probas, index=X_clean.index, columns=classes)

        logger.info(f"Generated probabilities for {len(proba_df)} samples")

        return proba_df

    def evaluate_model(self, model_name: str, X_test: pd.DataFrame,
                      y_test: pd.Series) -> Dict:
        """
        Evaluate model performance

        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        model = self.models[model_name]

        # Generate predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }

        logger.info(f"{model_name} metrics: {metrics}")

        return metrics
