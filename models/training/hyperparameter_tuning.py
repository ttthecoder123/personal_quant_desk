"""
Hyperparameter tuning with cross-validation
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from loguru import logger


class HyperparameterTuner:
    """
    Hyperparameter tuning for trading models

    Uses cross-validation to find optimal model parameters
    while avoiding overfitting through purged CV schemes.
    """

    def __init__(self, cv_scheme: str = 'purged', n_splits: int = 5):
        """
        Initialize tuner

        Args:
            cv_scheme: Cross-validation scheme ('purged', 'walk_forward')
            n_splits: Number of CV splits
        """
        self.cv_scheme = cv_scheme
        self.n_splits = n_splits
        logger.info(f"Initialized HyperparameterTuner with {cv_scheme} CV")

    def get_cv_splitter(self, purge_gap: int = 10):
        """
        Get appropriate CV splitter

        Args:
            purge_gap: Purge gap for time series CV

        Returns:
            CV splitter object
        """
        from models.training.cv_schemes import PurgedKFold, WalkForwardCV

        if self.cv_scheme == 'purged':
            return PurgedKFold(n_splits=self.n_splits, purge_gap=purge_gap)
        elif self.cv_scheme == 'walk_forward':
            return WalkForwardCV(n_splits=self.n_splits, gap=purge_gap)
        else:
            raise ValueError(f"Unknown CV scheme: {self.cv_scheme}")

    def tune_random_forest(self, X: pd.DataFrame, y: pd.Series,
                          n_iter: int = 20,
                          sample_weight: Optional[pd.Series] = None) -> Dict:
        """
        Tune Random Forest hyperparameters

        Args:
            X: Features
            y: Labels
            n_iter: Number of parameter combinations to try
            sample_weight: Optional sample weights

        Returns:
            Dictionary with best parameters and results
        """
        from sklearn.ensemble import RandomForestClassifier

        logger.info(f"Tuning Random Forest with {n_iter} iterations")

        # Parameter grid
        param_distributions = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [20, 50, 100],
            'min_samples_leaf': [10, 25, 50],
            'max_features': ['sqrt', 'log2', 0.5]
        }

        # Base model
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        # Get CV splitter
        cv = self.get_cv_splitter()

        # Random search
        search = RandomizedSearchCV(
            rf,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        # Fit
        if sample_weight is not None:
            fit_params = {'sample_weight': sample_weight}
            search.fit(X, y, **fit_params)
        else:
            search.fit(X, y)

        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")

        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'cv_results': pd.DataFrame(search.cv_results_)
        }

    def tune_lightgbm(self, X: pd.DataFrame, y: pd.Series,
                     n_iter: int = 20,
                     sample_weight: Optional[pd.Series] = None) -> Dict:
        """
        Tune LightGBM hyperparameters

        Args:
            X: Features
            y: Labels
            n_iter: Number of parameter combinations to try
            sample_weight: Optional sample weights

        Returns:
            Dictionary with best parameters and results
        """
        try:
            import lightgbm as lgb
        except ImportError:
            logger.error("LightGBM not installed")
            return {}

        logger.info(f"Tuning LightGBM with {n_iter} iterations")

        # Parameter grid
        param_distributions = {
            'n_estimators': [50, 100, 200],
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.6, 0.8, 1.0],
            'bagging_fraction': [0.6, 0.8, 1.0],
            'min_child_samples': [10, 20, 50]
        }

        # Base model
        lgbm = lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)

        # Get CV splitter
        cv = self.get_cv_splitter()

        # Random search
        search = RandomizedSearchCV(
            lgbm,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        # Fit
        if sample_weight is not None:
            fit_params = {'sample_weight': sample_weight}
            search.fit(X, y, **fit_params)
        else:
            search.fit(X, y)

        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")

        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'cv_results': pd.DataFrame(search.cv_results_)
        }

    def grid_search(self, model: Any, param_grid: Dict,
                   X: pd.DataFrame, y: pd.Series,
                   sample_weight: Optional[pd.Series] = None) -> Dict:
        """
        Grid search for any model

        Args:
            model: Model to tune
            param_grid: Parameter grid
            X: Features
            y: Labels
            sample_weight: Optional sample weights

        Returns:
            Dictionary with best parameters and results
        """
        logger.info(f"Running grid search with {len(param_grid)} parameters")

        # Get CV splitter
        cv = self.get_cv_splitter()

        # Grid search
        search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )

        # Fit
        if sample_weight is not None:
            fit_params = {'sample_weight': sample_weight}
            search.fit(X, y, **fit_params)
        else:
            search.fit(X, y)

        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")

        return {
            'best_params': search.best_params_,
            'best_score': search.best_score_,
            'best_estimator': search.best_estimator_,
            'cv_results': pd.DataFrame(search.cv_results_)
        }


def get_default_param_grids() -> Dict[str, Dict]:
    """
    Get default parameter grids for common models

    Returns:
        Dictionary of parameter grids
    """
    return {
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [20, 50, 100],
            'min_samples_leaf': [10, 25, 50],
            'max_features': ['sqrt', 'log2']
        },
        'lightgbm': {
            'n_estimators': [50, 100, 200],
            'num_leaves': [15, 31, 63],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.6, 0.8, 1.0],
            'min_child_samples': [10, 20, 50]
        },
        'logistic': {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    }
