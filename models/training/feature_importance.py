"""
Feature importance methods
Based on López de Prado's "Advances in Financial Machine Learning", Chapter 8
"""
import pandas as pd
import numpy as np
from typing import Optional, Union
from sklearn.metrics import accuracy_score, log_loss
from loguru import logger


class FeatureImportance:
    """
    Feature importance methods (López de Prado Ch. 8)

    Three complementary approaches:
    1. MDI (Mean Decrease Impurity) - fast, built-in
    2. MDA (Mean Decrease Accuracy) - more robust
    3. SHAP - game-theoretic approach
    """

    @staticmethod
    def mean_decrease_impurity(model, feature_names: Optional[list] = None) -> pd.Series:
        """
        Mean Decrease Impurity (MDI) feature importance

        Fast method using the model's built-in feature importances.
        Works for tree-based models (Random Forest, Gradient Boosting).

        Args:
            model: Trained tree-based model
            feature_names: Optional list of feature names

        Returns:
            Series of feature importances
        """
        logger.info("Calculating MDI feature importance")

        try:
            importances = model.feature_importances_
        except AttributeError:
            logger.error("Model does not have feature_importances_ attribute")
            return pd.Series()

        if feature_names is None:
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            else:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

        importance_df = pd.Series(importances, index=feature_names)
        importance_df = importance_df.sort_values(ascending=False)

        logger.info(f"Top 5 MDI features: {importance_df.head().to_dict()}")

        return importance_df

    @staticmethod
    def mean_decrease_accuracy(model, X_test: pd.DataFrame, y_test: pd.Series,
                              n_iterations: int = 10,
                              scoring: str = 'accuracy') -> pd.Series:
        """
        Mean Decrease Accuracy (MDA) feature importance

        Measures importance by permuting each feature and measuring
        the decrease in model performance. More robust than MDI.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            n_iterations: Number of permutation iterations
            scoring: Scoring metric ('accuracy' or 'log_loss')

        Returns:
            Series of feature importances
        """
        logger.info(f"Calculating MDA feature importance with {n_iterations} iterations")

        # Calculate baseline score
        if scoring == 'accuracy':
            baseline_score = accuracy_score(y_test, model.predict(X_test))
        elif scoring == 'log_loss':
            baseline_score = -log_loss(y_test, model.predict_proba(X_test))
        else:
            raise ValueError(f"Unknown scoring method: {scoring}")

        logger.debug(f"Baseline {scoring}: {baseline_score:.4f}")

        importances = {}

        for col in X_test.columns:
            scores = []

            for iteration in range(n_iterations):
                X_permuted = X_test.copy()
                # Permute single feature
                X_permuted[col] = np.random.permutation(X_permuted[col].values)

                # Calculate accuracy with permuted feature
                if scoring == 'accuracy':
                    permuted_score = accuracy_score(y_test, model.predict(X_permuted))
                elif scoring == 'log_loss':
                    permuted_score = -log_loss(y_test, model.predict_proba(X_permuted))

                # Importance is the decrease in performance
                scores.append(baseline_score - permuted_score)

            importances[col] = np.mean(scores)

        importance_df = pd.Series(importances)
        importance_df = importance_df.sort_values(ascending=False)

        logger.info(f"Top 5 MDA features: {importance_df.head().to_dict()}")

        return importance_df

    @staticmethod
    def shap_importance(model, X_sample: pd.DataFrame,
                       check_additivity: bool = False) -> pd.Series:
        """
        SHAP (SHapley Additive exPlanations) values

        Game-theoretic approach to feature importance that provides
        consistent and locally accurate attributions.

        Args:
            model: Trained model
            X_sample: Sample of features (use subset for speed)
            check_additivity: Whether to check additivity constraint

        Returns:
            Series of feature importances
        """
        try:
            import shap
        except ImportError:
            logger.error("SHAP not installed. Install with: pip install shap")
            return pd.Series()

        logger.info(f"Calculating SHAP importance on {len(X_sample)} samples")

        try:
            # Create explainer (auto-detects model type)
            explainer = shap.TreeExplainer(model, check_additivity=check_additivity)

            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)

            # Handle multi-class output
            if isinstance(shap_values, list):
                # Take absolute mean across classes and samples
                shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                # Take absolute mean across samples
                shap_importance = np.abs(shap_values).mean(axis=0)

            importance_df = pd.Series(shap_importance, index=X_sample.columns)
            importance_df = importance_df.sort_values(ascending=False)

            logger.info(f"Top 5 SHAP features: {importance_df.head().to_dict()}")

            return importance_df

        except Exception as e:
            logger.error(f"SHAP calculation failed: {e}")
            return pd.Series()

    @staticmethod
    def clustered_feature_importance(importances: pd.Series,
                                    correlation_matrix: pd.DataFrame,
                                    linkage_method: str = 'complete') -> pd.Series:
        """
        Cluster features and average their importance

        Addresses multicollinearity by grouping correlated features
        and using their combined importance.

        Args:
            importances: Feature importance series
            correlation_matrix: Feature correlation matrix
            linkage_method: Hierarchical clustering linkage method

        Returns:
            Clustered feature importance
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        logger.info("Calculating clustered feature importance")

        # Convert correlation to distance
        distance = ((1 - correlation_matrix.fillna(0)) / 2.0) ** 0.5

        # Hierarchical clustering
        link = linkage(squareform(distance.values), method=linkage_method)

        # Form clusters
        clusters = fcluster(link, t=0.5, criterion='distance')

        # Average importance within clusters
        clustered_importance = pd.Series(index=importances.index, dtype=float)

        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_features = importances.index[cluster_mask]

            # Average importance for this cluster
            avg_importance = importances[cluster_features].mean()

            # Assign to all features in cluster
            clustered_importance[cluster_features] = avg_importance

        clustered_importance = clustered_importance.sort_values(ascending=False)

        logger.info(f"Created {len(np.unique(clusters))} feature clusters")

        return clustered_importance

    @staticmethod
    def combine_importances(mdi: pd.Series, mda: pd.Series,
                          shap_imp: Optional[pd.Series] = None,
                          weights: Optional[dict] = None) -> pd.Series:
        """
        Combine multiple feature importance measures

        Args:
            mdi: MDI importance
            mda: MDA importance
            shap_imp: SHAP importance (optional)
            weights: Dictionary of weights for each method

        Returns:
            Combined importance series
        """
        logger.info("Combining feature importances")

        if weights is None:
            if shap_imp is not None:
                weights = {'mdi': 0.3, 'mda': 0.4, 'shap': 0.3}
            else:
                weights = {'mdi': 0.4, 'mda': 0.6}

        # Normalize each importance measure
        mdi_norm = mdi / mdi.sum()
        mda_norm = mda / mda.sum()

        # Align indices
        common_idx = mdi_norm.index.intersection(mda_norm.index)

        combined = (
            weights['mdi'] * mdi_norm.loc[common_idx] +
            weights['mda'] * mda_norm.loc[common_idx]
        )

        if shap_imp is not None and 'shap' in weights:
            shap_norm = shap_imp / shap_imp.sum()
            common_idx = combined.index.intersection(shap_norm.index)
            combined = (
                (weights['mdi'] + weights['mda']) * combined.loc[common_idx] +
                weights['shap'] * shap_norm.loc[common_idx]
            )
            combined = combined / combined.sum()

        combined = combined.sort_values(ascending=False)

        logger.info(f"Top 5 combined features: {combined.head().to_dict()}")

        return combined

    @staticmethod
    def select_top_features(importances: pd.Series,
                          threshold: Optional[float] = None,
                          top_n: Optional[int] = None) -> list:
        """
        Select top features based on importance

        Args:
            importances: Feature importance series
            threshold: Minimum importance threshold
            top_n: Number of top features to select

        Returns:
            List of selected feature names
        """
        if top_n is not None:
            selected = importances.head(top_n).index.tolist()
            logger.info(f"Selected top {top_n} features")
        elif threshold is not None:
            selected = importances[importances >= threshold].index.tolist()
            logger.info(f"Selected {len(selected)} features with importance >= {threshold}")
        else:
            # Default: select features with cumulative importance >= 80%
            cumsum = importances.cumsum() / importances.sum()
            selected = cumsum[cumsum <= 0.8].index.tolist()
            logger.info(f"Selected {len(selected)} features (80% cumulative importance)")

        return selected


def calculate_all_importances(model, X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame, y_test: pd.Series,
                              use_shap: bool = True,
                              shap_sample_size: int = 100) -> dict:
    """
    Convenience function to calculate all importance measures

    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        use_shap: Whether to calculate SHAP importance
        shap_sample_size: Sample size for SHAP calculation

    Returns:
        Dictionary with all importance measures
    """
    logger.info("Calculating all feature importances")

    fi = FeatureImportance()

    results = {}

    # MDI
    results['mdi'] = fi.mean_decrease_impurity(model, feature_names=X_train.columns)

    # MDA
    results['mda'] = fi.mean_decrease_accuracy(model, X_test, y_test)

    # SHAP
    if use_shap:
        # Use sample for speed
        X_sample = X_train.sample(n=min(shap_sample_size, len(X_train)), random_state=42)
        results['shap'] = fi.shap_importance(model, X_sample)

    # Combined
    results['combined'] = fi.combine_importances(
        results['mdi'],
        results['mda'],
        results.get('shap')
    )

    return results
