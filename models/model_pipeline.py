"""
Main orchestration pipeline for signal generation and labeling
Integrates López de Prado's framework end-to-end
"""
import pandas as pd
import numpy as np
from loguru import logger
from pathlib import Path
import joblib
from typing import Dict, Tuple, Optional
import yaml

from models.labeling.triple_barrier import TripleBarrierLabeling, apply_triple_barrier_labeling
from models.labeling.meta_labeling import MetaLabeling
from models.labeling.sample_weights import SampleWeights
from models.labeling.event_sampling import EventSampling
from models.signals.base_signals import BaseSignals
from models.signals.ml_signals import MLSignals
from models.training.feature_importance import FeatureImportance, calculate_all_importances


class ModelPipeline:
    """
    Main orchestration for signal generation and labeling

    Implements the complete workflow:
    1. Event detection
    2. Triple-barrier labeling
    3. Sample weighting
    4. Primary model training
    5. Meta-labeling
    6. Signal generation
    """

    def __init__(self, config_path: str = 'models/config/model_config.yaml'):
        """
        Initialize pipeline

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.labels = {}
        self.signals = {}
        self.features_cache = {}
        logger.info("Initialized ModelPipeline")

    def _load_config(self, path: str) -> dict:
        """Load configuration from YAML file"""
        config_path = Path(path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return self._get_default_config()

        with open(config_path) as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {path}")
        return config

    def _get_default_config(self) -> dict:
        """Get default configuration"""
        return {
            'labeling': {
                'cusum_threshold': 0.02,
                'volatility_multiplier': 2.0,
                'max_holding_days': 10,
                'barrier_ratios': [1.5, 1.0]
            },
            'training': {
                'test_size': 0.2,
                'cv_splits': 5,
                'purge_gap': 10
            },
            'meta_labeling': {
                'use_meta': True,
                'probability_calibration': 'isotonic'
            }
        }

    def run_triple_barrier_labeling(self, symbol: str,
                                   prices: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Run triple-barrier labeling for a symbol

        Args:
            symbol: Symbol to label
            prices: Optional price series (will load if not provided)

        Returns:
            DataFrame with labels
        """
        logger.info(f'Running triple-barrier labeling for {symbol}')

        # Load price data if not provided
        if prices is None:
            prices = self._load_prices(symbol)

        # Detect events using CUSUM filter
        event_sampler = EventSampling(prices)
        events_idx = event_sampler.cusum_filter(
            threshold=self.config['labeling']['cusum_threshold']
        )

        logger.info(f"Detected {len(events_idx)} events")

        if len(events_idx) == 0:
            logger.warning("No events detected, returning empty labels")
            return pd.DataFrame()

        # Calculate volatility for barriers
        volatility = TripleBarrierLabeling.get_daily_volatility(prices)

        # Apply triple-barrier labeling
        labels = apply_triple_barrier_labeling(
            prices=prices,
            events_idx=events_idx,
            volatility=volatility,
            pt_sl=self.config['labeling']['barrier_ratios'],
            num_days=self.config['labeling']['max_holding_days'],
            volatility_multiplier=self.config['labeling']['volatility_multiplier']
        )

        # Calculate sample weights
        logger.info("Calculating sample weights")
        sample_weights = SampleWeights()
        labels['uniqueness'] = sample_weights.get_sample_uniqueness(labels)
        labels['return_weight'] = sample_weights.get_return_attribution(labels)
        labels['time_decay'] = sample_weights.get_time_decay(labels)

        # Combined weight
        labels['sample_weight'] = labels[['uniqueness', 'return_weight', 'time_decay']].mean(axis=1)

        self.labels[symbol] = labels
        logger.info(f'Generated {len(labels)} labels for {symbol}')
        logger.info(f"Label statistics:\n{labels['label'].describe()}")

        return labels

    def train_primary_model(self, symbol: str,
                          features: Optional[pd.DataFrame] = None,
                          labels: Optional[pd.DataFrame] = None) -> object:
        """
        Train primary signal generation model

        Args:
            symbol: Symbol to train on
            features: Optional features (will load if not provided)
            labels: Optional labels (will generate if not provided)

        Returns:
            Trained model
        """
        logger.info(f'Training primary model for {symbol}')

        # Load features if not provided
        if features is None:
            features = self._load_features(symbol)

        # Get labels if not provided
        if labels is None:
            if symbol not in self.labels:
                self.run_triple_barrier_labeling(symbol)
            labels = self.labels[symbol]

        # Train ML model
        ml_signals = MLSignals(features, labels)
        X_train, X_test, y_train, y_test, w_train, w_test = ml_signals.prepare_data(
            test_size=self.config['training']['test_size'],
            sample_weights=labels['sample_weight']
        )

        logger.info(f"Training ensemble model")
        model = ml_signals.train_ensemble(X_train, y_train, sample_weight=w_train)

        # Evaluate
        metrics = ml_signals.evaluate_model('ensemble', X_test, y_test)
        logger.info(f"Model evaluation: {metrics}")

        # Calculate feature importance
        logger.info("Calculating feature importances")
        importances = calculate_all_importances(
            model, X_train, y_train, X_test, y_test,
            use_shap=True, shap_sample_size=100
        )

        # Store model and metadata
        self.models[f'{symbol}_primary'] = {
            'model': model,
            'metrics': metrics,
            'importances': importances,
            'ml_signals': ml_signals
        }

        # Save model
        self._save_model(symbol, 'primary', model)

        logger.info(f"Top 10 features: {importances['combined'].head(10).to_dict()}")

        return model

    def train_meta_model(self, symbol: str,
                        features: Optional[pd.DataFrame] = None,
                        prices: Optional[pd.Series] = None) -> object:
        """
        Train meta-labeling model

        Args:
            symbol: Symbol to train on
            features: Optional features
            prices: Optional prices

        Returns:
            Trained meta-model
        """
        logger.info(f'Training meta-model for {symbol}')

        # Load data
        if features is None:
            features = self._load_features(symbol)
        if prices is None:
            prices = self._load_prices(symbol)

        # Get labels
        if symbol not in self.labels:
            self.run_triple_barrier_labeling(symbol, prices)
        labels = self.labels[symbol]

        # Generate primary signals using base signals
        logger.info("Generating primary base signals")
        base_signal_gen = BaseSignals(features, prices)
        primary_signals = base_signal_gen.composite_signal()

        # Generate meta-labels
        meta_labeler = MetaLabeling(primary_signals, prices)
        meta_labels_df = meta_labeler.generate_meta_labels(labels)

        # Train meta-model
        meta_model = meta_labeler.train_meta_model(
            features,
            meta_labels_df['meta_label'],
            calibration_method=self.config['meta_labeling']['probability_calibration']
        )

        # Store
        self.models[f'{symbol}_meta'] = {
            'model': meta_model,
            'meta_labeler': meta_labeler,
            'primary_signals': primary_signals
        }

        # Save model
        self._save_model(symbol, 'meta', meta_model)

        return meta_model

    def generate_signals(self, symbol: str,
                        use_meta: bool = True,
                        features: Optional[pd.DataFrame] = None,
                        prices: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate final trading signals

        Args:
            symbol: Symbol to generate signals for
            use_meta: Whether to use meta-labeling
            features: Optional features
            prices: Optional prices

        Returns:
            DataFrame with signals
        """
        logger.info(f'Generating signals for {symbol} (use_meta={use_meta})')

        # Load data
        if features is None:
            features = self._load_features(symbol)
        if prices is None:
            prices = self._load_prices(symbol)

        # Generate primary signals
        if f'{symbol}_primary' not in self.models:
            logger.info("Primary model not trained, training now")
            self.train_primary_model(symbol, features)

        primary_model_data = self.models[f'{symbol}_primary']
        ml_signal_gen = primary_model_data['ml_signals']
        primary_signals = ml_signal_gen.generate_signals('ensemble')

        # Apply meta-labeling if requested
        if use_meta and self.config['meta_labeling']['use_meta']:
            if f'{symbol}_meta' not in self.models:
                logger.info("Meta-model not trained, training now")
                self.train_meta_model(symbol, features, prices)

            meta_model_data = self.models[f'{symbol}_meta']
            meta_labeler = meta_model_data['meta_labeler']

            # Get bet sizing from meta-model
            bet_size = meta_labeler.predict_bet_size(meta_model_data['model'], features)

            # Combine primary signal with bet size
            common_idx = primary_signals.index.intersection(bet_size.index)
            final_signals = primary_signals.loc[common_idx] * bet_size.loc[common_idx]
        else:
            final_signals = primary_signals
            bet_size = pd.Series(1.0, index=primary_signals.index)

        # Create signal DataFrame
        signal_df = pd.DataFrame({
            'signal': final_signals,
            'signal_strength': np.abs(final_signals),
            'primary_signal': primary_signals,
            'bet_size': bet_size
        })

        self.signals[symbol] = signal_df

        logger.info(f"Generated {len(signal_df)} signals")
        logger.info(f"Signal distribution:\n{signal_df['signal'].describe()}")

        return signal_df

    def backtest_signals(self, symbol: str,
                        signals: Optional[pd.DataFrame] = None,
                        prices: Optional[pd.Series] = None) -> Dict:
        """
        Basic signal performance metrics

        Args:
            symbol: Symbol to backtest
            signals: Optional signals (will generate if not provided)
            prices: Optional prices

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Backtesting signals for {symbol}")

        # Load data
        if signals is None:
            if symbol not in self.signals:
                self.generate_signals(symbol)
            signals = self.signals[symbol]

        if prices is None:
            prices = self._load_prices(symbol)

        # Calculate returns
        returns = prices.pct_change()

        # Align signals and returns
        common_idx = signals.index.intersection(returns.index)
        sig = signals.loc[common_idx, 'signal']
        ret = returns.loc[common_idx]

        # Calculate signal returns
        signal_returns = sig.shift(1) * ret  # Shift to avoid look-ahead bias

        # Performance metrics
        metrics = {
            'total_signals': (sig != 0).sum(),
            'long_signals': (sig > 0).sum(),
            'short_signals': (sig < 0).sum(),
            'hit_rate': (signal_returns[sig.shift(1) != 0] > 0).mean(),
            'avg_return': signal_returns[sig.shift(1) != 0].mean(),
            'total_return': signal_returns.sum(),
            'sharpe_ratio': signal_returns.mean() / signal_returns.std() * np.sqrt(252) if signal_returns.std() > 0 else 0,
            'max_drawdown': (signal_returns.cumsum().cummax() - signal_returns.cumsum()).max()
        }

        logger.info(f"Backtest metrics: {metrics}")

        return metrics

    def _load_prices(self, symbol: str) -> pd.Series:
        """Load price data for symbol"""
        try:
            from data.storage.parquet_storage import ParquetStorage
            storage = ParquetStorage()
            df = storage.load_timeseries(symbol)
            return df['Close']
        except Exception as e:
            logger.error(f"Failed to load prices for {symbol}: {e}")
            raise

    def _load_features(self, symbol: str) -> pd.DataFrame:
        """Load features for symbol"""
        if symbol in self.features_cache:
            return self.features_cache[symbol]

        try:
            from data.features.feature_store import FeatureStore
            feature_store = FeatureStore()
            features = feature_store.load_features(symbol)
            self.features_cache[symbol] = features
            logger.info(f"Loaded {len(features)} features for {symbol}")
            return features
        except Exception as e:
            logger.error(f"Failed to load features for {symbol}: {e}")
            raise

    def _save_model(self, symbol: str, model_type: str, model: object):
        """Save trained model"""
        model_path = Path(f'models/saved/{symbol}_{model_type}.joblib')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")

    def load_model(self, symbol: str, model_type: str) -> object:
        """Load trained model"""
        model_path = Path(f'models/saved/{symbol}_{model_type}.joblib')
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model

    def save_signals(self, symbol: str):
        """Save generated signals"""
        if symbol not in self.signals:
            raise ValueError(f"No signals generated for {symbol}")

        signal_path = Path(f'models/signals/{symbol}_signals.parquet')
        signal_path.parent.mkdir(parents=True, exist_ok=True)

        self.signals[symbol].to_parquet(signal_path)
        logger.info(f"Saved signals to {signal_path}")
