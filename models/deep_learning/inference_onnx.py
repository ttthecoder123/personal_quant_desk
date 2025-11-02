#!/usr/bin/env python3
"""
ONNX Inference Engine for M1 MacBook
Optimized for CPU inference with low latency

Features:
- ONNX Runtime with CPU optimizations
- Batch inference support
- Model caching for fast repeated inference
- Performance monitoring
"""

import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Optional
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ONNXInferenceEngine:
    """Fast ONNX inference engine for M1 CPU"""

    def __init__(self, model_dir: str = "models/trained"):
        """
        Initialize inference engine

        Args:
            model_dir: Directory containing ONNX models
        """
        self.model_dir = Path(model_dir)
        self.sessions = {}  # Cache loaded models

        # Configure ONNX Runtime for M1 CPU optimization
        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session_options.intra_op_num_threads = 8  # M1 Pro has 8 performance cores
        self.session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        logger.info("ONNX Inference Engine initialized")

    def load_model(self, model_name: str) -> ort.InferenceSession:
        """
        Load ONNX model (with caching)

        Args:
            model_name: Name of the model (e.g., 'lstm', 'transformer')

        Returns:
            ONNX inference session
        """
        if model_name in self.sessions:
            return self.sessions[model_name]

        # Find model file
        model_path = self.model_dir / model_name / f"{model_name}_predictor.onnx"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Create inference session
        session = ort.InferenceSession(
            str(model_path),
            sess_options=self.session_options,
            providers=['CPUExecutionProvider']  # M1 uses CPU provider
        )

        # Cache session
        self.sessions[model_name] = session

        logger.info(f"✓ Loaded model: {model_name}")
        logger.info(f"  Input: {session.get_inputs()[0].name}, shape: {session.get_inputs()[0].shape}")
        logger.info(f"  Output: {session.get_outputs()[0].name}, shape: {session.get_outputs()[0].shape}")

        return session

    def predict(self, model_name: str, features: np.ndarray) -> np.ndarray:
        """
        Run inference

        Args:
            model_name: Name of the model
            features: Input features, shape (batch, seq_length, num_features)

        Returns:
            Predictions, shape (batch, num_horizons)
        """
        session = self.load_model(model_name)

        # Prepare input
        input_name = session.get_inputs()[0].name
        ort_inputs = {input_name: features.astype(np.float32)}

        # Run inference
        start_time = time.time()
        ort_outputs = session.run(None, ort_inputs)
        inference_time = (time.time() - start_time) * 1000  # ms

        predictions = ort_outputs[0]

        logger.debug(f"Inference time: {inference_time:.2f}ms for batch size {len(features)}")

        return predictions

    def predict_batch(self, model_name: str, features_list: List[np.ndarray]) -> np.ndarray:
        """
        Batch inference for multiple samples

        Args:
            model_name: Name of the model
            features_list: List of feature arrays

        Returns:
            Predictions for all samples
        """
        # Stack into batch
        features_batch = np.stack(features_list, axis=0)

        # Run inference
        predictions = self.predict(model_name, features_batch)

        return predictions

    def predict_single(self, model_name: str, features: np.ndarray) -> np.ndarray:
        """
        Single sample inference (convenience method)

        Args:
            model_name: Name of the model
            features: Feature array, shape (seq_length, num_features)

        Returns:
            Prediction, shape (num_horizons,)
        """
        # Add batch dimension
        features_batch = features[np.newaxis, ...]

        # Run inference
        predictions = self.predict(model_name, features_batch)

        # Remove batch dimension
        return predictions[0]

    def benchmark(self, model_name: str, num_runs: int = 100):
        """
        Benchmark model inference speed

        Args:
            model_name: Name of the model
            num_runs: Number of inference runs
        """
        session = self.load_model(model_name)

        # Get input shape
        input_shape = session.get_inputs()[0].shape
        # Replace dynamic dimensions with concrete values
        input_shape = [1 if isinstance(dim, str) else dim for dim in input_shape]

        # Generate dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warm-up
        for _ in range(10):
            self.predict(model_name, dummy_input)

        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            self.predict(model_name, dummy_input)
        total_time = time.time() - start_time

        avg_time = (total_time / num_runs) * 1000  # ms
        throughput = num_runs / total_time  # samples/sec

        logger.info(f"\n=== Benchmark Results: {model_name} ===")
        logger.info(f"Average inference time: {avg_time:.2f}ms")
        logger.info(f"Throughput: {throughput:.1f} samples/sec")
        logger.info(f"Total runs: {num_runs}")

        return avg_time, throughput


class ModelEnsemble:
    """Ensemble multiple models for improved predictions"""

    def __init__(self, inference_engine: ONNXInferenceEngine):
        self.engine = inference_engine
        self.model_weights = {}

    def add_model(self, model_name: str, weight: float = 1.0):
        """Add model to ensemble with weight"""
        self.model_weights[model_name] = weight

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Ensemble prediction

        Args:
            features: Input features for all models

        Returns:
            Weighted average predictions
        """
        predictions = []
        total_weight = sum(self.model_weights.values())

        for model_name, weight in self.model_weights.items():
            pred = self.engine.predict_single(model_name, features)
            predictions.append(pred * (weight / total_weight))

        ensemble_pred = np.sum(predictions, axis=0)

        return ensemble_pred


# Integration with existing signal generator
class DeepLearningSignalGenerator:
    """
    Integrates deep learning models into existing signal generation pipeline
    Compatible with your existing models/signal_generator.py
    """

    def __init__(self, model_names: List[str] = ['lstm']):
        self.inference_engine = ONNXInferenceEngine()
        self.model_names = model_names

    def generate_signals(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals from deep learning models

        Args:
            features_df: DataFrame with features (from your feature engineering)

        Returns:
            DataFrame with signals for each model and horizon
        """
        # Prepare sequence data (last 60 days)
        seq_length = 60
        feature_cols = [col for col in features_df.columns if col not in ['date', 'symbol']]

        signals = []

        for model_name in self.model_names:
            # Get last seq_length rows
            if len(features_df) < seq_length:
                logger.warning(f"Not enough data for sequence (need {seq_length}, have {len(features_df)})")
                continue

            features = features_df[feature_cols].iloc[-seq_length:].values

            # Run inference
            predictions = self.inference_engine.predict_single(model_name, features)

            # Convert predictions to signals (-1, 0, 1)
            for i, horizon in enumerate([1, 5, 20]):  # Match training horizons
                signal = np.sign(predictions[i])  # -1 (short), 0 (neutral), 1 (long)
                confidence = abs(predictions[i])  # Use magnitude as confidence

                signals.append({
                    'model': model_name,
                    'horizon': f'{horizon}d',
                    'signal': signal,
                    'confidence': confidence,
                    'raw_prediction': predictions[i]
                })

        return pd.DataFrame(signals)


def main():
    """Example usage"""

    # Initialize engine
    engine = ONNXInferenceEngine()

    # Benchmark LSTM model
    if Path("models/trained/lstm/lstm_predictor.onnx").exists():
        engine.benchmark('lstm', num_runs=100)

    # Example: Generate signals
    # Load your features
    features_df = pd.read_parquet("data/features/computed/features_latest.parquet")

    # Generate signals
    signal_gen = DeepLearningSignalGenerator(model_names=['lstm'])
    signals = signal_gen.generate_signals(features_df)

    print("\n=== Generated Signals ===")
    print(signals)


if __name__ == '__main__':
    main()
