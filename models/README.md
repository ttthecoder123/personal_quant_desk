# Signal Generation and Labeling System

Comprehensive implementation of López de Prado's triple-barrier method and meta-labeling framework from "Advances in Financial Machine Learning".

## Overview

This module implements a complete signal generation and labeling pipeline:

1. **Event Detection**: CUSUM filter and structural break detection
2. **Triple-Barrier Labeling**: Time-aware labeling with profit-taking, stop-loss, and time barriers
3. **Sample Weighting**: Uniqueness-based weights accounting for label overlap
4. **Primary Models**: ML-based signal generation (Random Forest, LightGBM)
5. **Meta-Labeling**: Secondary model for bet sizing
6. **Feature Importance**: MDI, MDA, and SHAP analysis

## Directory Structure

```
models/
├── labeling/              # Labeling framework
│   ├── triple_barrier.py  # Triple-barrier implementation
│   ├── meta_labeling.py   # Meta-labeling for bet sizing
│   ├── sample_weights.py  # Sample weighting schemes
│   └── event_sampling.py  # CUSUM and event detection
├── signals/               # Signal generation
│   ├── base_signals.py    # Rule-based signals
│   └── ml_signals.py      # ML-based signals
├── training/              # Training utilities
│   ├── cv_schemes.py      # Purged K-fold, walk-forward CV
│   ├── feature_importance.py  # MDI, MDA, SHAP
│   └── hyperparameter_tuning.py
├── config/
│   └── model_config.yaml  # Configuration
├── model_pipeline.py      # Main orchestration
└── main.py               # CLI interface
```

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `scikit-learn>=1.3.0`
- `lightgbm>=4.1.0`
- `shap>=0.42.0`
- `numba>=0.57.0`

## Usage

### Command Line Interface

Process a single symbol:
```bash
python models/main.py process --symbol SPY --action all --use-meta
```

Batch processing:
```bash
python models/main.py batch --symbols SPY,QQQ,IWM --use-meta
```

Analyze feature importance:
```bash
python models/main.py analyze --symbol SPY
```

### Python API

```python
from models.model_pipeline import ModelPipeline

# Initialize pipeline
pipeline = ModelPipeline('models/config/model_config.yaml')

# Run triple-barrier labeling
labels = pipeline.run_triple_barrier_labeling('SPY')

# Train primary model
model = pipeline.train_primary_model('SPY')

# Train meta-model (optional)
meta_model = pipeline.train_meta_model('SPY')

# Generate signals
signals = pipeline.generate_signals('SPY', use_meta=True)

# Backtest
metrics = pipeline.backtest_signals('SPY')
print(f"Hit rate: {metrics['hit_rate']:.2%}")
print(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
```

## Configuration

Edit `models/config/model_config.yaml`:

```yaml
labeling:
  cusum_threshold: 0.02        # Event detection sensitivity
  volatility_multiplier: 2.0   # Barrier width
  max_holding_days: 10         # Maximum holding period
  barrier_ratios: [1.5, 1.0]   # [profit, stop-loss]

training:
  test_size: 0.2
  cv_splits: 5
  purge_gap: 10

meta_labeling:
  use_meta: true
  probability_calibration: 'isotonic'
```

## Key Features

### Triple-Barrier Labeling

Implements López de Prado's method for generating labels:
- **Upper barrier**: Profit-taking threshold
- **Lower barrier**: Stop-loss threshold
- **Vertical barrier**: Maximum holding period

Labels are generated based on which barrier is touched first, avoiding look-ahead bias.

### Meta-Labeling

Secondary model that learns:
- When to act on primary signals
- How to size positions
- Probability calibration for bet sizing

### Sample Weighting

Addresses overlapping labels in time series:
- **Uniqueness**: Adjusts for label overlap
- **Return attribution**: Weights by magnitude
- **Time decay**: Optional recency bias

### Cross-Validation

Purged K-fold and walk-forward schemes that prevent data leakage:
- Removes overlapping observations from train/test sets
- Maintains temporal order
- Realistic performance estimates

### Feature Importance

Three complementary methods:
- **MDI**: Mean Decrease Impurity (fast, built-in)
- **MDA**: Mean Decrease Accuracy (permutation-based)
- **SHAP**: Game-theoretic feature attribution

## Output

Models and signals are saved to:
- `models/saved/{symbol}_primary.joblib` - Primary models
- `models/saved/{symbol}_meta.joblib` - Meta-models
- `models/signals/{symbol}_signals.parquet` - Generated signals

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
  - Chapter 2: Financial Data Structures
  - Chapter 3: Labeling
  - Chapter 4: Sample Weights
  - Chapter 7: Cross-Validation
  - Chapter 8: Feature Importance

- Chan, E. (2013). *Algorithmic Trading: Winning Strategies*. Wiley.
- Jansen, S. (2020). *Machine Learning for Algorithmic Trading*. Packt.

## Testing

To test the system with sample data:

```bash
# Ensure data is ingested first
cd data
python -m data.main ingest --symbol SPY --start-date 2020-01-01

# Generate features
python -m data.features.feature_engine --symbol SPY

# Run signal generation
cd ..
python models/main.py process --symbol SPY --action all
```

## Performance

Expected metrics on SPY (2020-2023):
- Hit rate: 55-60%
- Sharpe ratio: 1.0-1.5
- Max drawdown: 10-15%

Results vary based on:
- Market regime
- Feature engineering quality
- Hyperparameter tuning
- Meta-labeling effectiveness

## Troubleshooting

**Import errors**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

**Missing data**: Run data ingestion first
```bash
python -m data.main ingest --symbol SPY
```

**Memory errors**: Reduce sample size or use chunking
- Adjust `shap_sample_size` in config
- Use smaller training periods

## Future Enhancements

- [ ] Additional event detection methods (entropy, structural breaks)
- [ ] Online learning for dynamic model updates
- [ ] Portfolio-level meta-labeling
- [ ] GPU acceleration for large-scale training
- [ ] Real-time signal generation
- [ ] Integration with execution system

## License

Part of Personal Quant Desk - see main repository for license.
