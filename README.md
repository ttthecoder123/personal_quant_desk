# Personal Quant Desk - AI-Assisted Trading System

A comprehensive, semi-automated trading system for commodities, indices, and FX pairs with AI-driven signal generation and enhanced data ingestion capabilities.

## 🚀 Project Status

- ✅ **Step 1: Project Structure** - COMPLETE
- ✅ **Step 2: Data Ingestion** - COMPLETE (with Alpha Vantage enhancement)
- ✅ **Step 3: Feature Engineering** - COMPLETE
- ⏳ **Step 4: Signal Generation** - NEXT
- ⏳ **Step 5: Strategy Development** - Pending
- ⏳ **Step 7: Risk Management** - Pending
- ⏳ **Step 10: Backtesting** - Pending
- ⏳ **Step 12: Execution** - Pending
- ⏳ **Step 13: Monitoring** - Pending

## 📊 Data Sources & Features

### Primary Data Sources
- **Alpha Vantage**: Recent 100 days (high quality, rate limited: 5 calls/min, 25/day)
- **Yahoo Finance**: Historical data (reliable, unlimited)
- **Hybrid Strategy**: Automatic source selection based on date range and asset class

### Enhanced Features
- 🔄 **Intelligent Source Selection**: Alpha Vantage for recent data quality, yfinance for historical
- 📈 **Corporate Action Detection**: Automatic detection of splits and dividends
- 🎯 **Quality Scoring**: Composite quality metrics (0-100) with production thresholds
- 💾 **Smart Caching**: SQLite-based API response caching to minimize rate limits
- 🔍 **Symbol Mapping**: Support for commodities (CL=F→WTI) and FX pairs (AUDUSD=X→AUD/USD)
- ⚡ **Rate Limiting**: Compliant with free API tiers

### Supported Instruments
- **Commodities**: WTI Oil (CL=F), Gold (GC=F), Copper (HG=F)
- **Indices**: SPY, QQQ, ASX (^AXJO)
- **FX Pairs**: AUDUSD, USDJPY, EURUSD

## 📁 Project Structure

```
personal_quant_desk/              # Single consolidated root
├── config/
│   ├── config.yaml              # System configuration
│   └── data_sources.yaml        # Alpha Vantage & data source config
├── data/
│   ├── ingestion/               # Enhanced data ingestion system
│   │   ├── alpha_vantage.py    # Alpha Vantage adapter with caching
│   │   ├── downloader.py       # HybridDataManager for intelligent sourcing
│   │   ├── validator.py        # Corporate action detection & validation
│   │   ├── quality_scorer.py   # Composite quality scoring
│   │   ├── storage.py          # Parquet storage with compression
│   │   ├── catalog.py          # Data catalog with quality tracking
│   │   └── __init__.py
│   ├── features/                # Feature engineering (Step 3)
│   │   ├── base_features.py    # Price/volume transformations
│   │   ├── technical_features.py  # Technical indicators (RSI, MACD, etc.)
│   │   ├── microstructure.py   # Market microstructure features
│   │   ├── regime_features.py  # Regime detection
│   │   ├── cross_asset.py      # Cross-asset correlations
│   │   ├── commodity_specific.py  # Commodity features
│   │   ├── feature_pipeline.py # Feature generation orchestration
│   │   ├── feature_store.py    # Feature storage & versioning
│   │   ├── config/
│   │   │   └── feature_config.yaml  # Feature configuration
│   │   ├── computed/           # Generated features (Parquet)
│   │   └── reports/            # Feature quality reports
│   ├── processed/               # Parquet files
│   ├── cache/                   # API response cache (SQLite)
│   ├── catalog/                 # Metadata storage
│   ├── quality_reports/         # Quality assessment reports
│   ├── config/
│   │   └── instruments.yaml    # Instrument definitions
│   ├── logs/                   # Data pipeline logs
│   └── main.py                 # CLI with hybrid mode options
├── models/                      # ML models (Step 4)
├── strategies/                  # Trading strategies (Step 5)
├── risk/                        # Risk management (Step 7)
├── execution/                   # Order execution (Step 12)
├── backtesting/                 # Strategy backtesting (Step 10)
├── monitoring/                  # System monitoring (Step 13)
├── notebooks/                   # Jupyter analysis notebooks
├── tests/                       # Comprehensive test suite
├── utils/                       # Shared utilities
├── logs/                        # System-wide logs
├── requirements.txt             # All dependencies
├── .env.template               # Environment variables template
├── .gitignore                  # Git ignore patterns
└── README.md                   # This file
```

## 🔧 Quick Start

### 1. Environment Setup
```bash
# Quick navigation (works from any directory!)
quant desk

# Or navigate manually
cd personal_quant_desk

# Copy environment template (includes working Alpha Vantage API key)
cp .env.template .env

# The .env file now contains a working Alpha Vantage API key
# You can use it as-is or replace with your own key if you have one
```

> 💡 **New!** You can now type `quant desk` from any terminal to jump to your project! See [TERMINAL_SHORTCUTS.md](TERMINAL_SHORTCUTS.md) for all commands.

### 2. Test API Key (Optional)
```bash
# Verify Alpha Vantage API key works (no dependencies needed)
python test_api_key.py
```

### 3. Install Dependencies
```bash
# Install core requirements
pip install -r requirements.txt

# For development (optional)
pip install -e .
```

### 4. Data Ingestion

#### Download Historical Data (5 years)
```bash
cd data
python main.py historical --years 5
```

#### Daily Updates with Hybrid Mode
```bash
cd data
python main.py update --days 5
```

#### Specific Instruments
```bash
cd data
python main.py update --symbols "SPY,AUDUSD=X,CL=F" --days 10
```

#### Disable Hybrid Mode (yfinance only)
```bash
cd data
python main.py historical --years 2 --no-hybrid
```

### 5. Quality Reports
```bash
cd data
python main.py report
```

### 6. Pipeline Statistics
```bash
cd data
python main.py stats
```

## 📊 Data Quality Features

### Quality Scoring (0-100)
- **Completeness** (30%): Non-missing data percentage
- **Consistency** (30%): OHLC relationships and logical checks
- **Timeliness** (20%): Data freshness and update frequency
- **Accuracy** (20%): Statistical accuracy and outlier analysis

### Quality Thresholds
- **Production Ready**: ≥85 (Excellent/Good quality)
- **Review Required**: 70-84 (Fair quality)
- **Reject**: <70 (Poor quality)

### Corporate Actions
- **Split Detection**: >50% overnight price changes
- **Dividend Detection**: <10% price adjustments on ex-dates
- **Confidence Scoring**: 0.0-1.0 based on price patterns and volume

## 🎯 Feature Engineering (Step 3 - COMPLETE)

### Feature Categories
The system generates **150+ features per symbol** across multiple categories:

#### Base Features
- **Returns**: Multiple horizons (1, 5, 20, 60, 120 periods)
- **Volume**: Transformations and rolling statistics (5, 20, 60 periods)
- **Volatility**: Multiple window sizes (5, 20, 60 periods)

#### Technical Indicators
- **Momentum**: RSI (14, 30), MACD, Stochastic, ADX
- **Trend**: Moving average crosses (10/20, 20/50, 50/200)
- **Directional**: Plus/Minus DI, trend strength

#### Market Microstructure
- **Kyle's Lambda**: Price impact (20, 60 window)
- **Spread Analysis**: Bid-ask spread estimation
- **Order Flow**: Volume-based microstructure metrics

#### Regime Detection
- **Volatility Regimes**: 252-day lookback
- **Trend States**: Multiple periods (20, 60, 120)
- **CUSUM**: Change point detection (60 window)

#### Cross-Asset Features
- **Correlations**: Rolling windows (20, 60, 120)
- **Pairs**: Gold-Copper, Oil-Gold, SPY-QQQ, AUD-Gold

#### Commodity-Specific
- **Seasonality**: Time-based patterns
- **Curve Analysis**: Commodity-specific features

### Feature Pipeline Commands

```bash
cd data

# Generate features for all symbols
python main.py features --symbols "SPY,CL=F,GC=F"

# Generate with custom configuration
python main.py features --config features/config/feature_config.yaml

# View feature statistics
python main.py feature-stats
```

### Feature Storage
- **Format**: Parquet with Snappy compression
- **Versioning**: Automatic version control
- **Location**: `data/features/computed/`
- **Reports**: Quality metrics in `data/features/reports/`

## 🔌 API Configuration

### Alpha Vantage (Free Tier)
- **API Key**: Pre-configured in `.env.template` (Z0WR10WWKFEH25JO)
- **Rate Limits**: 5 calls/minute, 25 calls/day
- **Best For**: Recent forex data (last 100 days)
- **Caching**: 24-hour SQLite cache to minimize API usage
- **Ready to Use**: No additional setup required

### Yahoo Finance
- **Rate Limits**: None (respectful usage)
- **Best For**: Historical data, indices, commodities
- **Reliability**: High for data older than 100 days

## 🛠️ Development

### Code Quality
```bash
# Code formatting
black .

# Linting
flake8 .

# Type checking
mypy .

# Tests
pytest
```

### Adding New Data Sources
1. Create adapter in `data/ingestion/`
2. Add configuration to `data/config/data_sources.yaml`
3. Update `HybridDataManager` selection logic
4. Add tests in `tests/`

## 📈 Usage Examples

### Basic Data Download
```python
from data.ingestion import HybridDataManager

# Initialize hybrid manager
manager = HybridDataManager()

# Download with intelligent source selection
data, metadata = manager.download_instrument('SPY', '2023-01-01', '2024-01-01')

print(f"Downloaded {len(data)} rows")
print(f"Sources used: {metadata['sources_used']}")
print(f"Quality score: {metadata['quality_score']}")
```

### Quality Assessment
```python
from data.ingestion import DataValidator, QualityScorer

# Validate data
validator = DataValidator()
cleaned_data, quality_metrics = validator.validate_ohlcv(data, 'SPY')

# Score quality
scorer = QualityScorer()
quality_result = scorer.calculate_composite_score(cleaned_data, 'SPY')

print(f"Overall quality: {quality_result.overall_score}")
print(f"Corporate actions detected: {len(quality_metrics.corporate_actions)}")
```

## 🚨 Monitoring & Alerts

### Quality Monitoring
- **Degradation Alerts**: Quality drops >10 points
- **Minimum Threshold**: Alert if quality <60
- **Trend Tracking**: Historical quality score trends

### API Usage Monitoring
- **Rate Limit Tracking**: Daily quota usage (Alpha Vantage)
- **Performance Monitoring**: Download times and success rates
- **Cache Hit Rates**: API cache effectiveness

## 🔄 Data Pipeline Flow

1. **Source Selection**: HybridDataManager chooses optimal data source
2. **Download**: API calls with rate limiting and caching
3. **Validation**: OHLC consistency, missing value handling
4. **Corporate Actions**: Split/dividend detection and flagging
5. **Quality Scoring**: Composite quality assessment
6. **Storage**: Parquet format with Snappy compression
7. **Cataloging**: Metadata storage with lineage tracking

## 🛡️ Error Handling

- **API Failures**: Automatic fallback between sources
- **Rate Limiting**: Exponential backoff and queue management
- **Data Quality**: Configurable acceptance thresholds
- **Circuit Breaker**: Stop retrying after repeated failures

## 📝 Logging

All operations are logged with structured logging:
- **Data Pipeline**: `data/logs/pipeline_YYYY-MM-DD.log`
- **System**: `logs/system.log`
- **API Calls**: Debug mode for API request/response logging

## 🔮 Next Steps

1. **Signal Generation** (Step 4): ML models for trade signals
2. **Strategy Development** (Step 5): Complete trading strategies
3. **Risk Management** (Step 7): Position sizing and risk controls
4. **Backtesting** (Step 10): Historical strategy performance
5. **Execution** (Step 12): Broker integration and order management
6. **Monitoring** (Step 13): Real-time system monitoring and alerts

## 🤝 Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Use type hints and docstrings
5. Run quality checks before submitting

## 📄 License

This project is for educational and personal use. Please respect API rate limits and terms of service.

---

**Note**: This is a sophisticated trading system. Always test thoroughly with paper trading before using real money. Past performance does not guarantee future results.
