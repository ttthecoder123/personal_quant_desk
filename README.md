# Personal Quant Desk - AI-Assisted Trading System

A comprehensive, semi-automated trading system for commodities, indices, and FX pairs with AI-driven signal generation and enhanced data ingestion capabilities.

## 🚀 Project Status

- ✅ **Step 1: Project Structure** - COMPLETE
- ✅ **Step 2: Data Ingestion** - COMPLETE (with Alpha Vantage enhancement)
- ✅ **Step 3: Feature Engineering** - COMPLETE
- ✅ **Step 4: Signal Generation** - COMPLETE
- ✅ **Step 5: Strategy Development** - COMPLETE
- ✅ **Step 6: Risk Management** - COMPLETE
- ⏳ **Step 7: Backtesting** - Pending
- ⏳ **Step 8: Execution** - Pending
- ⏳ **Step 9: Monitoring** - Pending

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
│   ├── signals/                 # Signal generation models
│   ├── labeling/                # Triple-barrier labeling
│   ├── training/                # Model training
│   └── config/                  # Model configurations
├── strategies/                  # Trading strategies (Step 5) - COMPLETE
│   ├── base/                    # Core infrastructure
│   │   ├── strategy_base.py     # Abstract base class
│   │   ├── position_manager.py  # Position tracking
│   │   └── performance_tracker.py  # Performance metrics
│   ├── mean_reversion/          # Mean reversion strategies
│   ├── momentum/                # Momentum strategies
│   ├── volatility/              # Volatility strategies
│   ├── hybrid/                  # Hybrid strategies
│   ├── portfolio/               # Portfolio construction
│   ├── execution/               # Execution layer
│   ├── config/                  # Strategy configurations
│   └── strategy_engine.py       # Main orchestration
├── risk/                        # Risk management (Step 6) - COMPLETE
├── execution/                   # Order execution (Step 8)
├── backtesting/                 # Strategy backtesting (Step 7)
├── monitoring/                  # System monitoring (Step 9)
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

## 📈 Signal Generation (Step 4 - COMPLETE)

### ML-Based Signal Generation
The system includes comprehensive machine learning models for trade signal generation:

- **Triple-Barrier Labeling**: Advanced labeling method for supervised learning
- **Meta-Labeling**: Bet sizing using ML confidence scores
- **Random Forest & XGBoost**: Primary prediction models
- **Feature Importance**: SHAP values for model interpretability
- **Signal Confidence**: Probabilistic outputs for position sizing

### Signal Pipeline
```bash
cd models

# Generate signals for all symbols
python main.py generate-signals --symbols "SPY,CL=F,GC=F"

# Train new models
python main.py train --model-type xgboost
```

## 🎯 Strategy Development (Step 5 - COMPLETE)

### Comprehensive Strategy System
Industrial-strength strategy development and portfolio construction implementing methodologies from Chan, Carver, and Jansen.

### Strategy Categories

#### Mean Reversion Strategies
- **Pairs Trading**: Cointegration-based pairs with Johansen test, OLS hedge ratios
- **Bollinger Reversion**: Dynamic Bollinger Bands with RSI divergence and volume confirmation
- **Ornstein-Uhlenbeck**: OU process parameter estimation with mean reversion half-life
- **Index Arbitrage**: Index vs constituent basket arbitrage

#### Momentum Strategies
- **Trend Following**: Carver's multi-timeframe EWMA with forecast scaling (-20 to +20)
- **Breakout Momentum**: Donchian channel breakouts with volatility squeeze detection
- **Cross-Sectional**: Relative momentum ranking across instruments
- **Time Series Momentum**: Sign-based momentum (Moskowitz et al.)

#### Volatility Strategies
- **Volatility Targeting**: Carver's systematic volatility targeting (20% annual)
- **Vol Arbitrage**: Implied vs realized volatility trading with Greeks
- **Gamma Scalping**: Delta-neutral gamma trading with rehedging
- **Dispersion Trading**: Index vs components volatility dispersion

#### Hybrid Strategies
- **ML-Enhanced**: Combines discretionary rules with ML predictions
- **Regime Switching**: Market regime detection (HMM or volatility-based)
- **Multi-Factor**: Combines momentum, volatility, value, quality factors
- **Ensemble**: Dynamic strategy combination with online learning

### Portfolio Construction (Carver Framework)

- **Portfolio Optimizer**: Forecast combination with correlation adjustment
- **Risk Parity**: Equal Risk Contribution (ERC) and Hierarchical Risk Parity (HRP)
- **Kelly Sizing**: Optimal position sizing with confidence weighting
- **Correlation Manager**: Rolling correlation with regime-dependent penalties
- **Rebalancer**: Threshold-based with transaction cost optimization

### Execution Layer

- **Order Generator**: Signal-to-order conversion with aggregation and netting
- **Execution Algorithms**: TWAP, VWAP, participation rate, adaptive execution
- **Slippage Model**: Linear and square-root market impact models
- **Cost Model**: Commission, fees, margin interest, currency conversion

### Strategy Engine Commands

```bash
# Initialize strategy engine
from strategies import StrategyEngine

# Create engine with default configs
engine = StrategyEngine(
    strategy_config_path="strategies/config/strategy_config.yaml",
    portfolio_config_path="strategies/config/portfolio_config.yaml",
    initial_capital=100000.0
)

# Generate signals for instruments
signals = engine.update_signals(
    symbols=["SPY", "CL=F", "GC=F"],
    market_data=market_data,
    features=features,
    ml_signals=ml_signals
)

# Calculate target positions
positions = engine.calculate_positions()

# Generate executable orders
orders = engine.generate_orders()

# Update performance metrics
metrics = engine.update_performance()

# Check risk limits
within_limits, violations = engine.check_risk_limits()

# Get summary
print(engine.get_summary())
```

### Configuration

**Strategy Configuration** (`strategies/config/strategy_config.yaml`):
- Individual strategy parameters
- Entry/exit thresholds
- Risk limits per strategy
- ML confidence thresholds
- Performance targets

**Portfolio Configuration** (`strategies/config/portfolio_config.yaml`):
- Portfolio-level constraints (20% vol target, 2x max leverage)
- Position limits (20% max per position, 2% max risk)
- Diversification settings (IDM calculation, max correlation 0.85)
- Rebalancing parameters (drift threshold, time interval)
- Risk budgeting across strategy categories

### Risk Management

- **Volatility Targeting**: 20% annualized portfolio volatility
- **Position Limits**: 20% max allocation per position, 2% max loss
- **Drawdown Control**: 20% maximum drawdown limit
- **Correlation Limits**: 85% maximum position correlation
- **Leverage Constraints**: 2x maximum leverage with margin buffers

### Performance Metrics

**Strategy-Level**:
- Sharpe ratio (target > 1.0)
- Sortino ratio, Calmar ratio
- Maximum drawdown, current drawdown
- Win rate, profit factor
- Kelly criterion calculation

**Portfolio-Level**:
- Portfolio Sharpe ratio
- Effective number of bets
- Risk contribution by strategy
- Correlation to benchmarks
- Drawdown duration

### Integration Points

- **Step 2 (Data)**: Real-time and historical data from ParquetStorage
- **Step 3 (Features)**: 150+ engineered features for signal generation
- **Step 4 (Models)**: ML signals with meta-labels and triple-barrier exits
- **Step 6 (Risk)**: Institutional-grade risk management with VaR, stress testing, and circuit breakers
- **Future Steps**: Backtesting (Step 7), Execution (Step 8), Monitoring (Step 9)

## 🛡️ Risk Management (Step 6 - COMPLETE)

### Institutional-Grade Risk Management System
Comprehensive risk management framework implementing industry best practices from JP Morgan's RiskMetrics, Basel III requirements, and quantitative risk management literature (McNeil, Taleb, Jorion).

### System Architecture

The risk management system consists of 8 major components with 45+ modules providing end-to-end risk monitoring and control:

#### 1. Core Risk Engine
- **VaR Models** (`var_models.py`): Historical, Parametric (Normal/t-dist), Monte Carlo, Cornish-Fisher
- **Risk Metrics** (`risk_metrics.py`): VaR, CVaR/ES, Maximum Drawdown, Volatility, Beta
- **Stress Testing** (`stress_testing.py`): Historical scenarios (2008, 2020), hypothetical shocks, sensitivity analysis
- **Risk Engine** (`risk_engine.py`): Real-time risk calculation with caching and aggregation

#### 2. Position Sizing
- **Dynamic Sizing** (`dynamic_sizing.py`): Signal strength-based, ML confidence weighting
- **Volatility Targeting** (`volatility_targeting.py`): ATR-based sizing, inverse volatility scaling (20% target)
- **Kelly Optimizer** (`kelly_optimizer.py`): Full/fractional Kelly, win rate estimation, Kelly with constraints
- **Risk Budgeting** (`risk_budgeting.py`): Equal Risk Contribution (ERC), risk parity allocation

#### 3. Portfolio Risk
- **Concentration Risk** (`concentration_risk.py`): HHI calculation, single name limits (20% max)
- **Correlation Risk** (`correlation_risk.py`): Rolling correlation matrices, regime-dependent penalties
- **Tail Risk** (`tail_risk.py`): Extreme value theory (GEV/GPD), tail dependence, copulas
- **Liquidity Risk** (`liquidity_risk.py`): Amihud illiquidity, bid-ask spread analysis, market impact

#### 4. Drawdown Control
- **Drawdown Manager** (`drawdown_manager.py`): Real-time tracking, peak-to-trough measurement
- **Circuit Breakers** (`circuit_breakers.py`): Multi-level triggers (5%, 10%, 15%, 20%)
- **Stop Loss System** (`stop_loss_system.py`): Individual position stops, time-based stops, trailing stops
- **Recovery Rules** (`recovery_rules.py`): Position reduction schedules, gradual ramp-up after breaches

#### 5. Market Risk
- **Volatility Forecasting** (`volatility_forecasting.py`): GARCH(1,1), EWMA, realized volatility
- **Regime Detection** (`regime_detection.py`): HMM-based, volatility regimes, bull/bear/sideways
- **Factor Risk** (`factor_risk.py`): Factor exposure analysis, Fama-French factors, risk attribution
- **Correlation Dynamics** (`correlation_dynamics.py`): DCC-GARCH, correlation forecasting

#### 6. Operational Risk
- **Model Risk** (`model_risk.py`): Model drift detection, prediction stability monitoring
- **Execution Risk** (`execution_risk.py`): Slippage tracking, fill rate monitoring, reject analysis
- **Data Risk** (`data_risk.py`): Missing data detection, outlier identification, staleness checks
- **System Risk** (`system_risk.py`): Health checks, latency monitoring, error rate tracking

#### 7. Reporting & Analytics
- **Risk Reports** (`risk_reports.py`): Daily risk summaries, VaR reports, limit utilization
- **Risk Dashboard** (`risk_dashboard.py`): Real-time risk visualization, Plotly dashboards
- **Compliance Reports** (`compliance_reports.py`): Regulatory reporting, audit trails
- **Attribution** (`attribution.py`): P&L attribution by strategy/asset, risk contribution analysis

#### 8. Alert System
- **Alert Manager** (`alert_manager.py`): Real-time alert generation, alert prioritization
- **Thresholds** (`thresholds.py`): Configurable warning/critical thresholds
- **Notification Channels** (`notification_channels.py`): Email, SMS, webhook integrations

### Risk Limits & Controls

**Portfolio-Level Limits**:
- **VaR Limit**: 2% daily VaR (95% confidence) - typical institutional standard
- **Maximum Drawdown**: 20% from equity peak before circuit breaker activation
- **Volatility Target**: 20% annualized portfolio volatility with dynamic scaling
- **Leverage Constraint**: 2x maximum gross leverage with 1.5x typical target
- **Liquidity Buffer**: 10% cash reserve for margin calls and opportunities

**Position-Level Limits**:
- **Single Position**: 20% maximum allocation per instrument
- **Position Loss**: 2% maximum loss per position (automatic stop loss)
- **Sector Concentration**: 40% maximum per asset class
- **Correlation Limit**: 0.85 maximum pairwise correlation between positions

**Market Risk Limits**:
- **Beta Exposure**: ±0.5 net beta to benchmarks (SPY, commodities index)
- **Sector Limits**: Commodities 40%, Equities 40%, FX 30%
- **Greek Limits**: Delta ±100, Gamma ±50, Vega ±30 (for options if applicable)

### Circuit Breaker Levels

**Level 1 (5% Drawdown)**: Warning alerts, increase monitoring frequency
**Level 2 (10% Drawdown)**: Reduce position sizes by 25%, review all strategies
**Level 3 (15% Drawdown)**: Reduce position sizes by 50%, halt new positions
**Level 4 (20% Drawdown)**: Close all positions, system halt, requires manual restart

### Stress Testing Scenarios

**Historical Scenarios**:
- **2008 Financial Crisis**: -40% equity, +80% volatility, credit freeze
- **2020 COVID Crash**: -35% equities, circuit breaker triggers
- **Flash Crash 2010**: Extreme intraday volatility spike
- **Oil Price Collapse 2020**: WTI negative pricing event

**Hypothetical Scenarios**:
- **Volatility Shock**: VIX +100% (doubles)
- **Correlation Breakdown**: All correlations → 1.0 (crisis mode)
- **Liquidity Crisis**: Bid-ask spreads widen 5x
- **Fat Tail Event**: 5-sigma moves across markets

### Risk Controller Integration

The **RiskController** (`risk_controller.py`) is the main entry point for risk management:

```python
from risk import RiskController

# Initialize risk controller
risk_controller = RiskController(
    config_path="risk/config/risk_limits.yaml",
    initial_capital=100000.0
)

# Pre-trade risk checks
can_trade, violations = risk_controller.check_pre_trade(
    symbol="SPY",
    quantity=100,
    price=450.0,
    side="BUY",
    current_positions=positions
)

if not can_trade:
    print(f"Trade blocked: {violations}")

# Update positions and calculate risk metrics
risk_controller.update_positions(positions)
metrics = risk_controller.calculate_risk_metrics()

print(f"Portfolio VaR (95%): ${metrics['var_95']:.2f}")
print(f"Expected Shortfall: ${metrics['cvar_95']:.2f}")
print(f"Current Drawdown: {metrics['current_drawdown']:.2%}")
print(f"Leverage: {metrics['leverage']:.2f}x")

# Check if circuit breakers triggered
breaker_status = risk_controller.check_circuit_breakers()
if breaker_status['triggered']:
    print(f"Circuit breaker Level {breaker_status['level']} triggered!")
    print(f"Action required: {breaker_status['action']}")

# Generate risk report
risk_report = risk_controller.generate_daily_report()
risk_controller.save_report(risk_report, "reports/risk_report_2024-01-15.pdf")
```

### Configuration Files

**Risk Limits** (`config/risk_limits.yaml`):
- Portfolio and position-level limits
- VaR parameters and confidence levels
- Circuit breaker thresholds
- Leverage and concentration constraints

**Alert Rules** (`config/alert_rules.yaml`):
- Warning and critical thresholds
- Alert recipients and channels
- Escalation procedures
- Alert cooldown periods

**Stress Scenarios** (`config/scenarios.yaml`):
- Historical scenario definitions
- Hypothetical shock parameters
- Correlation assumptions
- Recovery time estimates

### Key Features

**Real-Time Monitoring**:
- Continuous VaR calculation (60-second refresh)
- Live drawdown tracking from equity peak
- Position-level risk contribution analysis
- Automatic alert generation on threshold breaches

**Preventive Controls**:
- Pre-trade risk checks block violations before execution
- Automatic position sizing based on volatility
- Kelly criterion prevents over-leveraging
- Correlation penalties reduce crowded trades

**Reactive Controls**:
- Circuit breakers halt trading on excessive drawdown
- Automatic stop losses limit position losses
- Recovery rules gradually restore after breaches
- Stress test results inform position adjustments

**Reporting & Compliance**:
- Daily risk reports with VaR, ES, drawdown metrics
- Trade-level audit trail for compliance
- Real-time risk dashboard visualization
- Monthly risk attribution analysis

### Integration Points

- **Step 5 (Strategies)**: Position sizing and portfolio optimization integrate with risk limits
- **Step 2 (Data)**: Market data feeds real-time risk calculations
- **Step 4 (Signals)**: ML confidence scores inform position sizing
- **Future Steps**: Backtesting (Step 7) will validate risk controls, Execution (Step 8) enforces pre-trade checks

### Modules Created (45+ Files)

**Core** (4): VaR models, risk metrics, stress testing, risk engine
**Position Sizing** (4): Dynamic sizing, volatility targeting, Kelly optimizer, risk budgeting
**Portfolio Risk** (4): Concentration, correlation, tail risk, liquidity risk
**Drawdown Control** (4): Drawdown manager, circuit breakers, stop losses, recovery rules
**Market Risk** (4): Volatility forecasting, regime detection, factor risk, correlation dynamics
**Operational Risk** (4): Model risk, execution risk, data risk, system risk
**Reporting** (4): Risk reports, dashboard, compliance, attribution
**Alerts** (3): Alert manager, thresholds, notification channels
**Main Controllers** (2): RiskManager, RiskController
**Configuration** (3): risk_limits.yaml, alert_rules.yaml, scenarios.yaml
**Package Inits** (9): Module __init__.py files

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

1. **Backtesting** (Step 7): Historical strategy performance testing with walk-forward analysis
2. **Execution** (Step 8): Broker integration and order management
3. **Monitoring** (Step 9): Real-time system monitoring and alerts

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
