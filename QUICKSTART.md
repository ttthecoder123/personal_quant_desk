# Personal Quant Desk - Quick Start Guide

🚀 **Ready to use in 5 minutes!** The system comes pre-configured with a working Alpha Vantage API key.

## ⚡ Ultra-Quick Start (3 Commands)

```bash
# 1. Jump to project (from anywhere!)
quant desk

# 2. Set up environment (API key already included!)
cp .env.template .env

# 3. Install dependencies
pip install -r requirements.txt

# 4. Test with real data download
quant data && python main.py update --symbols "SPY" --days 3
```

## 🎯 What You Get Out of the Box

✅ **Working Alpha Vantage API Key** - No signup needed
✅ **Hybrid Data Strategy** - Smart source selection
✅ **Quality Scoring** - Enterprise-grade validation
✅ **Corporate Action Detection** - Automatic split/dividend flagging
✅ **Professional Caching** - Respects API rate limits

## 📊 Try These Examples

### Basic Stock Data (Yahoo Finance)
```bash
cd data
python main.py update --symbols "SPY,QQQ" --days 5 --no-hybrid
```

### Forex Data (Alpha Vantage - High Quality)
```bash
cd data
python main.py update --symbols "AUDUSD=X,EURUSD=X" --days 5
```

### Commodities (Hybrid Strategy)
```bash
cd data
python main.py update --symbols "CL=F,GC=F" --days 10
```

### Historical Download (5 years)
```bash
cd data
python main.py historical --years 2 --symbols "SPY"
```

## 📈 View Your Data

After downloading, check the results:

```bash
# View quality report
cd data
python main.py report

# Check pipeline stats
python main.py stats

# Manual inspection
ls -la processed/  # Your parquet files
ls -la cache/      # API cache database
```

## 🔍 Verify Everything Works

```bash
# Run comprehensive verification
python verify_setup.py

# Test imports without installing everything
cd data
python -c "print('Testing...'); import yaml; print('✅ Basic imports work')"
```

## 🎮 Interactive Examples

### Python API Usage
```python
# Navigate to the data directory first
cd data

# Start Python
python

# Test the hybrid manager
>>> from ingestion import HybridDataManager
>>> manager = HybridDataManager()
>>> data, metadata = manager.download_instrument('SPY', '2024-01-01', '2024-01-10')
>>> print(f"Downloaded {len(data)} rows")
>>> print(f"Quality score: {metadata.get('quality_score', 'N/A')}")
>>> print(f"Sources used: {metadata.get('sources_used', 'N/A')}")
```

### Quality Assessment
```python
>>> from ingestion import DataValidator
>>> validator = DataValidator()
>>> cleaned_data, metrics = validator.validate_ohlcv(data, 'SPY')
>>> print(f"Quality score: {metrics.quality_score:.2f}")
>>> print(f"Corporate actions: {len(metrics.corporate_actions)}")
```

## 🚨 Troubleshooting

### If you get "No module named 'pandas'"
```bash
# Install just the core data dependencies first
pip install pandas yfinance requests pyyaml loguru click
```

### If Alpha Vantage rate limit exceeded
```bash
# Use yfinance only mode
cd data
python main.py update --symbols "SPY" --days 5 --no-hybrid
```

### If you want your own Alpha Vantage key
1. Get free key at: https://www.alphavantage.co/support/#api-key
2. Edit `.env` file and replace the API key
3. Higher limits available with premium plans

## 🎉 What's Next?

Once you have data flowing:

1. **Explore in Jupyter**: `jupyter notebook notebooks/`
2. **Check Data Quality**: Review the HTML reports in `data/quality_reports/`
3. **Customize Instruments**: Edit `data/config/instruments.yaml`
4. **Advanced Configuration**: Modify `data/config/data_sources.yaml`

## 💡 Pro Tips

- **Start Small**: Test with 1-2 symbols and few days first
- **Check Logs**: All operations logged in `data/logs/`
- **Monitor Cache**: SQLite cache in `data/cache/` saves API calls
- **Quality First**: Check quality scores before analysis
- **Hybrid Mode**: Let the system choose best data source automatically

---

**Ready to build your quant trading system? Start with the 3-command setup above!** 🚀