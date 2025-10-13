"""
Create minimal synthetic test data for feature engineering validation.
Data is organized by year as expected by ParquetStorage.
"""
import pandas as pd
import numpy as np
from pathlib import Path

dates = pd.date_range('2020-01-01', '2024-10-01', freq='D')
n = len(dates)

for symbol in ['SPY', 'GC=F']:
    base_price = 300 if symbol == 'SPY' else 1800
    returns = np.random.normal(0.0003, 0.01, n)
    prices = base_price * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
        'High': prices * (1 + np.random.uniform(0, 0.01, n)),
        'Low': prices * (1 + np.random.uniform(-0.01, 0, n)),
        'Close': prices,
        'Volume': np.random.randint(50000000, 150000000, n)
    }, index=dates)
    
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    for year in range(2020, 2025):
        year_data = df[df.index.year == year]
        if not year_data.empty:
            year_dir = Path(f'data/processed/{year}')
            year_dir.mkdir(parents=True, exist_ok=True)
            year_data.to_parquet(year_dir / f'{symbol}.parquet')
            print(f'Created {symbol} data for {year}: {len(year_data)} rows')

print('\nTest data created successfully!')
