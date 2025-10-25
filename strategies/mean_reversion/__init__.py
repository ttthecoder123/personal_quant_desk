"""
Mean Reversion Strategies

Implements mean reversion strategies from Chan's "Quantitative Trading":
- Pairs Trading (cointegration-based)
- Bollinger Band Reversion
- Ornstein-Uhlenbeck Process
- Index Arbitrage
"""

from strategies.mean_reversion.pairs_trading import PairsTradingStrategy
from strategies.mean_reversion.bollinger_reversion import BollingerReversionStrategy
from strategies.mean_reversion.ornstein_uhlenbeck import OUProcessStrategy
from strategies.mean_reversion.index_arbitrage import IndexArbitrageStrategy

__all__ = [
    "PairsTradingStrategy",
    "BollingerReversionStrategy",
    "OUProcessStrategy",
    "IndexArbitrageStrategy",
]
