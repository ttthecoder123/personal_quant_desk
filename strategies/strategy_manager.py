"""
Strategy Manager for coordinating and executing trading strategies.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
import asyncio

from utils.logger import get_strategy_logger, log_signal

log = get_strategy_logger()


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, name: str, config: dict):
        """Initialize base strategy."""
        self.name = name
        self.config = config
        self.positions = {}
        self.signals = []

    @abstractmethod
    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Generate trading signals based on market data."""
        pass

    @abstractmethod
    def calculate_position_size(self, signal: Dict, portfolio_value: float) -> float:
        """Calculate the position size for a given signal."""
        pass


class MomentumStrategy(BaseStrategy):
    """Momentum-based trading strategy."""

    def __init__(self, config: dict):
        """Initialize momentum strategy."""
        super().__init__("Momentum", config)
        self.lookback_period = config.get('lookback_period', 20)
        self.holding_period = config.get('holding_period', 5)
        self.entry_threshold = config.get('entry_threshold', 2.0)
        self.exit_threshold = config.get('exit_threshold', 0.5)
        self.universe_size = config.get('universe_size', 10)

    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Generate momentum signals."""
        signals = []

        try:
            # Calculate momentum scores for each instrument
            momentum_scores = {}
            for symbol, data in market_data.items():
                if len(data) < self.lookback_period:
                    continue

                # Calculate momentum (rate of change)
                returns = data['Close'].pct_change(self.lookback_period)
                current_return = returns.iloc[-1]

                # Calculate z-score
                mean_return = returns.mean()
                std_return = returns.std()
                z_score = (current_return - mean_return) / std_return if std_return > 0 else 0

                momentum_scores[symbol] = {
                    'return': current_return,
                    'z_score': z_score,
                    'price': data['Close'].iloc[-1]
                }

            # Select top momentum stocks
            sorted_scores = sorted(
                momentum_scores.items(),
                key=lambda x: x[1]['z_score'],
                reverse=True
            )

            # Generate buy signals for top performers
            for symbol, score in sorted_scores[:self.universe_size]:
                if score['z_score'] > self.entry_threshold:
                    signal = {
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'strategy': self.name,
                        'action': 'BUY',
                        'price': score['price'],
                        'strength': min(score['z_score'] / self.entry_threshold, 2.0),
                        'metadata': {
                            'momentum_return': score['return'],
                            'z_score': score['z_score']
                        }
                    }
                    signals.append(signal)
                    log_signal(f"Momentum BUY signal: {symbol} (z-score: {score['z_score']:.2f})")

            # Generate sell signals for positions exceeding holding period or weak momentum
            for symbol in self.positions:
                if symbol in momentum_scores:
                    score = momentum_scores[symbol]
                    if score['z_score'] < self.exit_threshold:
                        signal = {
                            'timestamp': datetime.now(),
                            'symbol': symbol,
                            'strategy': self.name,
                            'action': 'SELL',
                            'price': score['price'],
                            'strength': 1.0,
                            'metadata': {
                                'momentum_return': score['return'],
                                'z_score': score['z_score']
                            }
                        }
                        signals.append(signal)
                        log_signal(f"Momentum SELL signal: {symbol} (z-score: {score['z_score']:.2f})")

        except Exception as e:
            log.error(f"Error generating momentum signals: {str(e)}")

        return signals

    def calculate_position_size(self, signal: Dict, portfolio_value: float) -> float:
        """Calculate position size based on signal strength."""
        base_size = portfolio_value / self.universe_size
        return base_size * signal['strength']


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion trading strategy."""

    def __init__(self, config: dict):
        """Initialize mean reversion strategy."""
        super().__init__("MeanReversion", config)
        self.lookback_period = config.get('lookback_period', 30)
        self.bb_config = config.get('bollinger_bands', {})
        self.rsi_config = config.get('rsi', {})

    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Generate mean reversion signals."""
        signals = []

        try:
            for symbol, data in market_data.items():
                if len(data) < self.lookback_period:
                    continue

                # Calculate Bollinger Bands
                bb_period = self.bb_config.get('period', 20)
                bb_std = self.bb_config.get('std_dev', 2.0)

                sma = data['Close'].rolling(window=bb_period).mean()
                std = data['Close'].rolling(window=bb_period).std()
                upper_band = sma + (bb_std * std)
                lower_band = sma - (bb_std * std)

                # Calculate RSI
                rsi_period = self.rsi_config.get('period', 14)
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))

                current_price = data['Close'].iloc[-1]
                current_rsi = rsi.iloc[-1]
                current_upper = upper_band.iloc[-1]
                current_lower = lower_band.iloc[-1]

                # Generate buy signal (oversold conditions)
                if (current_price < current_lower and
                    current_rsi < self.rsi_config.get('oversold', 30)):
                    signal = {
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'strategy': self.name,
                        'action': 'BUY',
                        'price': current_price,
                        'strength': 1.0 - (current_rsi / 30),
                        'metadata': {
                            'rsi': current_rsi,
                            'bb_position': 'below_lower',
                            'bb_lower': current_lower,
                            'bb_upper': current_upper
                        }
                    }
                    signals.append(signal)
                    log_signal(f"Mean Reversion BUY: {symbol} (RSI: {current_rsi:.2f})")

                # Generate sell signal (overbought conditions)
                elif (current_price > current_upper and
                      current_rsi > self.rsi_config.get('overbought', 70)):
                    signal = {
                        'timestamp': datetime.now(),
                        'symbol': symbol,
                        'strategy': self.name,
                        'action': 'SELL',
                        'price': current_price,
                        'strength': (current_rsi - 70) / 30,
                        'metadata': {
                            'rsi': current_rsi,
                            'bb_position': 'above_upper',
                            'bb_lower': current_lower,
                            'bb_upper': current_upper
                        }
                    }
                    signals.append(signal)
                    log_signal(f"Mean Reversion SELL: {symbol} (RSI: {current_rsi:.2f})")

        except Exception as e:
            log.error(f"Error generating mean reversion signals: {str(e)}")

        return signals

    def calculate_position_size(self, signal: Dict, portfolio_value: float) -> float:
        """Calculate position size based on signal strength."""
        max_position = portfolio_value * 0.05  # 5% max per position
        return max_position * signal['strength']


class ArbitrageStrategy(BaseStrategy):
    """Statistical arbitrage strategy for correlated pairs."""

    def __init__(self, config: dict):
        """Initialize arbitrage strategy."""
        super().__init__("Arbitrage", config)
        self.correlation_threshold = config.get('pairs_correlation_threshold', 0.80)
        self.z_score_entry = config.get('z_score_entry', 2.0)
        self.z_score_exit = config.get('z_score_exit', 0.5)
        self.lookback_period = config.get('lookback_period', 60)
        self.pairs = []

    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Generate arbitrage signals for correlated pairs."""
        signals = []

        try:
            # Find correlated pairs
            symbols = list(market_data.keys())
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    symbol1, symbol2 = symbols[i], symbols[j]
                    data1, data2 = market_data[symbol1], market_data[symbol2]

                    if len(data1) < self.lookback_period or len(data2) < self.lookback_period:
                        continue

                    # Calculate correlation
                    returns1 = data1['Close'].pct_change().dropna()
                    returns2 = data2['Close'].pct_change().dropna()

                    if len(returns1) != len(returns2):
                        continue

                    correlation = returns1.tail(self.lookback_period).corr(
                        returns2.tail(self.lookback_period)
                    )

                    if abs(correlation) >= self.correlation_threshold:
                        # Calculate spread z-score
                        spread = data1['Close'] / data2['Close']
                        spread_mean = spread.rolling(self.lookback_period).mean()
                        spread_std = spread.rolling(self.lookback_period).std()
                        z_score = (spread.iloc[-1] - spread_mean.iloc[-1]) / spread_std.iloc[-1]

                        # Generate signals based on z-score
                        if abs(z_score) > self.z_score_entry:
                            if z_score > 0:
                                # Spread is high: sell symbol1, buy symbol2
                                signals.append({
                                    'timestamp': datetime.now(),
                                    'symbol': symbol1,
                                    'strategy': self.name,
                                    'action': 'SELL',
                                    'price': data1['Close'].iloc[-1],
                                    'strength': min(abs(z_score) / self.z_score_entry, 2.0),
                                    'metadata': {
                                        'pair': symbol2,
                                        'correlation': correlation,
                                        'z_score': z_score
                                    }
                                })
                                signals.append({
                                    'timestamp': datetime.now(),
                                    'symbol': symbol2,
                                    'strategy': self.name,
                                    'action': 'BUY',
                                    'price': data2['Close'].iloc[-1],
                                    'strength': min(abs(z_score) / self.z_score_entry, 2.0),
                                    'metadata': {
                                        'pair': symbol1,
                                        'correlation': correlation,
                                        'z_score': -z_score
                                    }
                                })
                            else:
                                # Spread is low: buy symbol1, sell symbol2
                                signals.append({
                                    'timestamp': datetime.now(),
                                    'symbol': symbol1,
                                    'strategy': self.name,
                                    'action': 'BUY',
                                    'price': data1['Close'].iloc[-1],
                                    'strength': min(abs(z_score) / self.z_score_entry, 2.0),
                                    'metadata': {
                                        'pair': symbol2,
                                        'correlation': correlation,
                                        'z_score': z_score
                                    }
                                })
                                signals.append({
                                    'timestamp': datetime.now(),
                                    'symbol': symbol2,
                                    'strategy': self.name,
                                    'action': 'SELL',
                                    'price': data2['Close'].iloc[-1],
                                    'strength': min(abs(z_score) / self.z_score_entry, 2.0),
                                    'metadata': {
                                        'pair': symbol1,
                                        'correlation': correlation,
                                        'z_score': -z_score
                                    }
                                })

                            log_signal(f"Arbitrage signal: {symbol1}/{symbol2} (z-score: {z_score:.2f})")

        except Exception as e:
            log.error(f"Error generating arbitrage signals: {str(e)}")

        return signals

    def calculate_position_size(self, signal: Dict, portfolio_value: float) -> float:
        """Calculate position size for arbitrage trades."""
        max_position = portfolio_value * 0.10  # 10% max per leg
        return max_position * signal['strength']


class StrategyManager:
    """Manages multiple trading strategies."""

    def __init__(self, config: dict):
        """Initialize strategy manager."""
        self.config = config
        self.strategies = {}
        self._initialize_strategies()

    def _initialize_strategies(self):
        """Initialize configured strategies."""
        log.info("Initializing trading strategies...")

        # Initialize Momentum Strategy
        if self.config.get('momentum', {}).get('enabled', False):
            self.strategies['momentum'] = MomentumStrategy(self.config['momentum'])
            log.info("Momentum strategy initialized")

        # Initialize Mean Reversion Strategy
        if self.config.get('mean_reversion', {}).get('enabled', False):
            self.strategies['mean_reversion'] = MeanReversionStrategy(
                self.config['mean_reversion']
            )
            log.info("Mean Reversion strategy initialized")

        # Initialize Arbitrage Strategy
        if self.config.get('arbitrage', {}).get('enabled', False):
            self.strategies['arbitrage'] = ArbitrageStrategy(self.config['arbitrage'])
            log.info("Arbitrage strategy initialized")

        log.success(f"Initialized {len(self.strategies)} strategies")

    async def generate_signals(self, market_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Generate signals from all active strategies."""
        all_signals = []

        # Run strategies in parallel
        tasks = []
        for name, strategy in self.strategies.items():
            tasks.append(strategy.generate_signals(market_data))

        results = await asyncio.gather(*tasks)

        # Combine all signals
        for signals in results:
            all_signals.extend(signals)

        log.info(f"Generated {len(all_signals)} total signals")
        return all_signals

    def update_positions(self, executed_trades: List[Dict]):
        """Update strategy positions based on executed trades."""
        for trade in executed_trades:
            strategy_name = trade.get('strategy')
            if strategy_name in self.strategies:
                strategy = self.strategies[strategy_name]
                symbol = trade['symbol']

                if trade['action'] == 'BUY':
                    if symbol not in strategy.positions:
                        strategy.positions[symbol] = {
                            'quantity': 0,
                            'entry_price': 0,
                            'entry_time': None
                        }
                    strategy.positions[symbol]['quantity'] += trade['quantity']
                    strategy.positions[symbol]['entry_price'] = trade['price']
                    strategy.positions[symbol]['entry_time'] = trade['timestamp']

                elif trade['action'] == 'SELL':
                    if symbol in strategy.positions:
                        strategy.positions[symbol]['quantity'] -= trade['quantity']
                        if strategy.positions[symbol]['quantity'] <= 0:
                            del strategy.positions[symbol]

    async def stop(self):
        """Stop all strategies."""
        log.info("Stopping all strategies...")
        # Perform any cleanup if needed
        log.info("Strategies stopped")