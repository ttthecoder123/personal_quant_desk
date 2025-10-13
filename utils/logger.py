"""
Logging configuration using loguru for the Quantitative Trading System.
"""

import sys
import os
from pathlib import Path
from typing import Optional
from loguru import logger
import yaml


class TradingLogger:
    """Custom logger configuration for the trading system."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the trading logger.

        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self._configured = False

    def _load_config(self, config_path: Optional[str] = None) -> dict:
        """Load logging configuration from yaml file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('logging', {})

        # Default configuration
        return {
            'level': 'INFO',
            'format': '{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}',
            'rotation': '100 MB',
            'retention': '30 days',
            'compression': 'zip',
            'files': {
                'main': 'logs/trading_system.log',
                'trades': 'logs/trades.log',
                'errors': 'logs/errors.log',
                'performance': 'logs/performance.log'
            }
        }

    def setup(self):
        """Configure the logger with the specified settings."""
        if self._configured:
            return

        # Remove default handler
        logger.remove()

        # Add console handler with color
        logger.add(
            sys.stdout,
            level=self.config['level'],
            format=self.config['format'],
            colorize=True,
            backtrace=True,
            diagnose=True
        )

        # Create logs directory if it doesn't exist
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        # Add file handlers
        for log_type, log_path in self.config['files'].items():
            # Ensure log directory exists
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)

            # Configure different log levels for different files
            if log_type == 'errors':
                level = 'ERROR'
            elif log_type == 'trades':
                level = 'INFO'
            else:
                level = self.config['level']

            # Add file handler
            logger.add(
                log_path,
                level=level,
                format=self.config['format'],
                rotation=self.config['rotation'],
                retention=self.config['retention'],
                compression=self.config['compression'],
                backtrace=True,
                diagnose=True,
                enqueue=True  # Thread-safe
            )

        self._configured = True
        logger.info("Trading System Logger initialized successfully")

    def get_logger(self, name: str = None):
        """
        Get a logger instance with optional name binding.

        Args:
            name: Optional name for the logger context

        Returns:
            Logger instance
        """
        if not self._configured:
            self.setup()

        if name:
            return logger.bind(name=name)
        return logger


# Create global logger instance
trading_logger = TradingLogger()
trading_logger.setup()

# Export logger for easy access
log = trading_logger.get_logger()


# Specialized loggers for different components
def get_trade_logger():
    """Get logger for trade execution."""
    return trading_logger.get_logger('TradeExecution')


def get_risk_logger():
    """Get logger for risk management."""
    return trading_logger.get_logger('RiskManagement')


def get_data_logger():
    """Get logger for data operations."""
    return trading_logger.get_logger('DataPipeline')


def get_strategy_logger():
    """Get logger for strategy execution."""
    return trading_logger.get_logger('Strategy')


def get_backtest_logger():
    """Get logger for backtesting."""
    return trading_logger.get_logger('Backtesting')


def get_ml_logger():
    """Get logger for machine learning models."""
    return trading_logger.get_logger('MachineLearning')


# Decorator for function logging
def log_execution(func):
    """Decorator to log function execution."""
    def wrapper(*args, **kwargs):
        logger.info(f"Executing {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.success(f"Successfully executed {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error executing {func.__name__}: {str(e)}")
            raise
    return wrapper


# Custom log levels for trading events
logger.level("TRADE", no=35, color="<cyan>", icon="💰")
logger.level("SIGNAL", no=25, color="<yellow>", icon="📊")
logger.level("ALERT", no=45, color="<red>", icon="🚨")


def log_trade(message: str, **kwargs):
    """Log trade execution events."""
    logger.log("TRADE", message, **kwargs)


def log_signal(message: str, **kwargs):
    """Log trading signals."""
    logger.log("SIGNAL", message, **kwargs)


def log_alert(message: str, **kwargs):
    """Log critical alerts."""
    logger.log("ALERT", message, **kwargs)