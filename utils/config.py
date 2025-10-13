"""
Configuration utilities for the trading system.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from dataclasses import dataclass

from utils.logger import log


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str
    port: int
    database: str
    user: str
    password: str


@dataclass
class BrokerConfig:
    """Broker configuration."""
    name: str
    account_id: str
    api_host: str
    api_port: int
    client_id: str


class ConfigManager:
    """Manages system configuration."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration manager."""
        self.config_path = Path(config_path)
        self.config = {}
        self.load_config()

    def load_config(self):
        """Load configuration from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        load_dotenv()  # Load environment variables

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Replace environment variable placeholders
        self._replace_env_vars(self.config)

        log.info("Configuration loaded successfully")

    def _replace_env_vars(self, config: Dict) -> Dict:
        """Replace environment variable placeholders in configuration."""
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]  # Remove ${ and }
                    config[key] = os.getenv(env_var, value)
                elif isinstance(value, (dict, list)):
                    self._replace_env_vars(value)
        elif isinstance(config, list):
            for item in config:
                if isinstance(item, (dict, list)):
                    self._replace_env_vars(item)

        return config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        db_config = self.get('data.storage.database', {})
        return DatabaseConfig(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 5432),
            database=db_config.get('database', 'trading_db'),
            user=db_config.get('user', 'postgres'),
            password=db_config.get('password', '')
        )

    def get_broker_config(self) -> BrokerConfig:
        """Get broker configuration."""
        broker_config = self.get('execution.broker', {})
        return BrokerConfig(
            name=broker_config.get('name', 'interactive_brokers'),
            account_id=broker_config.get('account_id', ''),
            api_host=broker_config.get('api_host', '127.0.0.1'),
            api_port=broker_config.get('api_port', 7497),
            client_id=broker_config.get('client_id', 1)
        )

    def get_instruments(self) -> Dict[str, Dict]:
        """Get trading instruments configuration."""
        return self.get('instruments', {})

    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration."""
        return self.get('risk_management', {})

    def get_strategies_config(self) -> Dict[str, Any]:
        """Get strategies configuration."""
        return self.get('strategies', {})

    def validate_config(self) -> bool:
        """Validate configuration completeness."""
        required_sections = [
            'system',
            'instruments',
            'risk_management',
            'data',
            'execution',
            'strategies'
        ]

        for section in required_sections:
            if section not in self.config:
                log.error(f"Missing required configuration section: {section}")
                return False

        log.info("Configuration validation passed")
        return True

    def reload_config(self):
        """Reload configuration from file."""
        log.info("Reloading configuration...")
        self.load_config()


# Global configuration instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Get global configuration manager."""
    return config_manager