"""
Configuration module for the Evolutionary Trading System.
Centralizes all configuration parameters for easy management.
"""
import os
from dataclasses import dataclass
from typing import Dict, Any
from enum import Enum
import logging

class MarketMode(Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

class NeuroEvolutionMethod(Enum):
    NEAT = "neat"
    HYPERNEAT = "hyperneat"
    CPPN = "cppn"

@dataclass
class TradingConfig:
    """Configuration for trading parameters"""
    initial_balance: float = 10000.0
    max_position_size: float = 0.1  # 10% of portfolio
    slippage_percent: float = 0.001  # 0.1% slippage
    commission_percent: float = 0.001  # 0.1% commission
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        if self.initial_balance <= 0:
            raise ValueError("Initial balance must be positive")
        if not 0 < self.max_position_size <= 1:
            raise ValueError("Max position size must be between 0 and 1")
        if self.slippage_percent < 0:
            raise ValueError("Slippage cannot be negative")
        return True

@dataclass
class RLConfig:
    """Configuration for reinforcement learning"""
    gamma: float = 0.99  # Discount factor
    learning_rate: float = 0.001
    batch_size: int = 32
    memory_size: int = 10000
    update_target_every: int = 100
    
    def validate(self) -> bool:
        if not 0 <= self.gamma <= 1:
            raise ValueError("Gamma must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        return True

@dataclass
class NEATConfig:
    """Configuration for NEAT algorithm"""
    population_size: int = 50
    max_generations: int = 100
    compatibility_threshold: float = 3.0
    survival_threshold: float = 0.2
    
    def validate(self) -> bool:
        if self.population_size < 10:
            raise ValueError("Population size must be at least 10")
        return True

@dataclass
class DataConfig:
    """Configuration for data handling"""
    candle_timeframe: str = "1h"
    lookback_window: int = 50
    features: list = None
    
    def __post_init__(self):
        if self.features is None:
            self.features = [
                'open', 'high', 'low', 'close', 'volume',
                'returns', 'volatility', 'rsi', 'macd'
            ]
    
    def validate(self) -> bool:
        valid_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        if self.candle_timeframe not in valid_timeframes:
            raise ValueError(f"Timeframe must be one of {valid_timeframes}")
        if self.lookback_window < 10:
            raise ValueError("Lookback window must be at least 10")
        return True

class ConfigManager:
    """Central configuration manager"""
    
    def __init__(self):
        self.mode = MarketMode(os.getenv('TRADING_MODE', 'paper'))
        self.trading = TradingConfig()
        self.rl = RLConfig()
        self.neat = NEATConfig()
        self.data = DataConfig()
        
        # Validate all configurations
        self.validate_all()
        
        # Setup logging
        self.setup_logging()
    
    def validate_all(self) -> None:
        """Validate all configurations"""
        try:
            self.trading.validate()
            self.rl.validate()
            self.neat.validate()
            self.data.validate()
        except ValueError as e:
            logging.error(f"Configuration validation failed: {e}")
            raise
    
    def setup_logging(self) -> None:
        """Configure logging system"""
        log_level = os.getenv('LOG_LEVEL', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('evolution_trading.log')
            ]
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for Firebase storage"""
        return {
            'mode': self.mode.value,
            'trading': self.trading.__dict__,
            'rl': self.rl.__dict__,
            'neat': self.neat.__dict__,
            'data': self.data.__dict__
        }