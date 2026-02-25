"""
Market data fetcher with real-time capabilities and error handling.
Uses CCXT for exchange connectivity with fallback mechanisms.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
from dataclasses import dataclass
from abc import ABC, abstractmethod
from .firebase_manager import FirebaseManager

@dataclass
class MarketData:
    """Data class for standardized market data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: str
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'symbol': self