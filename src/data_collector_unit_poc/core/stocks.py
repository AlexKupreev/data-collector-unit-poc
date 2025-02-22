"""Core entities and functions related to commodity markets"""
from dataclasses import dataclass
from datetime import date
from typing import Optional

@dataclass
class CommodityInfo:
    """Commodity market information entity"""
    symbol: str
    name: str
    category: str  # energy, coal, oil, metals
    exchange: str  # e.g., ICE, NYMEX, LME
    unit: str  # e.g., USD/barrel, USD/ton
    currency: str
    last_price: float
    last_update: date

@dataclass
class CommodityHistoricalData:
    """Historical commodity market data entity"""
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    open_interest: float  # Number of outstanding contracts
