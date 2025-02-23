"""Commodity market-related jobs"""
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

import yfinance as yf
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyorc
from deltalake import write_deltalake
from dotenv import load_dotenv

from data_collector_unit_poc.settings import environment
from data_collector_unit_poc.core.stocks import CommodityInfo, CommodityHistoricalData

# Define commodity symbols for different categories
ENERGY_COMMODITIES = [
    ("NG=F", "Natural Gas", "NYMEX", "USD/MMBtu"),
    ("RB=F", "RBOB Gasoline", "NYMEX", "USD/gallon"),
    ("HO=F", "Heating Oil", "NYMEX", "USD/gallon"),
]

OIL_COMMODITIES = [
    ("CL=F", "Crude Oil WTI", "NYMEX", "USD/barrel"),
    ("BZ=F", "Brent Crude Oil", "ICE", "USD/barrel"),
]

COAL_COMMODITIES = [
    ("MTF=F", "Coal Futures", "NYMEX", "USD/ton"),
    ("FCAG=F", "Coal Futures (API2)", "ICE", "USD/ton"),  # Rotterdam Coal Futures
]

INDUSTRIAL_METALS_COMMODITIES = [
    ("HG=F", "Copper", "COMEX", "USD/lb"),
    ("ALI=F", "Aluminum", "LME", "USD/ton"),
    ("ZNC=F", "Zinc", "LME", "USD/ton"),
    ("PB=F", "Lead", "LME", "USD/ton"),
    ("NI=F", "Nickel", "LME", "USD/ton"),
]

PRECIOUS_METALS_COMMODITIES = [
    ("GC=F", "Gold", "COMEX", "USD/oz"),
    ("SI=F", "Silver", "COMEX", "USD/oz"),
    ("PL=F", "Platinum", "NYMEX", "USD/oz"),
    ("PA=F", "Palladium", "NYMEX", "USD/oz"),
]

# Combined metals category for classification
METALS_COMMODITIES = INDUSTRIAL_METALS_COMMODITIES + PRECIOUS_METALS_COMMODITIES

ALL_COMMODITIES = (
    ENERGY_COMMODITIES + 
    OIL_COMMODITIES + 
    COAL_COMMODITIES + 
    INDUSTRIAL_METALS_COMMODITIES + 
    PRECIOUS_METALS_COMMODITIES
)

def get_commodity_info(symbol: str, name: str, exchange: str, unit: str) -> CommodityInfo | None:
    """Get basic information about a commodity"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        return CommodityInfo(
            symbol=symbol,
            name=name,
            category=next(cat for cat, commodities in [
                ("energy", ENERGY_COMMODITIES),
                ("oil", OIL_COMMODITIES),
                ("coal", COAL_COMMODITIES),
                ("metals", METALS_COMMODITIES)
            ] if (symbol, name, exchange, unit) in commodities),
            exchange=exchange,
            unit=unit,
            currency="USD",  # All commodities are typically priced in USD
            last_price=float(info.get('regularMarketPrice', 0)),
            last_update=datetime.now().date()
        )
    except Exception as e:
        print(f"Error fetching info for {symbol}: {str(e)}")
        return None

def download_commodity_historical_data(symbol: str, period: str = "max") -> pd.DataFrame:
    """Download historical data for a commodity in bulk"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            print(f"No data available for {symbol}")
            return pd.DataFrame()
        
        # Reset index to make date a column
        df = df.reset_index()
        
        # Rename columns to match our schema
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Open Interest': 'open_interest'
        })
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Handle open interest separately since it might not exist
        if 'open_interest' not in df.columns:
            df['open_interest'] = 0
        else:
            df['open_interest'] = df['open_interest'].fillna(0)
        
        # Ensure proper types
        df = df.astype({
            'open': float,
            'high': float,
            'low': float,
            'close': float,
            'volume': float,
            'open_interest': float
        })
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date']).dt.date
        
        return df
    except Exception as e:
        print(f"Error downloading data for {symbol}: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame instead of list

StorageFormat = Literal["csv", "parquet", "avro", "orc", "delta"]

def build_local_commodity_storage_path(symbol: str, format: StorageFormat = "csv") -> str:
    """Build local storage path for commodity data in specified format"""
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "commodities")
    os.makedirs(base_path, exist_ok=True)
    
    extensions = {
        "csv": "csv.gz",
        "parquet": "parquet",
        "avro": "avro",
        "orc": "orc",
        "delta": "delta",  # Delta Lake uses a directory
    }
    return os.path.join(base_path, f"{symbol.replace('=', '_')}.{extensions[format]}")

def store_data_parquet(df: pd.DataFrame, filepath: str):
    """Store DataFrame in Parquet format, preserving historical data"""
    # Add ingestion timestamp
    df['ingested_at'] = pd.Timestamp.now()
    
    # If file exists, append new data to existing data
    if os.path.exists(filepath):
        existing_df = pd.read_parquet(filepath)
        # Combine existing and new data, drop duplicates keeping latest version
        combined_df = pd.concat([existing_df, df])
        combined_df = combined_df.sort_values('date')
        # Convert to table and write
        table = pa.Table.from_pandas(combined_df)
    else:
        # For first write
        table = pa.Table.from_pandas(df)
    
    pq.write_table(table, filepath)

def store_data_orc(df: pd.DataFrame, filepath: str):
    """Store DataFrame in ORC format using pandas"""
    df.reset_index().to_orc(filepath, index=False)

def store_data_delta(df: pd.DataFrame, filepath: str):
    """Store DataFrame in Delta Lake format"""
    write_deltalake(filepath, df, mode="overwrite")

def store_commodity_data(symbol: str, df: pd.DataFrame, cancel_event=None):
    """Store commodity data with focus on historical data preservation in Parquet format"""
    if df.empty:
        print(f"No data available for {symbol}")
        return
    
    # Use Parquet as primary format for historical data
    format: StorageFormat = "parquet"
    if cancel_event and cancel_event.is_set():
        print("Job cancellation requested, stopping...")
        return
        
    filepath = build_local_commodity_storage_path(symbol, format)
    print(f"Storing {symbol} historical data in Parquet format...")
    
    try:
        store_data_parquet(df, filepath)
    except Exception as e:
        print(f"Error storing {symbol} data in Parquet format: {str(e)}")

def ingest_commodity_data_job(cancel_event=None, period: str = "max"):
    """Ingest commodity market data
    
    Args:
        cancel_event: Event to cancel the job
        period: Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    """
    total_commodities = len(ALL_COMMODITIES)
    for idx, (symbol, name, exchange, unit) in enumerate(ALL_COMMODITIES, 1):
        if cancel_event and cancel_event.is_set():
            print("Job cancellation requested, stopping...")
            return
            
        print(f"Processing commodity {idx}/{total_commodities}: {name} ({symbol})")
        
        # Get commodity info
        info = get_commodity_info(symbol, name, exchange, unit)
        if info:
            print(f"Retrieved info for {symbol}: {info.name} ({info.category})")
        
        # Get historical data
        historical_data = download_commodity_historical_data(symbol, period)
        store_commodity_data(symbol, historical_data, cancel_event)
        
        # Sleep to avoid hitting API rate limits
        time.sleep(2)

def ingest_commodity_initial_data_job(cancel_event=None):
    """Ingest maximum available historical data for all commodities"""
    return ingest_commodity_data_job(cancel_event, period="max")

def ingest_commodity_daily_data_job(cancel_event=None):
    """Ingest last day's data for all commodities"""
    return ingest_commodity_data_job(cancel_event, period="1d")

if __name__ == "__main__":
    started_at = time.time()
    ingest_commodity_data_job()
    completed_at = time.time()
    time_spent = (completed_at - started_at)
    print(f"Completed at {datetime.now()}, execution took ~{int(time_spent / 60)} min")
