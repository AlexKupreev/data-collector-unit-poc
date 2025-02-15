"""Weather-related jobs"""
import gzip
import os
import time
from datetime import date, datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Literal, Union
from uuid import uuid4

import boto3
import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyorc
from deltalake import DeltaTable, write_deltalake
from hudi import HudiTable
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import IntegerType, LongType, NestedField
from dotenv import load_dotenv

from data_collector_unit_poc.settings import (
    environment,
    noaa_isd_local_persistent_path,
    s3_noaa_isd_path,
)
from data_collector_unit_poc.core.weather import NoaaIsdLocation


def store_noaa_isd():
    """Store downloaded NOAA ISD files."""


def get_noaa_isd_locations() -> list[NoaaIsdLocation]:
    """Get the ISD locations of interest.
    
    Fields in the dataframe:
    - USAF
    - WBAN
    - STATION NAME
    - CTRY
    - ST
    - CALL
    - LAT
    - LON
    - ELEV(M)
    - BEGIN
    - END
    """
    if environment == "local":
        isd_history_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "..",
            "..",
            "..",
            "data",
            "isd-history-clean.txt"
        )
    else:
        isd_history_path = "/app/data/isd-history-clean.txt"
        
    df_locations = pd.read_fwf(isd_history_path)
    
    # take the first 100 locations that have data after Jan  1, 2025 and ICAO
    df_locations["end_date"] = pd.to_datetime(df_locations["END"], format="%Y%m%d")
    df_locations_icao_actual = df_locations[(~df_locations["CALL"].isna()) & (df_locations["end_date"] > pd.Timestamp(year=2025, month=1, day=1))]
    
    if environment == "local":
        num_locations = 10
    else:
        num_locations = 100
        
    chosen = df_locations_icao_actual.iloc[:num_locations,:]
    locations = []
    for _, row in chosen.iterrows():
        location = NoaaIsdLocation(
            usaf=row["USAF"],
            wban=int(row["WBAN"]),
            station_name=row.get("STATION NAME"),
            country=row.get("CTRY"),
            us_state=row.get("ST"),
            icao=row.get("CALL"),
            lat=row.get("LAT"),
            lon=row.get("LON"),
            elevation=row.get("ELEV(M)"),
            begin=pd.to_datetime(row["BEGIN"], format="%Y%m%d").date(),
            end=pd.to_datetime(row["END"], format="%Y%m%d").date()
        )
        locations.append(location)
    return locations


StorageFormat = Literal["csv", "parquet", "avro", "orc", "delta", "hudi", "iceberg"]

def build_local_noaa_isd_storage_path(
    location: NoaaIsdLocation,
    format: StorageFormat = "csv"
) -> str:
    """Build local storage path for location NOAA ISD weather in specified format"""
    extensions = {
        "csv": "csv.gz",
        "parquet": "parquet",
        "avro": "avro",
        "orc": "orc",
        "delta": "delta",  # Delta Lake uses a directory
        "hudi": "hudi",    # Hudi uses a directory
        "iceberg": "iceberg"  # Iceberg uses a directory
    }
    return os.path.join(
        noaa_isd_local_persistent_path,
        f"{location.usaf}-{location.wban}.{extensions[format]}"
    )

def store_data_orc(df: pd.DataFrame, filepath: str):
    """Store DataFrame in ORC format"""
    schema = "struct<year:bigint,month:bigint,day:bigint,hour:bigint,air_temp:bigint,dew_point_temp:bigint,sea_level_pressure:bigint,wind_direction:bigint,wind_speed_rate:bigint,sky_condition:bigint,precipitation_depth_1h:bigint,precipitation_depth_6h:bigint>"
    
    with open(filepath, 'wb') as f:
        with pyorc.Writer(f, schema) as writer:
            for _, row in df.iterrows():
                writer.write(tuple(row))
                
def store_data_orc_pandas(df: pd.DataFrame, filepath: str):
    """Store DataFrame in ORC format using pandas"""
    # TODO check if de-indexing works
    df.loc[:,:].reset_index().to_orc(filepath, index=False)

def read_data_orc(filepath: str) -> pd.DataFrame:
    """Read ORC file into DataFrame"""
    with open(filepath, 'rb') as f:
        reader = pyorc.Reader(f)
        data = [row for row in reader]
    return pd.DataFrame(data)

def read_data_orc_pandas(filepath: str) -> pd.DataFrame:
    """Read ORC file into DataFrame using pandas"""
    return pd.read_orc(filepath)

def store_data_delta(df: pd.DataFrame, filepath: str):
    """Store DataFrame in Delta Lake format"""
    write_deltalake(filepath, df, mode="overwrite")

def read_data_delta(filepath: str) -> pd.DataFrame:
    """Read Delta Lake table into DataFrame"""
    dt = DeltaTable(filepath)
    return dt.to_pandas()

def store_data_hudi(df: pd.DataFrame, filepath: str):
    """Store DataFrame in Apache Hudi format"""
    # Convert to parquet first since Hudi Python API is limited
    parquet_path = f"{filepath}/data.parquet"
    hudi_dirname = os.path.dirname(parquet_path)
    os.makedirs(hudi_dirname, exist_ok=True)
    store_data_parquet(df, parquet_path)
    
    # Use Hudi's file-based indexing
    os.makedirs(f"{filepath}/.hoodie", exist_ok=True)
    with open(f"{filepath}/.hoodie/hoodie.properties", "w") as f:
        f.write(f"""
hoodie.table.name=weather_{uuid4().hex[:8]}
hoodie.datasource.write.recordkey.field=year,month,day,hour
hoodie.datasource.write.precombine.field=year
""")

def read_data_hudi(filepath: str) -> pd.DataFrame:
    """Read Hudi table into DataFrame"""
    # Read the parquet file directly since we're using file-based indexing
    return pd.read_parquet(f"{filepath}/data.parquet")

def store_data_iceberg(df: pd.DataFrame, filepath: str):
    """Store DataFrame in Apache Iceberg format"""
    # Define schema
    schema = Schema(
        NestedField(1, "year", IntegerType(), required=True),
        NestedField(2, "month", IntegerType(), required=True),
        NestedField(3, "day", IntegerType(), required=True),
        NestedField(4, "hour", IntegerType(), required=True),
        NestedField(5, "air_temp", IntegerType(), required=False),
        NestedField(6, "dew_point_temp", IntegerType(), required=False),
        NestedField(7, "sea_level_pressure", IntegerType(), required=False),
        NestedField(8, "wind_direction", IntegerType(), required=False),
        NestedField(9, "wind_speed_rate", IntegerType(), required=False),
        NestedField(10, "sky_condition", IntegerType(), required=False),
        NestedField(11, "precipitation_depth_1h", IntegerType(), required=False),
        NestedField(12, "precipitation_depth_6h", IntegerType(), required=False)
    )
    
    # Create catalog
    catalog = load_catalog(
        "local",
        uri='thrift://localhost:9083/'
        # warehouse_path=os.path.dirname(filepath),
        # catalog_impl="hive"
    )
    
    # Create table
    table_name = f"weather_{uuid4().hex[:8]}"
    catalog.create_table(
        identifier=table_name,
        schema=schema,
        location=filepath
    )
    
    # Write data
    table = catalog.load_table(table_name)
    table.append(df)

def read_data_iceberg(filepath: str) -> pd.DataFrame:
    """Read Iceberg table into DataFrame"""
    catalog = load_catalog(
        "local",
        warehouse_path=os.path.dirname(filepath),
        catalog_impl="hive"
    )
    table = catalog.load_table(os.path.basename(filepath))
    return table.scan().to_pandas()

def store_data_parquet(df: pd.DataFrame, filepath: str):
    """Store DataFrame in Parquet format"""
    table = pa.Table.from_pandas(df)
    pq.write_table(table, filepath)

def read_data_parquet(filepath: str) -> pd.DataFrame:
    """Read Parquet file into DataFrame"""
    return pd.read_parquet(filepath)


def download_noaa_isd_for_year(location: NoaaIsdLocation, year: int) -> pd.DataFrame:
    """Load data for NOAA ISD location for a year"""
    columns = [
        "year", "month", "day", "hour", "air_temp", "dew_point_temp",
        "sea_level_pressure", "wind_direction", "wind_speed_rate",
        "sky_condition", "precipitation_depth_1h", "precipitation_depth_6h"
    ]
    
    try:
        with httpx.Client() as client:
            url = f"https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/{year}/{location.usaf}-{location.wban}-{year}.gz"
            
            print(f"getting data from URL {url}...")
            response = client.get(url, timeout=40)
            response.raise_for_status()  # Ensure we raise an error for bad responses

            # Define column specifications based on the fixed-width format
            colspecs = [
                (0, 4),    # Year
                (5, 7),    # Month
                (8, 10),   # Day
                (11, 13),  # Hour
                (14, 19),  # Air Temperature
                (20, 25),  # Dew Point Temperature
                (26, 31),  # Sea Level Pressure
                (32, 37),  # Wind Direction
                (38, 43),  # Wind Speed Rate
                (44, 49),  # Sky Condition
                (50, 55),  # Precipitation Depth 1 hour
                (56, 61),  # Precipitation Depth 6 hours
            ]

            # Decompress the gzipped content
            with gzip.open(BytesIO(response.content), 'rt') as gz_file:
                df = pd.read_fwf(
                    gz_file,
                    colspecs=colspecs,
                    header=None,
                    na_values=-9999,
                    dtype="Int64"
                )
                
            df.columns = columns

    except httpx.HTTPStatusError as ex:
        print(f"HTTP status ERROR: {ex!s} (location: {location}, year: {year})")
        df = pd.DataFrame(columns=columns)
        
    except Exception as ex:
        print(f"ERROR: {ex!s} (location: {location}, year: {year})")
        df = pd.DataFrame(columns=columns)
        # sleep a bit to let the connection reset if needed
        time.sleep(10)
        
    return df

def store_location_weather_data(location: NoaaIsdLocation, cancel_event=None):
    """Store location NOAA ISD data in multiple formats.
    
    If no file exists, go through the full history and merge it.
    If the file exists, get data only for the current year
    (and previous if it's still the beginning of a year).
    Data is stored in multiple formats:
    - CSV (gzipped)
    - Parquet
    - Avro
    - ORC
    - Delta Lake
    - Apache Hudi
    - Apache Iceberg
    """
    # ignore catalog types for now : delta, hudi, iceberg
    formats: list[StorageFormat] = ["csv", "parquet", "avro", "orc"]
    filepaths = {
        format: build_local_noaa_isd_storage_path(location, format)
        for format in formats
    }
    BUCKET_NAME = os.getenv("BUCKET_NAME")
    
    chunks = []
    # # Check if any format file exists
    if not any(os.path.isfile(fp) for fp in filepaths.values()):
    # # for now rewrite all files
    # if True:
        # see for the start year, get and merge all years' data
        for filepath in filepaths.values():
            filedir = os.path.dirname(filepath)
            os.makedirs(filedir, exist_ok=True)
        
        for year in range(location.begin.year, location.end.year + 1):
            if cancel_event and cancel_event.is_set():
                print("Job cancellation requested, stopping...")
                return
            print("started processing:", location, year)
            chunks.append(download_noaa_isd_for_year(location, year))
            print("completed processing:", location, year)
        
    else:
        # take only the current year + the previous if no more than 10 days passed
        try:
            # Try to read from any existing format, preferring CSV
            for format, filepath in filepaths.items():
                if os.path.isfile(filepath):
                    if format == "csv":
                        stored = pd.read_csv(filepath, compression="infer", dtype="Int64")
                    elif format == "parquet":
                        stored = read_data_parquet(filepath)
                    elif format == "avro":
                        stored = pd.read_parquet(filepath)  # temporarily use parquet for avro
                    elif format == "orc":
                        stored = read_data_orc_pandas(filepath)
                    elif format == "delta":
                        stored = read_data_delta(filepath)
                    elif format == "hudi":
                        stored = read_data_hudi(filepath)
                    else:  # iceberg
                        stored = read_data_iceberg(filepath)
                    break
            chunks.append(stored)
            
            current_date = pd.Timestamp.now()
            last_year_end = current_date.replace(year=current_date.year - 1, month=12, day=31)
            years = []
            if current_date - pd.Timedelta(days=10) < last_year_end:
                years.append(current_date.year - 1)
                
            years.append(current_date.year)
            
            for year in years:
                if cancel_event and cancel_event.is_set():
                    print("Job cancellation requested, stopping...")
                    return
                print("started processing:", location, year)
                chunks.append(download_noaa_isd_for_year(location, year))
                print("completed processing:", location, year)
        
        except EOFError as exc:
            print(f"Error during decompression of file {filepath}: {exc!s}. Delete stored file to recreate later...")
            os.unlink(filepath)
    
    filled_chunks = [x for x in chunks if not x.empty]
    if filled_chunks:
        full = pd.concat(filled_chunks)
        
        # we need to overwrite old values, so remove duplicates preserving last ones
        full.drop_duplicates(subset=["year", "month", "day", "hour"], keep="last")
        
        # Store in all formats
        for format, filepath in filepaths.items():
            print(f"Storing data in {format} format...")
            if format == "csv":
                full.to_csv(filepath, index=False, compression="infer")
            elif format == "parquet":
                store_data_parquet(full, filepath)
            elif format == "avro":
                store_data_parquet(full, filepath)  # temporarily use parquet for avro
            elif format == "orc":
                store_data_orc_pandas(full, filepath)
            elif format == "delta":
                store_data_delta(full, filepath)
            elif format == "hudi":
                store_data_hudi(full, filepath)
            else:  # iceberg
                store_data_iceberg(full, filepath)
            
            # and push to s3
            if environment == "production" and BUCKET_NAME:
                s3_client = boto3.client('s3')
                
                backup_name = os.path.basename(filepath)
                backup_path = f"{s3_noaa_isd_path}/{backup_name}"
                print(f"upload file {filepath} to backup_path {backup_path}")
                s3_client.upload_file(filepath, BUCKET_NAME, backup_path)
    else:
        print(f"No data found for location: {location}")


def ingest_noaa_isd_lite_job(cancel_event=None):
    """Ingest NOAA ISD data"""
    scope = get_noaa_isd_locations()
    total_locations = len(scope)
    curr_loc_num = 0
    for location in scope:
        # Check if cancellation was requested
        if cancel_event and cancel_event.is_set():
            print("Job cancellation requested, stopping...")
            return
            
        curr_loc_num += 1
        print(f"store data for location {curr_loc_num}/{total_locations}: {location}")
        store_location_weather_data(location, cancel_event)
        time.sleep(5)


if __name__ == "__main__":
    started_at = time.time()
    locations = get_noaa_isd_locations()
    store_location_weather_data(locations[0])
    # ingest_noaa_isd_lite_job()
    completed_at = time.time()
    time_spent = (completed_at - started_at)
    print(f"Completed at {datetime.now()}, execution took ~{int(time_spent / 60)} min")
    print(locations[:10])
