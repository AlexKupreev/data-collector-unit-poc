"""Specific data formats related jobs"""
import time
import os
import glob
from collections import defaultdict

import pandas as pd

from data_collector_unit_poc.jobs.weather import (
    read_data_parquet,
    read_data_orc_pandas,
)
from data_collector_unit_poc.settings import noaa_isd_local_persistent_path

NUM_FILES_TO_BENCHMARK = 100

def run_format_benchmark() -> dict[str, dict[str, float]]:
    """Run benchmark for different data formats"""
    formats = {
        'parquet': ('*.parquet', read_data_parquet),
        'orc': ('*.orc', read_data_orc_pandas),
        'csv.gz': ('*.csv.gz', lambda f: pd.read_csv(f, compression='gzip'))
    }
    
    results = {}
    
    for format_name, (pattern, reader_func) in formats.items():
        # Find N files of this format
        files = glob.glob(os.path.join(noaa_isd_local_persistent_path, pattern))
        if not files:
            continue
        
        files = files[:NUM_FILES_TO_BENCHMARK]
        
        format_times = []
        total_start = time.time()
        
        for file in files:
            try:
                # Time reading each file
                start = time.time()
                df = reader_func(file)
                end = time.time()
                
                # Immediately delete dataframe to free memory
                del df
                
                format_times.append(end - start)
            except Exception as e:
                print(f"Error reading {file}: {e}")
                
        total_end = time.time()
        
        if format_times:
            results[format_name] = {
                'total_time': total_end - total_start,
                'mean_time': sum(format_times) / len(format_times),
                'num_files': len(files)
            }
    
    return results
