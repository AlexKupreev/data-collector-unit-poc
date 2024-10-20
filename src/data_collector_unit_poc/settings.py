"""Common settings"""
import os

EMBED_CHUNK_SIZE = 40
MAX_CONTENT_VECTORIZED = 1000 * 5

environment = os.getenv("ENVIRONMENT", "production")

s3_noaa_isd_path = "data/noaa/isd"
    
if environment == "local":
    noaa_isd_local_persistent_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "..",
        "..",
        "data",
        "noaa",
        "isd"
    )
else:
    noaa_isd_local_persistent_path = "/data/noaa/isd/"
