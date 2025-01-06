"""Data storage layer"""
import os
import logging

import boto3
from dotenv import load_dotenv
from fastapi import FastAPI

logger = logging.getLogger('uvicorn.error')

load_dotenv()

S3_URL = os.getenv("AWS_ENDPOINT_URL_S3")

def init(fastapi_app: FastAPI) -> None:
    """Initialize data storage assets"""
    if S3_URL:
        s3_client = boto3.client('s3', endpoint_url=S3_URL)
        buckets = s3_client.list_buckets()
        logger.info(str(buckets))
        
        # if data sqlite is missing, download it to the persistent storage from s3
        # we should have only one bucket per app for now
        BUCKET_NAME = os.getenv("BUCKET_NAME")
        environment = os.getenv("ENVIRONMENT", "production")
        db_filepath = "/data/posts.db"
        s3_path = "data"
        
        logger.debug("BUCKET_NAME: %s", BUCKET_NAME)
        if environment == "production":
            if os.path.isfile(db_filepath):
                # if a file exists, load it to s3 for backup
                backup_name = os.path.basename(db_filepath) + ".bak"
                backup_path = f"/{s3_path}/{backup_name}"
                
                logger.debug("backup_path: %s", backup_path)
                s3_client.upload_file(db_filepath, BUCKET_NAME, backup_path)
            else:
                # if a file does not exist, download it from s3 to persistent storage
                store_path = f"{s3_path}/{os.path.basename(db_filepath)}"
                
                logger.debug("store_path: %s", store_path)
                
                s3_client.download_file(BUCKET_NAME, store_path, db_filepath)
