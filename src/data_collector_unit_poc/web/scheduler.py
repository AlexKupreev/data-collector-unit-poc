from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

from data_collector_unit_poc.jobs.weather import ingest_noaa_isd_lite_job


def my_hourly_task():
    print(f"Hourly task is running at {datetime.now()}")


def build_now_trigger() -> DateTrigger:
    """Build a trigger that runs right when invoked"""
    return DateTrigger()    

# Set up the scheduler
scheduler = BackgroundScheduler()

ingest_noaa_isd_lite_trigger = CronTrigger(hour="8")
scheduler.add_job(
    ingest_noaa_isd_lite_job,
    ingest_noaa_isd_lite_trigger
)

scheduler.start()

# Ensure the scheduler shuts down properly on application exit.
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    scheduler.shutdown()
