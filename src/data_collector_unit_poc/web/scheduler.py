from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler  # runs tasks in the background
from apscheduler.triggers.cron import CronTrigger  # allows to specify a recurring time for execution

from data_collector_unit_poc.jobs.summarizer_ingest_poc import main


def my_hourly_task():
    print(f"Hourly task is running at {datetime.now()}")
    

def ingest_news():
    main()   


# Set up the scheduler
scheduler = BackgroundScheduler()

trigger = CronTrigger(hour=7, minute=0)
hourly_trigger = CronTrigger(minute=30)

scheduler.add_job(my_hourly_task, hourly_trigger)
scheduler.add_job(ingest_news, trigger)
scheduler.start()

# Ensure the scheduler shuts down properly on application exit.
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    scheduler.shutdown()
