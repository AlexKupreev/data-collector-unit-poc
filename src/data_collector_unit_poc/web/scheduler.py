from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Dict
import threading
import uuid

from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.job import Job

from data_collector_unit_poc.jobs.weather import ingest_noaa_isd_lite_job
from data_collector_unit_poc.jobs.commodities import (
    ingest_commodity_initial_data_job,
    ingest_commodity_daily_data_job
)

# Track currently running jobs and their cancellation flags
running_jobs: Dict[str, threading.Event] = {}
running_jobs_lock = threading.Lock()

def track_job_execution(func):
    """Decorator to track job execution and support cancellation"""
    def wrapper(*args, **kwargs):
        job_id = str(uuid.uuid4())
        cancel_event = threading.Event()
        with running_jobs_lock:
            running_jobs[job_id] = cancel_event
        try:
            # Pass the cancel_event to the function
            kwargs['cancel_event'] = cancel_event
            return func(*args, **kwargs)
        finally:
            with running_jobs_lock:
                running_jobs.pop(job_id, None)
    return wrapper

@track_job_execution
def my_hourly_task(**kwargs):
    print(f"Hourly task is running at {datetime.now()}")

@track_job_execution
def wrapped_ingest_noaa_isd_lite_job(**kwargs):
    cancel_event = kwargs.pop('cancel_event', None)
    # Pass the cancel_event to the actual job function
    return ingest_noaa_isd_lite_job(cancel_event=cancel_event)

@track_job_execution
def wrapped_ingest_commodity_initial_data_job(**kwargs):
    cancel_event = kwargs.pop('cancel_event', None)
    # Pass the cancel_event to the actual job function
    return ingest_commodity_initial_data_job(cancel_event=cancel_event)

@track_job_execution
def wrapped_ingest_commodity_daily_data_job(**kwargs):
    cancel_event = kwargs.pop('cancel_event', None)
    # Pass the cancel_event to the actual job function
    return ingest_commodity_daily_data_job(cancel_event=cancel_event)


def build_now_trigger() -> DateTrigger:
    """Build a trigger that runs right when invoked"""
    return DateTrigger()    


def get_running_jobs() -> List[Dict]:
    """Get list of currently running jobs and scheduled jobs"""
    jobs = []
    
    # Add scheduled jobs
    for job in scheduler.get_jobs():
        job_info = {
            'id': job.id,
            'name': job.name or job.func.__name__,
            'next_run_time': job.next_run_time.strftime('%Y-%m-%d %H:%M:%S') if job.next_run_time else None,
            'trigger': str(job.trigger),
            'status': 'scheduled'
        }
        jobs.append(job_info)
    
    # Add currently running jobs
    with running_jobs_lock:
        for job_id, _ in running_jobs.items():
            job_info = {
                'id': job_id,
                'name': 'Running Job',
                'next_run_time': None,
                'trigger': 'manual',
                'status': 'running'
            }
            jobs.append(job_info)
    
    return jobs


def terminate_job(job_id: str) -> bool:
    """Terminate a running job by its ID"""
    # First try to remove from scheduler
    job = scheduler.get_job(job_id)
    if job:
        scheduler.remove_job(job_id)
        return True
    
    # Then check if it's a running job
    with running_jobs_lock:
        cancel_event = running_jobs.get(job_id)
        if cancel_event:
            # Set the cancellation flag
            cancel_event.set()
            return True
    
    return False


# Set up the scheduler
scheduler = BackgroundScheduler()

# Schedule weather data ingestion
ingest_noaa_isd_lite_trigger = CronTrigger(hour="8")
scheduler.add_job(
    wrapped_ingest_noaa_isd_lite_job,  # Use wrapped version
    ingest_noaa_isd_lite_trigger
)

# Schedule commodity data ingestion (run at 7 AM daily)
ingest_commodity_daily_trigger = CronTrigger(hour="7")
scheduler.add_job(
    wrapped_ingest_commodity_daily_data_job,  # Use wrapped version for daily updates
    ingest_commodity_daily_trigger
)

scheduler.start()

# Ensure the scheduler shuts down properly on application exit.
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    scheduler.shutdown()
