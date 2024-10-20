from datetime import UTC, datetime

from dagster import (
    job,
    op,
    get_dagster_logger
)


@op
def get_current_utc_time():
    current_utc_time = datetime.now(tz=UTC)
    
    get_dagster_logger().info(f"Current UTC time: {current_utc_time}")


@job
def current_time_job():
    get_current_utc_time()
