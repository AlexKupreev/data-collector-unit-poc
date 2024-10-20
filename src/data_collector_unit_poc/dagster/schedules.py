from dagster import ScheduleDefinition

from .jobs import current_time_job


current_time_schedule = ScheduleDefinition(
    job=current_time_job,
    cron_schedule="*/5 * * * *",
    execution_timezone="UTC"
)
