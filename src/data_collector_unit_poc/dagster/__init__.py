"""Dagster entry point"""
import warnings

from dagster import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)

from dagster import (  # noqa: E402
    Definitions,
)

from .jobs import current_time_job
from .schedules import current_time_schedule

jobs = [
    current_time_job,
]

schedules = [
    current_time_schedule,
]

defs = Definitions(
    jobs=jobs,
    schedules=schedules,
)
