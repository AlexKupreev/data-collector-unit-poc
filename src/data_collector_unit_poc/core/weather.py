"""Core entities and functions related to weather"""
from dataclasses import dataclass
from datetime import date

@dataclass
class NoaaIsdLocation:
    """NOAA ISD location entity"""
    usaf: str
    wban: int
    station_name: str | None
    country: str | None
    us_state: str | None
    icao: str | None
    lat: float | None
    lon: float | None
    elevation: float | None
    begin: date
    end: date
