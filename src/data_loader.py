"""
data_loader.py
==============
Loads the real data used by the project:

1. Transportation Safety Board of Canada (TSB) Rail Occurrence Database
   (RODS), "Rail occurrence data from January 1983 to present", published on
   open.canada.ca (dataset 4f8d9b0d-c4af-4be2-b751-3f5a403993ff).
2. Open-Meteo historical weather (ERA5 archive, free, no key).

The TSB server blocks scripted downloads, so the CSVs are downloaded once in a
browser into data/raw/tsb/ (see README). Nothing is ever synthesised: if the
files are missing, loading fails with instructions.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DIR: Path = ROOT / "data" / "raw"
TSB_DIR: Path = RAW_DIR / "tsb"
TSB_OCCURRENCE_FILE: str = "RODSdb_MDOTW_VW_OCCURRENCE_PUBLIC.csv"
TSB_TRAIN_FILE: str = "RODSdb_MDOTW_VW_TRAIN_PUBLIC.csv"
TSB_DATASET_URL: str = (
    "https://open.canada.ca/data/en/dataset/4f8d9b0d-c4af-4be2-b751-3f5a403993ff"
)
WEATHER_CACHE: Path = RAW_DIR / "weather_daily_by_province.parquet"
OPEN_METEO_URL: str = "https://archive-api.open-meteo.com/v1/archive"

# TSB spells provinces in upper case (and without "and Labrador").
TSB_PROVINCE_MAP: dict[str, str] = {
    "ALBERTA": "Alberta",
    "BRITISH COLUMBIA": "British Columbia",
    "MANITOBA": "Manitoba",
    "NEW BRUNSWICK": "New Brunswick",
    "NEWFOUNDLAND": "Newfoundland and Labrador",
    "NEWFOUNDLAND AND LABRADOR": "Newfoundland and Labrador",
    "NORTHWEST TERRITORIES": "Northwest Territories",
    "NOVA SCOTIA": "Nova Scotia",
    "NUNAVUT": "Nunavut",
    "ONTARIO": "Ontario",
    "PRINCE EDWARD ISLAND": "Prince Edward Island",
    "QUÉBEC": "Quebec",
    "QUEBEC": "Quebec",
    "SASKATCHEWAN": "Saskatchewan",
    "YUKON": "Yukon",
}

# Columns kept from the occurrence table (pre-incident context + outcomes).
OCCURRENCE_COLUMNS: tuple[str, ...] = (
    "OccNo", "OccDate", "OccYear", "OccTime", "OccurrenceTypeID_DisplayEng",
    "AccIncTypeID_DisplayEng", "ProvinceID_DisplayEng", "Latitude", "Longitude",
    "SubdMileage", "TrackTypeID_DisplayEng", "ActivityTypeID_DisplayEng",
    "TotalFatalInjuries", "TotalSeriousInjuries", "TotalMinorInjuries",
)
TRAIN_COLUMNS: tuple[str, ...] = (
    "OccNo", "TrainSeq", "TrainTypeID_DisplayEng", "ApproxTrainSpeed_MPH",
    "NumCars", "NumLocomotives", "TrainTonnage",
)

WEATHER_RETRIES: int = 5
WEATHER_TIMEOUT_S: int = 120
WEATHER_PAUSE_S: float = 2.0
RATE_LIMIT_WAIT_S: int = 65
HTTP_TOO_MANY_REQUESTS: int = 429
WEATHER_PART_DIR: Path = RAW_DIR / "weather_parts"


def _read_tsb_csv(name: str, usecols: tuple[str, ...]) -> pd.DataFrame:
    """Read one TSB CSV, stripping the (doubled) byte-order mark in its header."""
    path = TSB_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download '{name}' from {TSB_DATASET_URL} "
            f"in a browser and save it to {TSB_DIR}/ (the TSB server rejects scripts)."
        )
    header = pd.read_csv(path, nrows=0, encoding="utf-8").columns
    clean = {c: c.replace("﻿", "").strip() for c in header}
    wanted = [raw for raw, c in clean.items() if c in usecols]
    df = pd.read_csv(path, usecols=wanted, encoding="utf-8", low_memory=False)
    return df.rename(columns=clean)


def _ddmmss_to_degrees(value: pd.Series) -> pd.Series:
    """TSB stores coordinates as DDMMSS / DDDMMSS integers (e.g. 494700)."""
    v = pd.to_numeric(value, errors="coerce")
    return (v // 10_000) + ((v // 100) % 100) / 60 + (v % 100) / 3600


def load_tsb_occurrences() -> pd.DataFrame:
    """
    Load TSB occurrences (one row per occurrence) joined to their lead train.

    Returns
    -------
    pd.DataFrame
        Raw TSB fields plus ``province`` (canonical name), ``date``,
        ``latitude`` and ``longitude`` (decimal degrees, west negative).
    """
    occ = _read_tsb_csv(TSB_OCCURRENCE_FILE, OCCURRENCE_COLUMNS).drop_duplicates("OccNo")
    trains = _read_tsb_csv(TSB_TRAIN_FILE, TRAIN_COLUMNS)
    lead = trains[trains["TrainSeq"] == 1].drop_duplicates("OccNo").drop(columns="TrainSeq")

    df = occ.merge(lead, on="OccNo", how="left")
    province_raw = df["ProvinceID_DisplayEng"].astype("string").str.strip().str.upper()
    return df.assign(
        province=province_raw.map(TSB_PROVINCE_MAP),
        date=pd.to_datetime(df["OccDate"].astype("string").str[:10], errors="coerce"),
        latitude=_ddmmss_to_degrees(df["Latitude"]),
        longitude=-_ddmmss_to_degrees(df["Longitude"]),
        _source_file=f"TSB RODS ({TSB_OCCURRENCE_FILE}, {TSB_TRAIN_FILE})",
    )


def load_tsb_accidents(start_year: int, end_year: int) -> pd.DataFrame:
    """TSB *accidents* (not reportable incidents) in [start_year, end_year]."""
    df = load_tsb_occurrences()
    keep = (
        (df["OccurrenceTypeID_DisplayEng"] == "ACCIDENT")
        & df["OccYear"].between(start_year, end_year)
        & df["province"].notna()
        & df["date"].notna()
    )
    out = df[keep].reset_index(drop=True)
    logger.info("TSB accidents %d-%d: %d rows", start_year, end_year, len(out))
    return out


def province_weather_points(accidents: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """One weather point per province: the median location of its accidents."""
    located = accidents.dropna(subset=["latitude", "longitude"])
    medians = located.groupby("province")[["latitude", "longitude"]].median().round(3)
    return {p: (float(r.latitude), float(r.longitude)) for p, r in medians.iterrows()}


def _fetch_daily_weather(province: str, lat: float, lon: float,
                         start_year: int, end_year: int) -> pd.DataFrame:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{start_year}-01-01", "end_date": f"{end_year}-12-31",
        "daily": "temperature_2m_mean,precipitation_sum,wind_speed_10m_max",
        "timezone": "America/Toronto",
    }
    last_error: Exception | None = None
    for attempt in range(WEATHER_RETRIES):
        try:
            r = requests.get(OPEN_METEO_URL, params=params, timeout=WEATHER_TIMEOUT_S)
            if r.status_code == HTTP_TOO_MANY_REQUESTS:
                # Long date ranges use a lot of the per-minute quota: wait it out.
                logger.warning("Open-Meteo rate limit for %s; waiting %ds", province, RATE_LIMIT_WAIT_S)
                time.sleep(RATE_LIMIT_WAIT_S)
                continue
            r.raise_for_status()
            daily = pd.DataFrame(r.json()["daily"])
            return pd.DataFrame({
                "province": province,
                "date": pd.to_datetime(daily["time"]),
                "wx_temp_c": daily["temperature_2m_mean"],
                "wx_precip_mm": daily["precipitation_sum"],
                "wx_wind_kmh": daily["wind_speed_10m_max"],
            })
        except (requests.RequestException, KeyError, ValueError) as exc:
            last_error = exc
            logger.warning("Open-Meteo attempt %d for %s failed: %s", attempt + 1, province, exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Open-Meteo weather unavailable for {province}") from last_error


def fetch_daily_weather(points: dict[str, tuple[float, float]], start_year: int,
                        end_year: int, cache: bool = True) -> pd.DataFrame:
    """
    Daily ERA5 weather (mean temperature, precipitation, max wind) per province.

    Raises RuntimeError rather than inventing weather when the API fails.
    """
    if cache and WEATHER_CACHE.exists():
        cached = pd.read_parquet(WEATHER_CACHE)
        covers = (set(points) <= set(cached["province"])
                  and cached["date"].dt.year.min() <= start_year
                  and cached["date"].dt.year.max() >= end_year)
        if covers:
            logger.info("Loading weather from cache: %s", WEATHER_CACHE)
            return cached
    WEATHER_PART_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for province, (lat, lon) in sorted(points.items()):
        part = WEATHER_PART_DIR / f"{province.replace(' ', '_')}_{start_year}_{end_year}.parquet"
        if cache and part.exists():
            frames.append(pd.read_parquet(part))
            continue
        logger.info("Fetching Open-Meteo daily weather for %s (%.2f, %.2f)", province, lat, lon)
        fetched = _fetch_daily_weather(province, lat, lon, start_year, end_year)
        fetched.to_parquet(part, index=False)
        frames.append(fetched)
        time.sleep(WEATHER_PAUSE_S)
    weather = pd.concat(frames, ignore_index=True).assign(_weather_source="open-meteo-era5")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(WEATHER_CACHE, index=False)
    return weather
