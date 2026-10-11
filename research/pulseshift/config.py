"""Sources, paths, and analysis constants for the Washington DC study."""

from pathlib import Path

YEARS = (2022, 2023, 2024)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
APP = ROOT.parent / "app"
RESULTS = ROOT.parent / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"

BIKESHARE_URL = (
    "https://s3.amazonaws.com/capitalbikeshare-data/{ym}-capitalbikeshare-tripdata.zip"
)

LCD_STATION = "72405013743"
LCD_URL = "https://www.ncei.noaa.gov/data/local-climatological-data/access/{year}/{station}.csv"

EPA_DAILY_AQI_URL = "https://aqs.epa.gov/aqsweb/airdata/daily_aqi_by_county_{year}.zip"

DC_LAT, DC_LON = 38.8951, -77.0364
OPENMETEO_AQI_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    "?latitude={lat}&longitude={lon}&hourly=us_aqi,pm2_5"
    "&start_date={year}-01-01&end_date={year}-12-31&timezone=GMT&timeformat=unixtime"
)

LOCAL_TZ = "America/New_York"

EXPECTED_FLOOR = 20
SUPPRESSION_RATIO = 0.5
SENSITIVITY_RATIOS = (0.4, 0.5, 0.6)

COLD_STRESS_BASE_F = 55.0
HEAT_STRESS_BASE_F = 85.0

TRAIN_YEARS = (2022, 2023)
TEST_YEAR = 2024

MDE_Z80 = 2.802
MDE_Z90 = 3.242

SHIFT_WINDOW_H = 3
HEAT_UNSAFE_F = 103.0
AQI_UNSAFE = 150
MIN_RISK_BENEFIT = 0.05
MEAN_RIDE_MIN = 13.0


def ym_list() -> list[str]:
    return [f"{year}{month:02d}" for year in YEARS for month in range(1, 13)]
