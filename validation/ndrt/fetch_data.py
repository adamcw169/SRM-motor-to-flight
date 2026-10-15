"""Download the NDRT 2020 validation data from the RocketPy repository (MIT licence).

Files (Ceotto et al. 2021, §Validation; flight data courtesy of the Notre Dame Rocket Team):
    Cesaroni_4895L1395-P.eng        certified static-test thrust curve of the Cesaroni L1395
    ndrt_2020_flight_data.csv       Raven altimeter record of the 23 Feb 2020 flight
    ndrt_2020_weather_data_ERA5.nc  ERA5 reanalysis around the launch site (NetCDF3)

Run:  python validation/ndrt/fetch_data.py
"""

import urllib.request
from pathlib import Path

COMMIT = "9bd6ad3af8f97bafa3201d4e70eba2877e66e040"  # RocketPy v1.13.0, pinned
BASE = f"https://raw.githubusercontent.com/RocketPy-Team/RocketPy/{COMMIT}/data"
FILES = {
    "Cesaroni_4895L1395-P.eng": "motors/cesaroni/Cesaroni_4895L1395-P.eng",
    "ndrt_2020_flight_data.csv": "rockets/NDRT_2020/ndrt_2020_flight_data.csv",
    "ndrt_2020_weather_data_ERA5.nc": "weather/ndrt_2020_weather_data_ERA5.nc",
}
DATA = Path(__file__).resolve().parent / "data"


def fetch(force: bool = False) -> Path:
    DATA.mkdir(exist_ok=True)
    for name, remote in FILES.items():
        dest = DATA / name
        if dest.exists() and not force:
            continue
        print(f"downloading {name}")
        urllib.request.urlretrieve(f"{BASE}/{remote}", dest)
    return DATA


if __name__ == "__main__":
    print("data in", fetch())
