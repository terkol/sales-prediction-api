import datetime as dt
import numpy as np
import pandas as pd
import os

from fmiopendata.wfs import download_stored_query

def to_wfs_time(t):
    t_utc = t.astimezone(dt.timezone.utc)
    return t_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch_forecast(hours_ahead=1, bbox="23.95, 59.95, 24.05, 60.05"):
    now = dt.datetime.now(dt.timezone.utc)
    start_str = to_wfs_time(now)
    end_str = to_wfs_time(now + dt.timedelta(hours=hours_ahead))

    a = [f"starttime={start_str}",f"endtime={end_str}",f"bbox={bbox}"]

    model_data = download_stored_query("fmi::forecast::harmonie::surface::grid", args=a)
    latest_run = max(model_data.data.keys())
    grid = model_data.data[latest_run]
    grid.parse()

    bbox = bbox.split(',')
    bbox = [float(i) for i in bbox]
    lat0, lon0 = (bbox[1]+bbox[3])/2, (bbox[0]+bbox[2])/2

    lats = grid.latitudes
    lons = grid.longitudes

    dist2 = (lats-lat0)**2+(lons-lon0)**2
    ir, ic = np.unravel_index(np.argmin(dist2), dist2.shape)
    print("Nearest grid point:", float(lats[ir, ic]), float(lons[ir, ic]))

    prediction_datetimes = sorted(grid.data.keys())
    
    times = []
    temps_C = []
    precs_mm = []
    for t in prediction_datetimes:
        levels = grid.data[t]
        temp_K = levels[2]["2 metre temperature"]["data"][ir, ic]
        temp_C = float(temp_K) - 273.15
        precip_val = levels[10]["surface precipitation amount, rain, convective"]["data"][ir, ic]

        times.append(t)
        temps_C.append(temp_C)
        precs_mm.append(precip_val)

    df = pd.DataFrame({"times": times, "temp_vals": temps_C, "precip_vals": precs_mm}).set_index("times")
    return df

def save_forecast(hours: int=1, box: str="24,60,24.5,60.5"):
    df = fetch_forecast(hours_ahead=hours, bbox=box)
    data_dir = os.path.dirname(os.path.dirname(__file__))+'\\data'
    df.to_csv(data_dir+f"\\weather_forecast_{hours}.csv")
    print(f"Saved data into file '{data_dir}\\weather_forecast_{hours}.csv'")

if __name__ == "__main__":
    save_forecast(hours=120)
