import datetime as dt
import pandas as pd
from fmiopendata.wfs import download_stored_query
import os

def to_wfs_time(t: dt.datetime) -> dt.datetime:
    t_utc = t.astimezone(dt.timezone.utc)
    return t_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch_weather_data(days: int=10, station_id: int=100974) -> pd.DataFrame:
    end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time-dt.timedelta(days=days)
    step = dt.timedelta(days=7) # Maximum step size for fetching fmi data

    df = pd.DataFrame()
    current = start_time
    chunk_idx = 0

    while current < end_time:
        chunk_idx += 1
        chunk_start = current
        chunk_end = min(current + step, end_time)

        start_s = to_wfs_time(chunk_start)
        end_s = to_wfs_time(chunk_end)

        print(f"Chunk {chunk_idx}: {start_s} -> {end_s}")

        args = [f"fmisid={station_id}", f"starttime={start_s}", f"endtime={end_s}", "timeseries=True"]
        obs = download_stored_query("fmi::observations::weather::multipointcoverage", args=args)

        station = obs.data[list(obs.data.keys())[0]]
        times = station["times"]
        temp_vals = station["Air temperature"]["values"]
        precip_vals = station["Precipitation intensity"]["values"]

        chunk_df = pd.DataFrame({'times': times, 'temp_vals': temp_vals, 'precip_vals': precip_vals}).set_index('times')
        df = pd.concat([df,chunk_df])
        current = chunk_end
    return df

def save_weather_data(days: int=10, id: int=100974):
    df = fetch_weather_data(days=days, station_id=id)
    data_dir = os.path.dirname(os.path.dirname(__file__))+'\\data'
    df.to_csv(data_dir + f"\\weather_obs_{days}_{id}.csv")
    print(f"Saved data into file '{data_dir}\weather_obs_{days}_{id}.csv'")
    print("Final time span:", df.index.min(), "->", df.index.max())

if __name__ == "__main__":
    save_weather_data(days=300)