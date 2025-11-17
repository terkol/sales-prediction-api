from fastapi import FastAPI
import joblib
import numpy as np
import datetime as dt
import os
import pandas as pd
from pytrends.request import TrendReq
from fmiopendata.wfs import download_stored_query
from fastapi import Response

data_dir = os.path.dirname(os.path.dirname(__file__))+'\\data'

def load_training_data(file: str='training_data.csv') -> pd.DataFrame:
    df = pd.read_csv(data_dir + f'\\{file}')
    df = df.set_index('times')
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[['sales', 'temp_vals', 'precip_vals', 'trend']]
    return df

history_df = load_training_data()
model = joblib.load(data_dir+"\\sales_model.joblib")
app = FastAPI()

def to_wfs_time(t: dt.datetime) -> dt.datetime:
    t_utc = t.astimezone(dt.timezone.utc)
    return t_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

def fetch_forecast(hours_ahead: int=1, bbox: str="23.95, 59.95, 24.05, 60.05") -> pd.DataFrame:
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
    df.index = pd.to_datetime(df.index, utc=True)
    return df

def fetch_normalized_trends(word: str="pizza", tf: str="2025-01-01 2025-12-31") -> pd.DataFrame:
    pytrends = TrendReq(hl="en-US", tz=0)
    pytrends.build_payload(kw_list=[word], timeframe=tf)
    df = pytrends.interest_over_time().reset_index()
    return df

def build_features(now: dt.datetime, history_df: pd.DataFrame):
    df = pd.DataFrame({"times": pd.date_range(now, now+dt.timedelta(days=4), freq='h')}).set_index('times')
    forecast = fetch_forecast(hours_ahead=120)
    df = df.join(forecast).dropna()
    df = pd.concat([history_df, df])
    try: 
        trends = fetch_normalized_trends(word='pizza')
        trends['trend'] = trends['pizza']
        trends['times'] = pd.to_datetime(trends['date'], utc=True)
        trends = trends.drop(columns=['isPartial', 'pizza', 'date'])
        df['trend'] = trends['trend'].iloc[-1]
    except Exception as e:
        print("Could not fetch Google Trends, using fallback:", e)
        df['trend'] = history_df['trend'].iloc[-1]

    df['const'] = 0
    df["hour"] = df.index.hour
    df["week"] = df.index.isocalendar().week
    df["weekday"] = df.index.isocalendar().day
    df["month"] = df.index.month

    df['si_promo'] = df['week'].isin(list(range(41,52)))
    df['pr_promo'] = df['week'].isin(list(range(40,52)))
    df['ha_promo'] = df['week'] == 44
    df['k_open'] = df['hour'].isin(list(range(7,23)))
    df['g_open'] = df['hour'].isin(list(range(10,22)))
    df['hours_open'] = (df.index - history_df.index[0]).total_seconds()/3600

    df = pd.get_dummies(df, columns=["weekday", "month", "hour"], prefix=["wd", "m", "h"])
    df = df.drop(columns=['week'])

    for i in [1, 4, 8, 24, 168]:
        df[f"temp_roll{i}"] = df["temp_vals"].shift(1).rolling(i).mean()
        df[f"rain_roll{i}"] = df["precip_vals"].shift(1).rolling(i).mean()
        df[f"temp_lag{i}"] = df["temp_vals"].shift(i)
        df[f"rain_lag{i}"] = df["precip_vals"].shift(i)
    mask = df.index > dt.datetime.now(dt.timezone.utc)
    x = df.iloc[mask].drop(columns=["sales"]).dropna()
    return x

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/predict")
def predict_sales_text():
    t = dt.datetime.now(dt.timezone.utc)
    X = build_features(pd.Timestamp(t).round("h").to_pydatetime(), history_df)
    y_pred = model.predict(X)
    df = pd.DataFrame({"preds": y_pred}, index=X.index)
    df.index = df.index.tz_convert('Europe/Helsinki')
    daily = df.groupby(df.index.date).sum()[:-1]
    lines = [f"{d.isoformat()}: {round(float(p), 2)}"for d, p in zip(daily.index, daily["preds"].values)]
    text = "\n".join(lines)
    return Response(content=text, media_type="text/plain")

