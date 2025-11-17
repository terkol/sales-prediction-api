import pandas as pd
import os

data_dir = os.path.dirname(os.path.dirname(__file__))+'\\data'

def finnish_time_to_utc(s: pd.Series) -> pd.Series:    
    s = s.dt.tz_localize("Europe/Helsinki", ambiguous="NaT", nonexistent="shift_forward")
    s = s.dt.tz_convert('UTC')
    return s

def load_sales(string: str='sales', loc: str="") -> pd.DataFrame:
    files = os.listdir(data_dir)
    csvs = [pd.read_excel(data_dir + "\\" + f) for f in files if string in f]
    csvs_sorted = sorted(csvs, key=lambda d: pd.to_datetime(d[" Date "], format="%m/%d/%Y, %I:%M:%S %p").min())
    csv = pd.concat(csvs_sorted).fillna(-1).drop_duplicates()
    loc_mask = csv[' APD '].str.contains(loc.upper())
    csv = csv.loc[loc_mask]
    csv['times'] = pd.to_datetime(csv[' Date '], format="%m/%d/%Y, %I:%M:%S %p")
    csv['times'] = finnish_time_to_utc(csv['times'])
    csv = csv[['times', ' Price ']]

    # Split larger single sales into many smaller ones
    df = pd.DataFrame()
    for i in range(2,11):
        mask = csv[" Price "].gt((i)*10-5)
        add = csv.copy().loc[mask]
        add['times'] = add['times'] + pd.Timedelta(minutes=(i-1)*3)
        df = pd.concat([df, add])
    sales = pd.DataFrame(pd.concat([df['times'], csv['times']]).sort_values())

    sales['sales'] = 1
    sales = sales.set_index('times')
    hourly = sales.resample('h').agg({'sales': 'sum'})
    return hourly


def load_weather(file: str="weather_data.csv") -> pd.DataFrame:
    df = pd.read_csv(data_dir + f"\\{file}")
    df['times'] = pd.to_datetime(df['times'], utc=True)
    df = df.set_index('times')
    hourly = df.resample('h').agg({'temp_vals': 'mean', 'precip_vals': 'mean'})
    return hourly

def load_trends(file: str='google_trends.csv') -> pd.DataFrame:
    df = pd.read_csv(data_dir + f"\\{file}")
    df = df.set_index('times')
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.resample('1h').ffill()
    return df

def load_construction(file: str='construction_data.csv') -> pd.DataFrame:
    df = pd.read_csv(data_dir + f"\\{file}")
    df['times'] = finnish_time_to_utc(pd.to_datetime(df['times']))
    df = df.set_index('times')
    df = df.dropna().astype(int)
    return df

def load_open_hours(file: str='open_hours_data.csv') -> pd.DataFrame:
    df = pd.read_csv(data_dir + f"\\{file}")
    df['times'] = finnish_time_to_utc(pd.to_datetime(df['times']))
    df = df.set_index('times')
    df = df.dropna().astype(int)
    return df

def load_promos(file: str='promo_data.csv') -> pd.DataFrame:
    df = pd.read_csv(data_dir + f"\\{file}")
    df = df.set_index('times')
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.astype(int)
    return df

def load_forecast(file: str='forecast_data.csv') -> pd.DataFrame:
    df = pd.read_csv(data_dir + f"\\{file}")
    df = df.set_index('times')
    df.index = pd.to_datetime(df.index, utc=True)
    return df

def load_data() -> pd.DataFrame:
    df_sales = load_sales('Details', 'JA')
    df_weather = load_weather(file='weather_obs_300_100974.csv')
    df_trends = load_trends(file='google_trends.csv')
    df_const = load_construction(file='toy_construction_data.csv')
    df_open = load_open_hours(file='toy_open_hours_data.csv')
    df_promos = load_promos(file='toy_promo_data.csv')
    df = df_sales.join([df_weather, df_trends, df_const, df_promos, df_open])
    df = df.dropna()
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:   
    start = df.index.min()
    df['hours_open'] = (df.index - start).total_seconds()/3600
    weekdays = pd.get_dummies(df.index.isocalendar().day).add_prefix('wd_').set_index(df.index)
    months = pd.get_dummies(df.index.month).add_prefix('m_').set_index(df.index)
    hours = pd.get_dummies(df.index.hour).add_prefix('h_').set_index(df.index) 
    df = df.join([weekdays, months, hours])
    for i in [1,4,8,24,24*7]:
        df[f'temp_roll{i}'] = df['temp_vals'].shift(1).rolling(window=i).mean()
        df[f'rain_roll{i}'] = df['precip_vals'].shift(1).rolling(window=i).mean()
        df[f'temp_lag{i}'] = df['temp_vals'].shift(i)
        df[f'rain_lag{i}'] = df['precip_vals'].shift(i)
    df = df.dropna()
    return df

if __name__ == "__main__":
    df = load_data()
    df = engineer_features(df)
    df.to_csv(data_dir + '\\training_data.csv')
    print(f"Saved training data into file '{data_dir}\\training_data.csv'")