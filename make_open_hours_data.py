import pandas as pd
import os

if __name__ == "__main__":
    times = pd.date_range('2025-01-01', '2025-12-31', freq='h')
    df = pd.DataFrame({'times': times})
    df = df.set_index('times')
    df['hour'] = df.index.hour
    df['k_open'] = df['hour'].isin(list(range(7,23)))
    df['g_open'] = df['hour'].isin(list(range(10,22)))
    df = df.drop(columns='hour')
    data_dir = os.path.dirname(os.path.dirname(__file__))+'\\data'
    df.to_csv(data_dir+"\\toy_open_hours_data.csv")
    print(f"Saved data into file '{data_dir}\\toy_open_hours_data.csv'")