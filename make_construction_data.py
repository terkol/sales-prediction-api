import pandas as pd
import os

if __name__ == "__main__":
    times = pd.date_range('2025-01-01', '2025-11-15', freq='h')
    df = pd.DataFrame({'times': times})
    df = df.set_index('times')
    df['day'] = df.index.isocalendar().day
    df['hour'] = df.index.hour
    df['month'] = df.index.month
    df['const'] = df['day'].isin([1,2,3,4]) & df['hour'].isin(list(range(9,18))) & df['month'].isin(list(range(8,12)))
    df = df.drop(columns=['day','hour','month'])
    data_dir = os.path.dirname(os.path.dirname(__file__))+'\\data'
    df.to_csv(data_dir+"\\toy_construction_data.csv")
    print(f"Saved data into file '{data_dir}\\toy_construction_data.csv'")