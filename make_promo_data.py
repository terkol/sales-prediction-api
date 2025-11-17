import pandas as pd
import os

if __name__ == "__main__":
    times = pd.date_range('2025-01-01', '2025-12-31', freq='h')
    df = pd.DataFrame({'times':times})
    df = df.set_index('times')
    df['week'] = df.index.isocalendar().week
    df['si_promo'] = df['week'].isin(list(range(41,52)))
    df['pr_promo'] = df['week'].isin(list(range(40,52)))
    df['ha_promo'] = df['week'] == 44
    df = df.drop(columns=('week'))
    data_dir = os.path.dirname(os.path.dirname(__file__))+'\\data'
    df.to_csv(data_dir+"\\toy_promo_data.csv")
    print(f"Saved data into file '{data_dir}\\toy_promo_data.csv'")