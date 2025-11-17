from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import pandas as pd
import os

def fetch_normalized_trends(word: str="pizza", tf: str="2025-01-01 2025-12-31") -> pd.DataFrame:
    pytrends = TrendReq(hl="en-US", tz=0)
    pytrends.build_payload(kw_list=[word], timeframe=tf)
    df = pytrends.interest_over_time().reset_index()
    return df

if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.dirname(__file__))+'\\data'
    print(dir)
    df = fetch_normalized_trends(word='fizza')
    df['trend'] = df['fizza']
    df['times'] = df['date']
    df = df.drop(columns=['isPartial', 'fizza', 'date'])
    df = df.set_index('times')
    df.to_csv(data_dir + "\\google_trends.csv")
