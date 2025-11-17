from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import pandas as pd
import os

def fetch_normalized_trends(word: str="pizza", tf: str="2025-01-01 2025-12-31") -> pd.DataFrame:
    pytrends = TrendReq(hl="en-US", tz=0)
    pytrends.build_payload(kw_list=[word], timeframe=tf)
    df = pytrends.interest_over_time().reset_index()
    return df

def save_google_trends(w: str='pizza'):
    data_dir = os.path.dirname(os.path.dirname(__file__))+'\\data'
    df = fetch_normalized_trends(word=w)
    df['trend'] = df[w]
    df['times'] = df['date']
    df = df.drop(columns=['isPartial', w, 'date'])
    df = df.set_index('times')
    df.to_csv(data_dir + "\\google_trends.csv")
    print(f"Saved data into file '{data_dir}\\google_trends.csv'")

if __name__ == "__main__":
    save_google_trends('pizza')
