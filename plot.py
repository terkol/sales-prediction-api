import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt

data_dir = os.path.dirname(os.path.dirname(__file__))+'\\data'

def load_training_data(file: str='training_data.csv') -> pd.DataFrame:
    df = pd.read_csv(data_dir + f'\\{file}')
    df = df.set_index('times')
    df.index = pd.to_datetime(df.index, utc=True)
    return df

plt.figure(figsize=(12,8))

history_df = load_training_data()
model = joblib.load(data_dir+"\\sales_model.joblib")
X = load_training_data()[-600:-1]
X_pred = X.copy()
y_pred = model.predict(X.drop(columns='sales'))
X_pred['sales'] = y_pred
daily = X.groupby(X.index.date).sum()[2:-1]
daily_pred = X_pred.groupby(X_pred.index.date).sum()[2:-1]
print((daily_pred['sales'] - daily['sales']).abs().mean())
plt.ylabel('Daily sales')
plt.xlabel('Date')
plt.bar(daily.index, daily['sales'], width=1, label='Real sales')
plt.plot(daily_pred.index, daily_pred['sales'], c='r', linewidth=5, label='Sales predicted by the model')
plt.legend()
plt.show()

