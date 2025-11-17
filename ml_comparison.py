import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge

data_dir = os.path.dirname(os.path.dirname(__file__))+'\\data'
np.set_printoptions(suppress=True)

def load_training_data(file: str='training_data.csv'):
    df = pd.read_csv(data_dir + f'\\{file}')
    df = df.set_index('times')
    df.index = pd.to_datetime(df.index)
    return df

def train_model(model, df: pd.DataFrame, iter: int=10, show_progress=False):
    print(type(model))
    losses = pd.DataFrame(columns=['Daily training loss','Daily test loss','R^2 daily','Hourly training loss','Hourly test loss','R^2 hourly'])
    for i in range(iter):
        if show_progress:
            print(f'{i/iter*100}%')
        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='sales'), df['sales'])
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        eval_df_train = pd.DataFrame({'actual': y_train.values, 'pred': y_pred_train}, index=y_train.index)
        daily_eval_train = eval_df_train.groupby(eval_df_train.index.date).sum()
        eval_df = pd.DataFrame({'actual': y_test.values, 'pred': y_pred}, index=y_test.index)
        daily_eval = eval_df.groupby(eval_df.index.date).sum()
        mse_daily_train = mean_squared_error(daily_eval_train['actual'], daily_eval_train['pred'])
        mse_daily = mean_squared_error(daily_eval['actual'], daily_eval['pred'])
        r2_daily = r2_score(daily_eval['actual'], daily_eval['pred'])
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        losses.loc[i] = [mse_daily_train, mse_daily, r2_daily, mse_train, mse_test, r2]
    print(losses.mean())
    print()

df = load_training_data()

train_model(LinearRegression(), df)
train_model(Ridge(), df)
train_model(RandomForestRegressor(), df, show_progress=True)
train_model(XGBRegressor(), df)
train_model(CatBoostRegressor(verbose=False), df, show_progress=True)
train_model(HistGradientBoostingRegressor(), df)
train_model(DummyRegressor(), df)
