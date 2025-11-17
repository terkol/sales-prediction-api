import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import joblib

data_dir = os.path.dirname(os.path.dirname(__file__))+'\\data'

def load_training_data(file: str='training_data.csv') -> pd.DataFrame:
    df = pd.read_csv(data_dir + f'\\{file}')
    df = df.set_index('times')
    df.index = pd.to_datetime(df.index)
    return df

def save_linear_regression_model(df: pd.DataFrame):
    X = df.drop(columns='sales')
    y = df['sales']
    model = LinearRegression().fit(X, y)
    coeffs = pd.DataFrame(index=X.columns)
    coeffs['coeffs'] = model.coef_
    coeffs.to_csv(data_dir + "\\sales_model_coeffs.csv")
    joblib.dump(model, data_dir + "\\sales_model.joblib")
    print(f"Saved model into file '{data_dir}\\sales_model.joblib'")

if __name__ == "__main__":
    df = load_training_data()
    save_linear_regression_model(df)