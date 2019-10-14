import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


def xy(df):
    # separate target and data
    y = df['price']
    X = df.drop(columns=['price'])
    return X, y


def clean(df):
    # transform columns
    cut = {'Fair': 1, 'Good': 2, 'Ideal': 3, 'Premium': 4, 'Very Good': 5}
    for k, v in cut.items():
        df['cut'] = df['cut'].apply(lambda x: v if x == k else x)

    color = {'D': 1, 'E': 2, 'F': 3, 'G': 4, 'H': 5, 'I': 6, 'J': 7}
    for k, v in color.items():
        df['color'] = df['color'].apply(lambda x: v if x == k else x)

    clarity = {'I1': 1, 'SI2': 2, 'SI1': 3, 'VS2': 4,
               'VS1': 5, 'VVS2': 6, 'VVS1': 7, 'IF': 8}
    for k, v in clarity.items():
        df['clarity'] = df['clarity'].apply(lambda x: v if x == k else x)

    # drop columns
    df.drop('depth', axis=1, inplace=True)

    # standardize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df)
    df = pd.DataFrame(scaler.transform(df), columns=df.columns)

    # evaluate columns
    weights = {"carat": 0.45448746, "cut": 0.00331754, "color": 0.03367316, "clarity": 0.06369432,
               "table": 0.00348952, "x": 0.24129245, "y": 0.09447758, "z": 0.10556796}
    for k, v in weights.items():
        df[k] = df[k].apply(lambda x: x*v)

    return df


def apply_model(X, y, model, test):
    # train model with X and y and predict for test
    model.fit(X, y)
    pred = model.predict(test)
    return pred


def check_model(X_train, y_train, model, X_test, y_test):
    model_test = model
    model_test.fit(X_train, y_train)
    y_pred = model_test.predict(X_test)
    auc_test = r2_score(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred)
    return auc_test, mse_test
