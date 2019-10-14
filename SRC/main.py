import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

result = open("../output/results.txt", "w+")

# import
data = pd.read_csv('../input/data.csv')
test = pd.read_csv('../input/test.csv')

# create submission docs
submission = pd.DataFrame(test['id'])
test = clean(test)
test.drop('id', axis=1, inplace=True)

# group data and clean
X, y = xy(data)
X = clean(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=35)

# check feature importance with Random Forest
"""
print(feature_importance_RF(X, y))
"""

# check models
models = {'Linear_Regression': LinearRegression(), 'Random_Forest': RandomForestRegressor(
    min_samples_leaf=2, min_samples_split=8, n_estimators=106), "Decision_Tree": DecisionTreeRegressor(random_state=0), "SVR": SVR(kernel="linear"), "KNeighbors": KNeighborsRegressor(n_neighbors=10)}

# grid search Random Forest
"""
params = {"min_samples_leaf": list(range(2, 5)), "min_samples_split": list(
    range(6, 15)), "n_estimators": list(range(100, 200, 10))}
Random_Forest_0 = RandomForestRegressor(
    min_samples_leaf=2, min_samples_split=11, n_estimators=106)
print("grid_searching...")
clf = GridSearchCV(Random_Forest_0, params, cv=5)
print("fitting model...")
clf.fit(X_train, y_train)
print(sorted(clf.cv_results_.keys()))
"""

"""models = {}

for x in range(2, 15, 2):
    models['Random_Forest_{}'.format(x)] = RandomForestRegressor(
        min_samples_leaf=2, min_samples_split=11, n_estimators=31)
"""

for name, model in models.items():
    print("checking {}...".format(name))
    result.write("Performance metrics {}:\n".format(name))
    auc_test, mse_test = check_model(X_train, y_train, model, X_test, y_test)
    print("- r2 score: {}\n- MSE: {}\n\n".format(auc_test, mse_test))
    result.write("- r2 score: {}\n- MSE: {}\n\n".format(auc_test, mse_test))

# apply models
for name, model in models.items():
    print("applying {}...".format(name))
    model_final = model
    pred = apply_model(X, y, model_final, test)
    submission_model = submission
    submission_model["price"] = pred
    submission_model.to_csv(
        "../output/submission_{}.csv".format(name), index=False)

print("\nall submissions created! check the 'output' folder")

result.close()
