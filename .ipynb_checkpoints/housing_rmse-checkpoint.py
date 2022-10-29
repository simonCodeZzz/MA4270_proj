import numpy as np
import pandas as pd

from RFF import RBF_kernel, Laplacian_kernel, RFF
from time import time

from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

housing_X, housing_y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(housing_X, housing_y, test_size=0.4, random_state=42)

n_features = X_train.shape[1]
n_data = X_train.shape[0]
n_data_list = np.linspace(0, n_data, num=11, dtype=int)[1:]
gamma = 1/n_features

exact_ridge = Pipeline([
                      ('std',StandardScaler()),
                      ('ridge', KernelRidge(alpha=1, kernel='rbf', gamma = gamma)),
                      ])

ridge_times = []
ridge_rmse = []

for n in n_data_list:
    start = time()
    exact_ridge.fit(X_train[:n,],y_train[:n])
    t_exact_ridge = time() - start
    ridge_times.append(t_exact_ridge)
    
    y_pred = exact_ridge.predict(X_test)                      
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 
    ridge_rmse.append(rmse)
    
    print(f'No. of data: {n} | Testing RMSE of exact Ridge = {rmse:.5f} | training time = {t_exact_ridge:.3f}')
    
np.save('data/exact_rmse_housing', np.array(ridge_rmse))
np.save('data/exact_time_housing', np.array(ridge_times))

# np.save('data/error_time_housing', np.array([rmse, t_exact_ridge]))
# print(f'Testing RMSE of exact Ridge = {rmse:.5f} | training time = {t_exact_ridge:.3f}')

Ds = np.load('data/Ds_housing.npy')
num_Ds = len(Ds)
num_data = len(n_data_list)
rff_times, rff_rmse = np.zeros((num_Ds,num_data)), np.zeros((num_Ds,num_data))



for i,D in enumerate(Ds):
    rff_pip = Pipeline([
                ('std',StandardScaler()),
                ('rff',RFF(gamma=gamma, metric = 'rbf', D=D)),
                ('ridge', Ridge(alpha=1)),
             ])

    for j,n in enumerate(n_data_list):
        start = time()
        rff_pip.fit(X_train[:n,],y_train[:n])
        t_rff = time() - start
        rff_times[i,j] = t_rff

        y_pred_rff = rff_pip.predict(X_test)                      
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_rff)) 
        rff_rmse[i,j] = rmse
    print(f"Iteration {i} done: Approxiamtion with {D} features, RFF takes about {t_rff:.3f} sec")

np.save('data/rff_rmse_housing', rff_rmse)
np.save('data/rff_times_housing', rff_times)

best_rmse = rff_rmse.min()
best_time = rff_times.flatten()[np.argmin(rff_rmse)]

print(f"Best RFF performance: Testing RMSE = {best_rmse:.5f} | Training time = {best_time:.3f}")