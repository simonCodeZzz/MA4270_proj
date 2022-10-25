import numpy as np
import pandas as pd

from RFF import RBF_kernel, Laplacian_kernel, RFF
from time import time
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True)

gamma = 1/2
Ds = np.array([2**p for p in range(3,14)]) # different dimensions of data
print(f"Monte Carlo numbers: {Ds}")

start = time()
K_rbf_true = RBF_kernel(X, gamma=gamma)
rbf_time_true = time() - start

start = time()
K_laplace_true = Laplacian_kernel(X,gamma=gamma)
laplace_time_true = time() - start

rbf_mse, laplace_mse = [] , []
rbf_time, laplace_time = [rbf_time_true] , [laplace_time_true]
print(f"True RBF time: {rbf_time_true:.3f} sec; True Laplacian time: {laplace_time_true:.3f} sec")

for i,D in enumerate(Ds):
    loop_time = time()
    
    RFF_rbf = RFF(gamma=gamma, D=D, metric="rbf")
    start = time()
    RFF_rbf.fit(X)
    K_rbf = RFF_rbf.compute_kernel(X)
    rbf_time_ = time() - start
    
    rbf_mse.append(((K_rbf_true-K_rbf)**2).mean())
    rbf_time.append(rbf_time_)

    RFF_laplace = RFF(gamma=gamma, D=D, metric="laplace")
    start = time()
    RFF_laplace.fit(X)
    K_laplace = RFF_laplace.compute_kernel(X)
    laplace_time_ = time() - start
    
    laplace_mse.append(((K_laplace_true-K_laplace)**2).mean())
    laplace_time.append(laplace_time_)
    
    print(f"Iteration {i} done: Approxiamtion with {D} features, RBF takes {rbf_time_:.3f} sec, Laplacian takes {laplace_time_:.3f} sec")
    
np.save("data/Ds_housing.npy", Ds)
np.save("data/rbf_mse_housing.npy", rbf_mse)
np.save("data/laplace_mse_housing.npy", laplace_mse)
np.save("data/rbf_time_housing.npy", rbf_time)
np.save("data/laplace_time_housing.npy", laplace_time)
print("Successfully saved: housing")