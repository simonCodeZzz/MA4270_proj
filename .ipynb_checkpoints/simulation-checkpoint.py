import numpy as np
import pandas as pd
from time import time
from RFF import RFF, RBF_kernel, Laplacian_kernel

# Simulate Data
np.random.seed(4270)
n, p = 5000, 200
X = np.random.randn(n,p)

gamma = 1/2
Ds = np.arange(0,2000+1,100)[1:]#Number of monte carlo samples D
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
    
    print(f"Iteration {i} done: Approxiamtion with {D} features, RBF takes {rbf_time_:.3f} sec, Laplacian takes {laplace_time_:.3f} ")
    

rbf_mse, laplace_mse = np.array(rbf_mse), np.array(laplace_mse)
rbf_time, laplace_time = np.array(rbf_time), np.array(laplace_time)

np.save("data/Ds.npy", Ds)
np.save("data/rbf_mse.npy", rbf_mse)
np.save("data/laplace_mse.npy", laplace_mse)
np.save("data/rbf_time.npy", rbf_time)
np.save("data/laplace_time.npy", laplace_time)
print("Successfully saved")