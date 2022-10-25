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

fashion_train = pd.read_csv("data/fashion-mnist_train.csv")
X_train = fashion_train.iloc[:,1:].values
y_train = fashion_train.iloc[:,0].values

fashion_test = pd.read_csv("data/fashion-mnist_test.csv")
X_test = fashion_test.iloc[:,1:].values
y_test = fashion_test.iloc[:,0].values

n_features = X_train.shape[1]
gamma = 1/n_features
sample_size = 10000
exact_svm = Pipeline([
                      ('std',StandardScaler()),
                      ('svc', OneVsRestClassifier(SVC(C=5, kernel="rbf", gamma=gamma))),
                      ])

start = time()
exact_svm.fit(X_train[:sample_size,:],y_train[:sample_size])
t_exact_svm = time() - start

y_pred = exact_svm.predict(X_test)                      
error = 1 - accuracy_score(y_test, y_pred)                    
np.save('data/error_time_fashion', np.array([error, t_exact_svm]))
print(f'Testing error exact SVM = {error:.5f} | training time ={t_exact_svm:.3f}')

Ds = np.load("data/Ds_fashion.npy")
num_Ds = len(Ds)
num_repeats = 10
rff_times, rff_errors = np.zeros((num_Ds,num_repeats)), np.zeros((num_Ds,num_repeats))

for i,D in enumerate(Ds):
    rff_pip = Pipeline([
                ('std',StandardScaler()),
                ('rff',RFF(gamma=gamma, metric = 'rbf', D=D)),
                ('linsvc', OneVsRestClassifier(LinearSVC(C=5))),
             ])

    for n in range(num_repeats):
        start = time()
        rff_pip.fit(X_train[:sample_size,:],y_train[:sample_size])
        t_rff = time() - start
        rff_times[i,n] = t_rff

        y_pred_rff = rff_pip.predict(X_test)                      
        error_rff = 1 - accuracy_score(y_test, y_pred_rff) 
        rff_errors[i,n] = error_rff
    print(f"Iteration {i} done: Approxiamtion with {D} features, RFF takes about {t_rff:.3f} sec per run")

np.save('data/rff_errors_fashion', rff_errors)
np.save('data/rff_times_fashion', rff_times)

best_error = rff_errors.min()
best_time = rff_times.flatten()[np.argmin(rff_errors)]

print(f"Best RFF performance: Testing error = {best_error:.5f}| Training time = {best_time:.3f}")