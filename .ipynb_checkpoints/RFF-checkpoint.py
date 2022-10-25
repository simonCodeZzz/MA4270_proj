from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from scipy.stats import cauchy, laplace
import numpy as np

class RFF(BaseEstimator):
    def __init__(self, gamma = 1, D = 50, metric = "rbf"):
        self.gamma = gamma
        self.metric = metric
        #Dimensionality D (number of MonteCarlo samples)
        self.D = D
        self.fitted = False
        
    def fit(self, X, y=None):
        """ Generates MonteCarlo random samples """
        d = X.shape[1]
        #Generate D iid samples from p(w) 
        if self.metric == "rbf":
            self.w = np.sqrt(2*self.gamma)*np.random.normal(size=(self.D,d))
        elif self.metric == "laplace":
            self.w = cauchy.rvs(scale = self.gamma, size=(self.D,d))
        
        #Generate D iid samples from Uniform(0,2*pi) 
        self.u = 2*np.pi*np.random.rand(self.D)
        self.fitted = True
        return self
    
    def transform(self,X):
        """ Transforms the data X (n_samples, n_features) to the new map space Z(X) (n_samples, n_components)"""
        if not self.fitted:
            raise NotFittedError("RBF_MonteCarlo must be fitted beform computing the feature map Z")
        #Compute feature map Z(x):
        Z = np.sqrt(2/self.D)*np.cos((X.dot(self.w.T) + self.u[np.newaxis,:]))
        return Z
    
    def compute_kernel(self, X):
        """ Computes the approximated kernel matrix K """
        if not self.fitted:
            raise NotFittedError("RBF_MonteCarlo must be fitted beform computing the kernel matrix")
        Z = self.transform(X)
        K = Z.dot(Z.T)
        return K

def RBF_kernel(X, gamma=1/2):
    """
     K_RBF(X, X') = exp(-gamma ||X - X'||^2)
    """
    K = euclidean_distances(X, X, squared=True) # help from sklearn to get pairwise euclidean distances
    K = -gamma * K
    K = np.exp(K, K)  
    return K

def Laplacian_kernel(X, gamma=1/2):
    """
     K_laplacian(X, X') = exp(-gamma ||X-X'||_1)
    """
    K = manhattan_distances(X, X) # help from sklearn to get pairwise manhattan distances
    K = -gamma * K
    K = np.exp(K, K)  
    return K
