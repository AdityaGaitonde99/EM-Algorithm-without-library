#!/usr/bin/env python
# coding: utf-8

# In[66]:
import sys
import numpy as np
from scipy.stats import multivariate_normal 
from scipy.stats import mode
from sklearn.datasets import load_iris
iris = load_iris()
X=np.loadtxt(sys.argv[1], delimiter=",", usecols=range(4))
k = int(sys.argv[2])

class EM:
    def __init__(self, k, max_iter=5):
        self.k = k
        self.max_iter = int(max_iter) 

    def initialize(self, X):
        self.shape = X.shape 
        self.n, self.m = self.shape 
        self.phi = np.full(shape=self.k, fill_value=1/self.k) 
        self.weights = np.full(shape=self.shape, fill_value=1/self.k)
        random_row = np.random.randint(low=0, high=self.n, size=self.k)
        self.mu = [  X[row_index,:] for row_index in random_row ] 
        self.sigma = [ np.cov(X.T) for _ in range(self.k) ] 
        
#E Step
    def e_step(self, X):
        self.weights = self.predict_proba(X)
        self.phi = self.weights.mean(axis=0)

#M Step
    def m_step(self, X):
        for i in range(self.k):
            weight = self.weights[:, [i]]
            total_weight = weight.sum()
            self.mu[i] = (X * weight).sum(axis=0) / total_weight
            self.sigma[i] = np.cov(X.T,aweights=(weight/total_weight).flatten(), bias=True)

    def fit(self, X):
        self.initialize(X)
        for iteration in range(self.max_iter):
            permutation = np.array([mode(iris.target[em.predict(X) == i]).mode.item() for i in range(em.k)])
            permuted_prediction = permutation[em.predict(X)]
            self.e_step(X)
            self.m_step(X)
        print('\n')
        print('Final Mean: ',self.mu)
        print('\n')
        print('Final Covariance: ',self.sigma)
        print('\n')
        print('No of iterations: ',iteration)
        print('\n')
        print('Cluster membership:',self.weights)
        print('\n')
        print('Size: ',self.weights.shape)
        print('\nThe accuracy before iteration ',iteration+1,end="")
        print(': ',np.mean(iris.target == permuted_prediction))

    def predict_proba(self, X):
        likelihood = np.zeros( (self.n, self.k) ) 
        for i in range(self.k):
            distribution = multivariate_normal(mean=self.mu[i],cov=self.sigma[i])
            likelihood[:,i] = distribution.pdf(X) 

        numerator = likelihood * self.phi
        denominator = numerator.sum(axis=1)[:, np.newaxis]
        weights = numerator / denominator
        return weights
    
    def predict(self, X):
        weights = self.predict_proba(X)
        return np.argmax(weights, axis=1)
    
def jitter(x):
    return x + np.random.uniform(low=-0.05, high=0.05, size=x.shape)

# In[69]:


np.random.seed(42)
em = EM(k, max_iter=10)
em.fit(X)

