import random
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.cluster import k_means
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge

class Scllc:
    def __init__(self, n_clusters, n_landmarks = 1000, n_neighbors = 5, func_landmark = 'kmeans', lambda_val = 100, delta = 0.001):
        self.n_clusters = n_clusters
        self.n_landmarks = n_landmarks
        self.n_neighbors = n_neighbors
        self.func_landmark = func_landmark
        self.lambda_val = lambda_val
        self.delta = delta;
        
    def __locality_linear_coding(self, data, neighbors):
        indicator = np.ones([neighbors.shape[0], 1])
        penalty = np.eye(self.n_neighbors)

        # Get the weights of every neighbors
        z = neighbors - indicator.dot(data.reshape(-1,1).T)
        local_variance = z.dot(z.T)
        local_variance = local_variance + self.lambda_val * penalty
        weights = scipy.linalg.solve(local_variance, indicator)
      
        weights = weights / np.sum(weights)
        weights = weights / np.sum(np.abs(weights))
        weights = np.abs(weights)

        return weights.reshape(self.n_neighbors)
    
    def fit(self, X):
        [n_data, n_dim] = X.shape
        # Select landmarks
        if self.func_landmark == 'kmeans':
            landmarks, centers, unknown = k_means(X, self.n_landmarks, n_init=1, max_iter=100, precompute_distances=True)
        elif self.func_landmark == 'random':
            random_ind = random.sample(list(range(1, n_data)), self.n_landmarks)
            landmarks = X[random_ind, :]
        else:
            raise ValueError(self.func_landmark, 'is invalid for selecing landmarks')
        nbrs = NearestNeighbors(metric='euclidean').fit(landmarks)
        
        # Create properties of the sparse matrix Z
        [dist, indy] = nbrs.kneighbors(X, n_neighbors = self.n_neighbors)
        indx = np.ones([n_data, self.n_neighbors]) * np.asarray(range(n_data))[:, None]
        valx = np.zeros([n_data, self.n_neighbors])
        self.delta = np.mean(valx)
        
        # Compute all the coded data 
        for index in range(n_data):
            # Compute the weights of its neighbors
            localmarks = landmarks[indy[index,:], :]
            weights = self.__locality_linear_coding(X[index,:], localmarks)
            # Compute the coded data
            valx[index] = weights
        
        # Construct sparse matrix 
        indx = indx.reshape(n_data * self.n_neighbors)
        indy = indy.reshape(n_data * self.n_neighbors)
        valx = valx.reshape(n_data * self.n_neighbors)

        Z = sparse.coo_matrix((valx,(indx,indy)),shape=(n_data,self.n_landmarks)) 
        Z = Z / np.sqrt(np.sum(Z, 0))

        # Get first k eigenvectors
        [U, Sigma, V] = svds(Z, k = self.n_clusters + 1)
        U = U[:, 0:self.n_clusters]
        embedded_data = U / np.sqrt(np.sum(U * U, 0))     
        
        # Run k-means and get results
        centers, labels, unknown = k_means(embedded_data, self.n_clusters, n_init=1, max_iter=100, precompute_distances=True)
        
        return labels.astype(int)
