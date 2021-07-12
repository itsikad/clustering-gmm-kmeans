from typing import Tuple

import numpy as np

from models.kmeans import KMeans
from models.gmm import GMM


def gen_gaussian_mixture(
    num_samples: int,
    num_components: int,
    num_dims: int
    ) -> Tuple:

    """
    Generates a Gaussian mixture dataset.

    Arguments:
        num_samples: number of samples in dataset

        num_components: number of Gaussian componenets in mixture

        num_dims: features dimensions

    Return:
        dataset: (N,D) array, each row is a sample drawn from 1 out of K Gaussian models
        
        prob: (K,) array, prior probabilites

        mu: (K,D) array, each row correponds to the expectation vector of a component
        
        cov: (K,D,D) array, covariance matrices per componenet 
        
        model_sel: (N,) array, labels for the component from which a sample was drawn
    """

    # generate probabilities
    prob = np.random.uniform(low=0, high=1, size=num_components)
    prob = prob / prob.sum()

    # expectations and covariance matrices
    mu = np.random.randn(num_components, num_dims)
    cov = np.random.randn(num_components, num_dims, num_dims)  # not a valid covariance matrix (not symmetric, not PSD)
    cov = np.matmul(cov.transpose(0,2,1), cov)

    # create dataset
    dataset = np.zeros((num_samples, num_dims))
    model_sel = np.random.choice(num_components, size=num_samples, p=prob)

    # generate samples
    for k in range(num_components):
        samples = np.random.multivariate_normal(mean=mu[k], cov=cov[k], size=num_samples)
        dataset += samples * (model_sel == k).reshape(-1,1)

    return dataset, prob, mu, cov, model_sel

# Experiment parameters
N = 1000  # number of samples
K = 3  # number of componentes
D = 8  # features dimensions
MAX_ITERATIONS = 100
TOLERANCE = 1e-8

# Create dataset
dataset, prob, mu, cov, model_sel = gen_gaussian_mixture(N, K, D)

# Clsuter using KMeans 
k_means= KMeans(num_clusters=K)
k_means.fit(
    x_train=dataset,
    max_iter=MAX_ITERATIONS,
    tolerance=TOLERANCE
    )

kmeans_preds = k_means.predict(x=dataset)

# Estimate the GMM parameters using GMM with EM

gmm = GMM(num_components=K)
gmm.fit(
    x_train=dataset,
    max_iter=MAX_ITERATIONS,
    tolerance=TOLERANCE
    )

predictions, reponsibilities = gmm.predict(dataset)