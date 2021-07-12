from typing import Tuple

import numpy as np

from .kmeans import KMeans


class GMM:
    """
    Expectation Maximizatoin implementation for finding a Gaussian Mixture Model.
    """

    def __init__(
        self,
        num_components: int
    ) -> None:

        """
        Arguments:
            num_components: Number of Gaussian components
        """

        self.num_components = num_components

    def predict(
        self,
        x: np.ndarray
    ) -> Tuple:

        """"
        Calculates the probabilities of every sample in x_data to be assigned to the k-th class

        Arguments:
            x: (N,D) array where N is the number of samples and D the number of features

        Return:
            predictions: (N,) array, where N is the number of samples,
                         the i-th element corresponds to the class assigned to the i-th sample

            responsibilities: (N,K) array where N is the number of samples and K is the number of features,
                              the (i,j)-th entry corresponds to the porbability of the i-th sample to be drawn from the j-th component
        """

        weighted_densities = self.weights * self.compute_gaussian_densities(x)  # (N,K)
        reponsibilities = weighted_densities / np.sum(weighted_densities, axis=1).reshape(-1,1)
        predictions = np.argmax(reponsibilities, axis=1)

        return predictions, reponsibilities

    def fit(
        self,
        x_train: np.ndarray,
        max_iter: int = 1000,
        tolerance: float = 1e-3
    ) -> None:

        """
        Executes the Expectation Maximization algorithm to find the mean vectors,
        covariance matrices and weights of the GMM.

        Arguments:
            x_train: (N,D) array, where N is the number of samples and D the number of features

            max_iter: An integer, maximum number of iterations

            tolerance: float, convergence criteria for the likelihood and paramters
        """

        # Initializations
        log_likelihood_prev = -np.inf
        self.num_samples = x_train.shape[0]

        self.responsibilities = self.init_rspnsblts(x_train)

        # EM iterations
        for iter_idx in range(max_iter):

            # M-Step
            self.weights = self.update_weights()
            self.expectations = self.update_expectations(x_train)
            self.covariances = self.update_covariances(x_train)

            # E-Step
            self.responsibilities = self.update_rspnsblts(x_train)

            # Log-likelihood evaluation
            log_likelihood = self.evaluate_log_likelihood(x_train)

            # Check convergence of log likelihood
            if np.linalg.norm(log_likelihood - log_likelihood_prev) < tolerance:
                break

            log_likelihood_prev = log_likelihood

    def init_rspnsblts(
        self,
        x_train: np.ndarray
    ) -> np.ndarray:

        """
        Initializes the responsibilities matrix using K-Means algorithm (hard assignment).

        Arguments:
            x_train: (N,D) array, where N is the number of samples and D the number of features

        Return:
            responsibilities: (N,K) array, represents the responsibilities matrix, i.e. the (n,k)-th entry is
                              1 if the n-th sample is assigned to the k-th component and 0 otherwise.
        """

        kmeans_classifier = KMeans(num_clusters=self.num_components)
        kmeans_classifier.fit(x_train, max_iter=1000, tolerance=1e-3)  # relaxed tolerance
        assignments = kmeans_classifier.predict(x_train)

        responsibilities = np.zeros((self.num_samples, self.num_components))
        responsibilities[np.arange(self.num_samples), assignments] = 1

        return responsibilities

    def update_rspnsblts(
        self,
        x_train: np.ndarray
    ) -> np.ndarray:

        """
        Updates the responsibilities matrix.

        Arguments:
            x_train: (N,D) array, where N is the number of samples and D the number of features

        Return:
            responsibilities: (N,K) array, represents the responsibilities matrix, i.e. the (n,k)-th entry is
                              the probability of the n-th sample to be assigned to the k-th class
        """

        weighted_densities = self.weights * self.compute_gaussian_densities(x_train)

        return weighted_densities / np.sum(weighted_densities, axis=1).reshape(-1,1)

    def update_expectations(
        self,
        x_train: np.ndarray
    ) -> np.ndarray:

        """
        Updates the mean vectors.

        Arguments:
            x_train: (N,D) array, where N is the number of samples and D the number of features

        Return:
            expectations: (K,D) array, weighted mean of samples per class
        """

        weighted_samples = np.expand_dims(self.responsibilities, axis=-1) * np.expand_dims(x_train, axis=1)

        return np.sum(weighted_samples, axis=0) / np.sum(self.responsibilities, axis=0).reshape(-1,1)

    def update_weights(
        self
    ) -> np.ndarray:

        """
        Updates the mixture's weights.

        Return:
            weights: (K,) array, the k-th entry corresponds to the weight of the k-th gaussian
        """

        return np.sum(self.responsibilities, axis=0) / self.num_samples

    def update_covariances(
        self,
        x_train: np.ndarray
    ) -> np.ndarray:

        """
        Updates the covariance matrices.
        
        Arguments:
            x_train: (N,D) array, where N is the number of samples and D the number of features

        Return:
            covariances: (K,D) array, the (k,d) element is the d-th diagonal element
                          of the k-th covariance matrix
        """

        # Calculate the squared distances between each sample and mean_vector pair
        pairwise_dist = np.expand_dims(x_train, axis=1) - np.expand_dims(self.expectations, axis=0)  # (N,K,D)
        pairwise_cov = np.matmul(np.expand_dims(pairwise_dist, axis=-1), np.expand_dims(pairwise_dist, axis=2))  # (N,K,D,D)

        # Element wise multiplication with the extended responsibilities matrix and
        # sum over the samples dimensions to get the desired (K,D,D) array of covariance matrices
        # Divide the k-th row by Nk (the number of samples in class k)
        weights = self.responsibilities / self.responsibilities.sum(axis=0)
        covariances = (np.expand_dims(weights, axis=(-1,-2)) * pairwise_cov).sum(axis=0)
        
        return covariances

    def evaluate_log_likelihood(
        self,
        x_train: np.ndarray
    ) -> np.ndarray:

        """
        Evaluates the log-likelihood function.

        Arguments:
            x_train: (N,D) array, where N is the number of samples and D the number of features

        Return:
            log_likelihood_eval: scalar
        """

        weighted_gaussians = np.dot(self.compute_gaussian_densities(x_train), self.weights)

        return np.sum(np.log(weighted_gaussians))

    def compute_gaussian_densities(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        """
        Computes the Gaussians density functions per each sample-class pair.

        Arguments:
            x: (N,D) array where N is the number of samples and D the number of features 

        Return:
            gaussian_densities: (N,K) array matrix,
                                The (n,k) entry is the Gaussian density function evaluated for the n-th sample
                                given it is drawn from the k-th class
        """

        # Squared distances for each sample-mena_vec pair
        pairwise_distances = np.expand_dims(x, axis=1) - np.expand_dims(self.expectations, axis=0)
        
        # Element wise multiplication with the extended covariance matrices
        # sum over the features dimension to get the desired (N,K) matrix

        quadratic = np.expand_dims(np.linalg.inv(self.covariances), axis=0) @ np.expand_dims(pairwise_distances,axis=-1)
        quadratic = np.sum(quadratic.squeeze() * pairwise_distances, axis=-1)
        cov_mat_det = np.power(np.linalg.det(2 * np.pi * self.covariances), -0.5)

        return cov_mat_det * np.exp(-0.5 * quadratic)
