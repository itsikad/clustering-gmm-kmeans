import numpy as np


class KMeans:

    """
    K-Means clustering using Lloyd's algorithm.
    """

    def __init__(
        self,
        num_clusters
    ) -> None:

        """
        Arguments:
            num_clusters: number of clusters
        """

        self.num_clusters = num_clusters

    def predict(
        self,
        x: np.ndarray
    ) -> np.ndarray:

        """
        Predicts the class for each sample in x_data

        Arguments:
            x: (N,D) array, where N is the number of samples and D is the number of features

        Return:
            assignments: (N,) array, the i-th entry corredponds to
                         the predicted class for the i-th input sample
        """

        distances = np.linalg.norm(np.expand_dims(x, axis=1) - np.expand_dims(self.centroids, axis=0), axis=2)
        assignments = np.argmin(distances, axis=1)

        return assignments

    def fit(
        self,
        x_train: np.ndarray,
        max_iter: int = 1000,
        tolerance: float = 1e-6,
    ) -> None:

        """
        This method trains a K-Means classifier using Lloyd's algorithm.

        Arguments:
            x_train: (N,D) array, where N is the number of samples and D is the number of features

            max_iter: int, maximum number of iteration

            tolerance: float, stopping criteria, difference between centroids updates

        Return:
            predictions: cluster assignment of the training samples

            centroids: (K,D) array, the i-th row corresponds to the prototype of the j-th cluster
        """

        N, D = x_train.shape

        # Initialize centroids to K random samples from dataset
        idx = np.random.choice(np.arange(N), size=self.num_clusters, replace=False)
        self.centroids = x_train[idx, :]

        centroids_diff = np.inf

        for iter_idx in range(max_iter):
            
            # update assignments
            assign_mat = self.update_assignment_matrix(x_train)

            # update centroids
            prev_centroids = self.centroids
            self.centroids = self.update_centroids(x_train, assign_mat)

            # stopping criteria
            centroids_diff = np.sum(np.linalg.norm(self.centroids - prev_centroids, axis=1))

            if centroids_diff < tolerance:
                break

    def update_assignment_matrix(
        self,
        x_train: np.ndarray
    ) -> np.ndarray:

        """
        Assignments matrix update step.

        Arguments:
            x_train: (N,D) array, where N is the number of samples and D is the number of features

        Return:
            assign_mat: (N,K) array, assignment matrix, the i,j element
                         is set to 1 if the i-th sample belongs to the j-th cluster, and 0 otherwise
        """

        N = x_train.shape[0]
        assign_mat = np.zeros((N, self.num_clusters))
        assignments = self.predict(x_train)
        assign_mat[np.arange(N), assignments] = 1

        return assign_mat

    def update_centroids(
        self,
        x_train: np.ndarray,
        assign_mat: np.ndarray
    ) -> np.ndarray:

        """
        Centroids update step.

        Arguments:
            x_train: (N,D) array, where N is the number of samples and D is the number of features

            assign_mat: (N,K) array, assignment matrix, the i,j element
                         is set to 1 if the i-th sample belongs to the j-th cluster, and 0 otherwise
        Return:
            centroids: (K,D) array, the i-th row correponds to the centroids of the i-th cluster
        """

        N, D = x_train.shape
        weights = assign_mat / np.sum(assign_mat, axis=0)  # (N,K)
        centroids = np.sum(np.expand_dims(x_train, axis=1) * np.expand_dims(weights, axis=-1), axis=0)  # (N,1,D) * (N,K,1) -> (N,K,D) -> (K,D)

        return centroids


