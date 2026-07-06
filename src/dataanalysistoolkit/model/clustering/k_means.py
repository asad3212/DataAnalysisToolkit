"""k_means.py

This module provides a class, KMeansClusterer, for performing K-Means
clustering on a dataset. It wraps scikit-learn's KMeans implementation and
adds convenience methods for choosing the number of clusters (elbow method)
and evaluating cluster quality (silhouette score).

KMeansClusterer is designed to be initialized with feature data. It can fit
a K-Means model, return cluster assignments and centroids, and help select
a good value of k before committing to a final model.

Returns:
    None: This module does not return anything but provides a class that can
    be instantiated for K-Means clustering.
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class KMeansClusterer:
    """
    A class for performing K-Means clustering on a dataset.

    This class wraps scikit-learn's KMeans estimator and provides methods to
    fit the model, retrieve cluster labels and centroids, and evaluate
    clustering quality via inertia and silhouette score.

    Attributes:
        n_clusters (int): The number of clusters to form.
        random_state (int): Seed used for reproducible results.
        model (KMeans): The underlying fitted scikit-learn KMeans estimator,
        or None before fit() is called.
    """

    def __init__(self, n_clusters=3, random_state=42):
        """
        Constructs all the necessary attributes for the KMeansClusterer object.

        Args:
            n_clusters (int): The number of clusters to form. Defaults to 3.
            random_state (int): Seed for reproducible cluster assignment.
            Defaults to 42.
        """
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1.")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None

    def fit(self, X):
        """
        Fits the K-Means model to the given data.

        Args:
            X (array-like): Feature data of shape (n_samples, n_features).

        Returns:
            KMeansClusterer: self, to allow method chaining.
        """
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        self.model.fit(X)
        return self

    def predict(self, X):
        """
        Predicts the closest cluster for each sample in X.

        Args:
            X (array-like): Feature data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Cluster index for each sample.

        Raises:
            RuntimeError: If called before fit().
        """
        self._check_fitted()
        return self.model.predict(X)

    def get_cluster_centers(self):
        """
        Returns the coordinates of the cluster centers.

        Returns:
            numpy.ndarray: Array of shape (n_clusters, n_features).

        Raises:
            RuntimeError: If called before fit().
        """
        self._check_fitted()
        return self.model.cluster_centers_

    def get_inertia(self):
        """
        Returns the within-cluster sum of squares (inertia) of the fitted
        model. Lower values indicate tighter, more compact clusters.

        Returns:
            float: The inertia of the fitted model.

        Raises:
            RuntimeError: If called before fit().
        """
        self._check_fitted()
        return self.model.inertia_

    def get_silhouette_score(self, X):
        """
        Computes the silhouette score for the fitted clustering on X.

        The silhouette score ranges from -1 to 1, where higher values
        indicate well-separated, well-defined clusters.

        Args:
            X (array-like): The same feature data used to fit the model.

        Returns:
            float: The mean silhouette score across all samples.

        Raises:
            RuntimeError: If called before fit().
        """
        self._check_fitted()
        labels = self.model.labels_
        return silhouette_score(X, labels)

    @staticmethod
    def find_optimal_k(X, k_range=range(2, 11), random_state=42):
        """
        Helper for the elbow method: fits K-Means for a range of k values
        and returns their inertia, to help visually choose a good k.

        Args:
            X (array-like): Feature data of shape (n_samples, n_features).
            k_range (iterable): Candidate values of k to try. Defaults to 2-10.
            random_state (int): Seed for reproducible results.

        Returns:
            dict: Mapping of k -> inertia for each candidate k.
        """
        inertias = {}
        for k in k_range:
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            model.fit(X)
            inertias[k] = model.inertia_
        return inertias

    def _check_fitted(self):
        """Raises a clear error if the model hasn't been fitted yet."""
        if self.model is None:
            raise RuntimeError(
                "KMeansClusterer has not been fitted yet. Call fit(X) first."
            )
