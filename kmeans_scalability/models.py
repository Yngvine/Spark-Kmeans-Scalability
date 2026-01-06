from abc import ABC, abstractmethod
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.feature import VectorAssembler
import time
import numpy as np

class KMeansInterface(ABC):
    """
    Abstract base class for K-Means implementations.
    """
    def __init__(self, k=4, max_iter=20, seed=42, features_col="features", prediction_col="prediction"):
        self.k = k
        self.max_iter = max_iter
        self.seed = seed
        self.features_col = features_col
        self.prediction_col = prediction_col
        self.model = None
        self.training_cost = None

    @abstractmethod
    def fit(self, data):
        """
        Train the K-Means model.
        :param data: Input data (Spark DataFrame or RDD depending on implementation)
        """
        pass

    @abstractmethod
    def transform(self, data):
        """
        Assign clusters to the data.
        :param data: Input data
        :return: Data with cluster assignments
        """
        pass

class SparkMLKMeansModel(KMeansInterface):
    """
    Wrapper around PySpark MLlib's KMeans.
    """
    def fit(self, data):
        print(f"Training Spark ML K-means with k={self.k}...")
        start_time = time.time()
        
        kmeans = SparkKMeans(k=self.k, seed=self.seed, maxIter=self.max_iter, featuresCol=self.features_col, predictionCol=self.prediction_col)
        self.model = kmeans.fit(data)
        
        elapsed_time = time.time() - start_time
        self.training_cost = self.model.summary.trainingCost
        print(f"Training completed in {elapsed_time:.2f} seconds")
        print(f"Within Set Sum of Squared Errors (WSSSE): {self.training_cost:.2f}")
        return self

    def transform(self, data):
        if self.model is None:
            raise ValueError("Model must be trained before transform")
        return self.model.transform(data)

class CustomRDDKMeansModel(KMeansInterface):
    """
    Custom implementation of K-Means using Spark RDDs.
    """
    def fit(self, data):
        # Placeholder for custom RDD implementation
        # Logic would involve:
        # 1. Initialize centroids
        # 2. Loop max_iter times:
        #    a. Assign points to nearest centroid (map)
        #    b. Compute new centroids (reduceByKey)
        #    c. Check convergence
        print("Training Custom RDD K-means (Not Implemented)...")
        pass

    def transform(self, data):
        # Placeholder
        pass

class CustomDataFrameKMeansModel(KMeansInterface):
    """
    Custom implementation of K-Means using Spark DataFrames.
    """
    def fit(self, data):
        # Placeholder for custom DataFrame implementation
        print("Training Custom DataFrame K-means (Not Implemented)...")
        pass

    def transform(self, data):
        # Placeholder
        pass
