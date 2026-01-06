from abc import ABC, abstractmethod
from typing import Any
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.feature import VectorAssembler
import time
import numpy as np

# PySpark SQL imports
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, ArrayType, DoubleType
# We use try/except to handle Pylance/Version confusion, but prefer standard location
try:
    from pyspark.sql.functions import pandas_udf, PandasUDFType # type: ignore
except ImportError:
    from pyspark.sql.pandas.functions import pandas_udf, PandasUDFType

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
        self.training_time = 0.0

    @abstractmethod
    def fit(self, data) -> Any:
        """
        Train the K-Means model.
        :param data: Input data (Spark DataFrame or RDD depending on implementation)
        """
        pass

    @abstractmethod
    def transform(self, data) -> Any:
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
        
        self.training_time = time.time() - start_time
        self.training_cost = self.model.summary.trainingCost
        print(f"Training completed in {self.training_time:.2f} seconds")
        print(f"Within Set Sum of Squared Errors (WSSSE): {self.training_cost:.2f}")
        return self

    def transform(self, data):
        if self.model is None:
            raise ValueError("Model must be trained before transform")
        return self.model.transform(data)


class LocalNumPyKMeans(KMeansInterface):
    """
    Local implementation of K-Means using basic NumPy (Single node execution).
    Designed to serve as a baseline for scalability comparison.
    """
    def __init__(self, k=4, max_iter=20, seed=42):
        super().__init__(k, max_iter, seed)
        self.centroids = None

    def fit(self, data):
        """
        Fits the model using NumPy.
        :param data: List of Rows from Spark collect() or numpy array
        """
        print(f"Training Local NumPy K-means with k={self.k}...")
        
        # Convert input to numpy array
        if isinstance(data, list):
             # Handle list of Rows from Spark collect()
            X = np.array([row.features.toArray() for row in data])
        elif hasattr(data, 'toPandas'):
             # Handle Spark DataFrame passed by mistake
             print("Warning: Spark DataFrame passed to Local model. Collecting to driver (this may be slow)...")
             pdf = data.select(self.features_col).toPandas()
             X = np.stack(pdf[self.features_col].apply(lambda x: x.toArray()).values)
        else:
             # Assume numpy array
            X = data

        start_time = time.time()
        np.random.seed(self.seed)

        # 1. Initialize Centroids (Randomly select k points)
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iter):
            # 2. Assignment Step: Vectors - Centroids
            # Broadcasting: (N, 1, D) - (1, K, D) -> (N, K, D)
            # We compute squared Euclidean distance
            distances = np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=2)
            
            # Get nearest cluster for each point
            labels = np.argmin(distances, axis=1)

            # 3. Update Step: Mean of points in each cluster
            new_centroids = np.zeros_like(self.centroids)
            for j in range(self.k):
                mask = labels == j
                if np.any(mask):
                    new_centroids[j] = X[mask].mean(axis=0)
                else:
                    new_centroids[j] = self.centroids[j] # Keep old if empty

            # Check convergence
            if np.allclose(self.centroids, new_centroids, atol=1e-4):
                print(f"Converged at iteration {i}")
                break
            
            self.centroids = new_centroids

        # Calculate final WSSSE
        distances = np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=2)
        min_distances = np.min(distances, axis=1)
        self.training_cost = np.sum(min_distances)
        
        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")
        print(f"WSSSE: {self.training_cost:.2f}")
        return self

    def transform(self, data):
        # Implementation skipped for scalability assignment focus (usually prediction only)
        pass

class CustomRDDKMeansModel(KMeansInterface):
    """
    Custom implementation of K-Means using Spark RDDs.
    """
    def fit(self, data):
        print(f"Training Custom RDD K-means with k={self.k}...")
        start_time = time.time()

        # Convert DataFrame to RDD of numpy arrays
        # data is expected to be a Spark DataFrame with a Vector column 'features'
        rdd = data.select(self.features_col).rdd.map(lambda row: row[0].toArray())
        rdd.cache()

        # 1. Initialize Centroids (Take sample)
        # We take random points as initial centroids
        self.centroids = rdd.takeSample(False, self.k, self.seed)
        self.centroids = np.array(self.centroids) # Convert list of arrays to numpy matrix

        for i in range(self.max_iter):
            # Broadcast centroids to all executors to avoid sending them with every task
            centroids_broadcast = data.sparkSession.sparkContext.broadcast(self.centroids)
            
            # 2. Assignment Step (Map)
            # For each point, find index of closest centroid.
            # Emits: (cluster_index, (point_vector, 1))
            # We emit (point, 1) to easily calculate sum and count later
            def closest_point(p, centers):
                best_index = 0
                closest_dist = float("inf")
                for idx, center in enumerate(centers):
                    dist = np.sum((p - center)**2)
                    if dist < closest_dist:
                        closest_dist = dist
                        best_index = idx
                return best_index

            closest = rdd.map(
                lambda p: (closest_point(p, centroids_broadcast.value), (p, 1))
            )

            # 3. Update Step (ReduceByKey)
            # Sum points and counts per cluster
            # (Vector1, count1) + (Vector2, count2) -> (Vector1+Vector2, count1+count2)
            point_stats = closest.reduceByKey(
                lambda stat1, stat2: (stat1[0] + stat2[0], stat1[1] + stat2[1])
            )

            # Compute new centers: Sum / Count
            new_centers_map = point_stats.mapValues(
                lambda stat: stat[0] / stat[1]
            ).collectAsMap()

            # Construct new centroid array (handle potential empty clusters)
            new_centroids = np.zeros_like(self.centroids)
            change = 0.0
            
            for j in range(self.k):
                if j in new_centers_map:
                    new_centroids[j] = new_centers_map[j]
                else:
                    new_centroids[j] = self.centroids[j] # Keep old if cluster died
            
            # Check convergence simple Euclidean distance between user old and new centroids
            change = np.sum((self.centroids - new_centroids) ** 2)
            self.centroids = new_centroids
            
            if change < 1e-4:
                print(f"Converged at iteration {i}")
                break
        
        # Calculate Cost (WSSSE) - Optional, adds one pass
        # cost = rdd.map(lambda p: np.min(np.sum((p - self.centroids)**2, axis=1))).sum()
        # self.training_cost = cost

        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")
        return self
    
    def transform(self, data):
        # Placeholder
        pass



class CustomDataFrameKMeansModel(KMeansInterface):
    """
    Custom implementation of K-Means using Spark DataFrames with UDFs.
    Primary implementation using User Defined Functions for flexibility.
    """
    def fit(self, data):
        print(f"Training Custom DataFrame K-means (UDF) with k={self.k}...")
        start_time = time.time()

        

        # 1. Initialize Centroids
        # Sample and collect k rows
        initial_rows = data.select(self.features_col).sample(False, 0.1, seed=self.seed).limit(self.k * 2).collect()
        initial_rows = initial_rows[:self.k]
        
        # Fallback
        if len(initial_rows) < self.k:
            initial_rows = data.select(self.features_col).limit(self.k).collect()

        self.centroids = [row[0] for row in initial_rows]

        # Define Pandas UDF for aggregation
        # We use explicit PandasUDFType.GROUPED_AGG to satisfy linters and ensure correct execution type.
        # This matches the signature: pandas_udf(returnType, functionType=None)
        @pandas_udf(ArrayType(DoubleType()), PandasUDFType.GROUPED_AGG)
        def vector_mean_udf(features_series):
            # Convert series of arrays to numpy matrix
            # features_series is a pandas Series of lists/arrays
            mat = np.stack(features_series.values)
            # Calculate mean along axis 0
            return np.mean(mat, axis=0).tolist()

        for i in range(self.max_iter):
             # Broadcast centroids
            current_centroids_list = [c.toArray().tolist() for c in self.centroids]
            bc_centroids = data.sparkSession.sparkContext.broadcast(current_centroids_list)

            # 2. Assignment Step: Standard Python UDF
            @F.udf(IntegerType())
            def find_closest_cluster(features):
                if features is None: return -1
                point = features.toArray()
                centers = bc_centroids.value
                dists = [np.sum((point - np.array(c))**2) for c in centers]
                return int(np.argmin(dists))

            # Helper to convert Vector to Array (needed for Pandas UDF input usually)
            @F.udf(ArrayType(DoubleType()))
            def vec_to_array(v):
                return v.toArray().tolist()

            # Add prediction column
            df_labeled = data.withColumn(self.prediction_col, find_closest_cluster(F.col(self.features_col)))
            
            # Convert vector column to array column for Pandas UDF interaction
            df_arrays = df_labeled.withColumn("features_arr", vec_to_array(F.col(self.features_col)))

            # 3. Update Step: GroupBy and Aggregate using Pandas UDF
            stats_df = df_arrays.groupBy(self.prediction_col).agg(
                vector_mean_udf(F.col("features_arr")).alias("new_centroid")
            )
            
            # Collect and update
            new_centroids_rows = stats_df.collect()
            new_centroids_map = {row[self.prediction_col]: row["new_centroid"] for row in new_centroids_rows}

            # Reconstruct centroids
            new_centroids = []
            max_movement = 0.0

            from pyspark.ml.linalg import Vectors

            for j in range(self.k):
                if j in new_centroids_map:
                    # Convert list back to Vector
                    new_vec = Vectors.dense(new_centroids_map[j])
                    old_vec = self.centroids[j]
                    
                    dist = np.sum((old_vec.toArray() - new_vec.toArray()) ** 2)
                    if dist > max_movement:
                        max_movement = dist
                    new_centroids.append(new_vec)
                else:
                    new_centroids.append(self.centroids[j])

            self.centroids = new_centroids
            
            if max_movement < 1e-4:
                print(f"Converged at iteration {i}")
                break

        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")
        return self

    def transform(self, data):
        if self.centroids is None:
             raise ValueError("Model must be trained before transform")

        from pyspark.sql import functions as F
        from pyspark.sql.types import IntegerType

        current_centroids_list = [c.toArray().tolist() for c in self.centroids]
        bc_centroids = data.sparkSession.sparkContext.broadcast(current_centroids_list)

        @F.udf(IntegerType())
        def predict_cluster(features):
            point = features.toArray()
            centers = bc_centroids.value
            dists = [np.sum((point - np.array(c))**2) for c in centers]
            return int(np.argmin(dists))

        return data.withColumn(self.prediction_col, predict_cluster(F.col(self.features_col)))


class OptimizedDataFrameKMeansModel(KMeansInterface):
    """
    Secondary implementation: Optimized DataFrame K-Means using Summarizer.
    This version uses Spark's native Summarizer for more efficient vector aggregation.
    """
    def fit(self, data):
        print(f"Training Optimized DataFrame K-means (Summarizer) with k={self.k}...")
        start_time = time.time()

        from pyspark.sql import functions as F
        from pyspark.sql.types import IntegerType
        from pyspark.ml.stat import Summarizer

        # 1. Initialize logic (same as above)
        initial_rows = data.select(self.features_col).sample(False, 0.1, seed=self.seed).limit(self.k * 2).collect()
        initial_rows = initial_rows[:self.k]
        if len(initial_rows) < self.k:
            initial_rows = data.select(self.features_col).limit(self.k).collect()
        self.centroids = [row[0] for row in initial_rows]

        for i in range(self.max_iter):
            current_centroids_list = [c.toArray().tolist() for c in self.centroids]
            bc_centroids = data.sparkSession.sparkContext.broadcast(current_centroids_list)

            @F.udf(IntegerType())
            def find_closest_cluster(features):
                if features is None: return -1
                point = features.toArray()
                centers = bc_centroids.value
                dists = [np.sum((point - np.array(c))**2) for c in centers]
                return int(np.argmin(dists))

            df_with_clusters = data.withColumn(self.prediction_col, find_closest_cluster(F.col(self.features_col)))

            # 3. Update Step: Uses Summarizer (Optimized)
            stats_df = df_with_clusters.groupBy(self.prediction_col).agg(
                Summarizer.mean(F.col(self.features_col)).alias("new_centroid")
            )
            
            new_centroids_rows = stats_df.collect()
            new_centroids_map = {row[self.prediction_col]: row["new_centroid"] for row in new_centroids_rows}

            new_centroids = []
            max_movement = 0.0

            for j in range(self.k):
                if j in new_centroids_map:
                    new_vec = new_centroids_map[j]
                    old_vec = self.centroids[j]
                    dist = np.sum((old_vec.toArray() - new_vec.toArray()) ** 2)
                    if dist > max_movement: max_movement = dist
                    new_centroids.append(new_vec)
                else:
                    new_centroids.append(self.centroids[j])

            self.centroids = new_centroids
            if max_movement < 1e-4:
                print(f"Converged at iteration {i}")
                break

        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")
        return self

    def transform(self, data):
        # Same transform logic as other UDF model
        if self.centroids is None: raise ValueError("Model must be trained")
        from pyspark.sql import functions as F
        from pyspark.sql.types import IntegerType
        current_centroids_list = [c.toArray().tolist() for c in self.centroids]
        bc_centroids = data.sparkSession.sparkContext.broadcast(current_centroids_list)
        @F.udf(IntegerType())
        def predict_cluster(features):
            point = features.toArray()
            centers = bc_centroids.value
            dists = [np.sum((point - np.array(c))**2) for c in centers]
            return int(np.argmin(dists))
        return data.withColumn(self.prediction_col, predict_cluster(F.col(self.features_col)))
