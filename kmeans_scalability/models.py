from abc import ABC, abstractmethod
from typing import Any
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.feature import VectorAssembler
import time
import numpy as np

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, ArrayType, DoubleType

try:
    from pyspark.sql.functions import pandas_udf, PandasUDFType 
except ImportError:
    from pyspark.sql.pandas.functions import pandas_udf, PandasUDFType


class KMeansInterface(ABC):
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
        print(f"Training Local NumPy K-means with k={self.k}...")
        
  
        if isinstance(data, list):         
            X = np.array([row.features.toArray() for row in data])

        elif hasattr(data, 'toPandas'):            
             print("Warning: Spark DataFrame passed to Local model. Collecting to driver (this may be slow)...")
             pdf = data.select(self.features_col).toPandas()
             X = np.stack(pdf[self.features_col].apply(lambda x: x.toArray()).values)

        else:     
            X = data

        start_time = time.time()
        np.random.seed(self.seed)
       
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iter):           
            distances = np.sum((X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]) ** 2, axis=2)
                      
            labels = np.argmin(distances, axis=1)
            
            new_centroids = np.zeros_like(self.centroids)
            for j in range(self.k):
                mask = labels == j
                if np.any(mask):
                    new_centroids[j] = X[mask].mean(axis=0)
                else:
                    new_centroids[j] = self.centroids[j] 
           
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

        rdd = data.select(self.features_col).rdd.map(lambda row: row[0].toArray())
        rdd.cache()

        self.centroids = rdd.takeSample(False, self.k, self.seed)
        self.centroids = np.array(self.centroids) # Convert list of arrays to numpy matrix

        for i in range(self.max_iter):        
            centroids_broadcast = data.sparkSession.sparkContext.broadcast(self.centroids)
                       
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
         
            point_stats = closest.reduceByKey(
                lambda stat1, stat2: (stat1[0] + stat2[0], stat1[1] + stat2[1])
            )
          
            new_centers_map = point_stats.mapValues(
                lambda stat: stat[0] / stat[1]
            ).collectAsMap()
           
            new_centroids = np.zeros_like(self.centroids)
            change = 0.0
            
            for j in range(self.k):
                if j in new_centers_map:
                    new_centroids[j] = new_centers_map[j]
                else:
                    new_centroids[j] = self.centroids[j] 
                       
            change = np.sum((self.centroids - new_centroids) ** 2)
            self.centroids = new_centroids
            
            if change < 1e-4:
                print(f"Converged at iteration {i}")
                break
        
        # Calculate final WSSSE
        centroids_broadcast = data.sparkSession.sparkContext.broadcast(self.centroids)
        
        def min_sq_dist(p, centers):
            closest_dist = float("inf")
            for center in centers:
                 dist = np.sum((p - center)**2)
                 if dist < closest_dist:
                     closest_dist = dist
            return closest_dist

        self.training_cost = rdd.map(lambda p: min_sq_dist(p, centroids_broadcast.value)).reduce(lambda x, y: x + y)
        print(f"WSSSE: {self.training_cost:.2f}")

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

        initial_rows = data.select(self.features_col).sample(False, 0.1, seed=self.seed).limit(self.k * 2).collect()
        initial_rows = initial_rows[:self.k]
                
        if len(initial_rows) < self.k:
            initial_rows = data.select(self.features_col).limit(self.k).collect()

        self.centroids = [row[0] for row in initial_rows]
       
        @pandas_udf(ArrayType(DoubleType()), PandasUDFType.GROUPED_AGG)
        def vector_mean_udf(features_series):          
            mat = np.stack(features_series.values)    
            return np.mean(mat, axis=0).tolist()
        

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
        
            @F.udf(ArrayType(DoubleType()))
            def vec_to_array(v):
                return v.toArray().tolist()
        
            df_labeled = data.withColumn(self.prediction_col, find_closest_cluster(F.col(self.features_col)))
                       
            df_arrays = df_labeled.withColumn("features_arr", vec_to_array(F.col(self.features_col)))
           
            stats_df = df_arrays.groupBy(self.prediction_col).agg(
                vector_mean_udf(F.col("features_arr")).alias("new_centroid")
            )            
          
            new_centroids_rows = stats_df.collect()
            new_centroids_map = {row[self.prediction_col]: row["new_centroid"] for row in new_centroids_rows}
           
            new_centroids = []
            max_movement = 0.0

            from pyspark.ml.linalg import Vectors

            for j in range(self.k):
                if j in new_centroids_map:               
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

        # Calculate final WSSSE
        current_centroids_list = [c.toArray().tolist() for c in self.centroids]
        bc_centroids = data.sparkSession.sparkContext.broadcast(current_centroids_list)

        @F.udf(DoubleType())
        def get_min_sq_dist(features):
            point = features.toArray()
            centers = bc_centroids.value
            dists = [np.sum((point - np.array(c))**2) for c in centers]
            return float(np.min(dists))

        self.training_cost = data.select(get_min_sq_dist(F.col(self.features_col)).alias("dist")).agg(F.sum("dist")).collect()[0][0]
        print(f"WSSSE: {self.training_cost:.2f}")

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

        # Calculate final WSSSE
        current_centroids_list = [c.toArray().tolist() for c in self.centroids]
        bc_centroids = data.sparkSession.sparkContext.broadcast(current_centroids_list)

        @F.udf(DoubleType())
        def get_min_sq_dist(features):
            point = features.toArray()
            centers = bc_centroids.value
            dists = [np.sum((point - np.array(c))**2) for c in centers]
            return float(np.min(dists))

        self.training_cost = data.select(get_min_sq_dist(F.col(self.features_col)).alias("dist")).agg(F.sum("dist")).collect()[0][0]
        print(f"WSSSE: {self.training_cost:.2f}")

        self.training_time = time.time() - start_time
        print(f"Training completed in {self.training_time:.2f} seconds")
        return self

    def transform(self, data):  
        if self.centroids is None: raise ValueError("Model must be trained")

        current_centroids_list = [c.toArray().tolist() for c in self.centroids]
        bc_centroids = data.sparkSession.sparkContext.broadcast(current_centroids_list)

        @F.udf(IntegerType())

        def predict_cluster(features):
            point = features.toArray()
            centers = bc_centroids.value
            dists = [np.sum((point - np.array(c))**2) for c in centers]
            return int(np.argmin(dists))
        
        return data.withColumn(self.prediction_col, predict_cluster(F.col(self.features_col)))
