# Spark-Kmeans-Scalability

## Project Overview

This project implements a scalable K-means clustering algorithm using Apache Spark to analyze Sentinel satellite embeddings from the Tessera UCAM dataset. The main objective is to study the scale-up behavior of K-means clustering on geospatial data and add interpretability through supervised classification.

## Dataset

The dataset consists of Sentinel embeddings from the **Tessera UCAM** (University of Cambridge) project, specifically GeoTessera embeddings that encode rich geospatial information from satellite imagery.

- **Data source**: Sentinel satellite embeddings (128-dimensional vectors)
- **Coverage**: Multiple geographic tiles with spatial resolution of 1200x1200 pixels per tile
- **Data size**: Variable, depending on the number of patches and their spatial extent
- **Infrastructure**:
  - Primary goal: Run Spark locally on PC
  - Fallback: Limit dataset to 75GB on virtual machines if local execution is not feasible

## Problem Statement

### 1. K-means Clustering for Geospatial Analysis

The core problem involves implementing a distributed K-means clustering algorithm to group similar embeddings and study its scalability characteristics:

- **Objective**: Clusterize high-dimensional embeddings into meaningful geographic groups
- **Scale-up study**: Analyze performance and behavior as dataset size increases
- **Advantage**: No shortage of data availability for comprehensive scalability testing

### 2. Interpretable Classification with KNN

To add interpretability to the unsupervised clustering results, we will build a supervised K-Nearest Neighbors (KNN) model:

- **Training data**: Hand-classified embeddings based on interpretability vectors
- **Classification categories** (subject to refinement based on K-means results):
  - Type of soil
  - Land cover: urban / vegetation / infrastructure
  - Surface type: landmass / water
  - Additional classes to be determined from initial clustering results

### 3. Model Comparison and Interpretability

The final phase involves:

- Comparing K-means clustering results with KNN classification
- Adding semantic meaning to the clusters discovered by K-means
- Understanding what geographic/environmental patterns each cluster represents
- Evaluating whether unsupervised clusters align with supervised classification categories

## Goals

1. **Scalability**: Demonstrate efficient processing of large-scale geospatial embeddings using Spark
2. **Clustering quality**: Identify meaningful geographic patterns through K-means
3. **Interpretability**: Bridge unsupervised clustering with human-interpretable land cover categories
4. **Performance analysis**: Study computational behavior as data volume scales up

## Technologies

- **Apache Spark**: Distributed computing framework
- **PySpark**: Python API for Spark
- **scikit-learn**: Machine learning algorithms (K-means, KNN, PCA)
- **GeoTessera**: Sentinel embedding dataset access
- **Rasterio**: Geospatial data processing and coordinate transformations
- **Matplotlib**: Visualization of clustering results and geographic mosaics
