# Spark K-means Scalability Study

## 📋 Project Overview

This project implements and benchmarks multiple K-means clustering implementations using Apache Spark to analyze Sentinel satellite embeddings from the GeoTessera dataset. The study focuses on scalability analysis, comparing distributed computing approaches, and adding interpretability to unsupervised clustering through supervised classification.

## 🎯 Objectives

1. **Scalability Analysis**: Study the scale-up behavior of different K-means implementations (local vs. distributed)
2. **Performance Benchmarking**: Compare Spark MLlib, custom RDD/DataFrame implementations, and local NumPy baseline
3. **Geospatial Clustering**: Identify meaningful geographic patterns in satellite imagery embeddings
4. **Interpretability**: Bridge unsupervised clustering with human-interpretable land cover categories using KNN classification

## 📊 Dataset

**Source**: GeoTessera - Sentinel satellite embeddings (Pamplona region, Spain)

- **Embedding dimension**: 128 features per pixel
- **Geographic coverage**: Multiple tiles around coordinates (42.8°N, -1.6°W)
- **Spatial resolution**: 1200×1200 pixels per tile
- **Total pixels**: ~2M pixels across multiple tiles
- **Data format**: Parquet (columnar storage for efficient processing)

## 🏗️ Project Structure

```
Spark-Kmeans-Scalability/
├── README.md                              # Project documentation
├── LICENSE                                # MIT License
├── pyproject.toml                         # Python dependencies
│
├── kmeans_scalability/                    # Custom K-means library
│   ├── __init__.py                        # Package initialization
│   ├── models.py                          # K-means model implementations
│   └── visualization.py                   # Plotting and visualization utilities
│
├── knn_vs_kmeans_classification.ipynb     # Main analysis: KNN classification vs K-Means
├── sizeup_scalability_test.ipynb          # SizeUp scalability benchmark
├── generateEmbeddingParquet.ipynb         # Data preprocessing notebook
│
├── GeoTessera_Pamplona_embeddings.parquet # Preprocessed dataset (~2M points)
├── KNN_POINTS.geojson                     # Manually labeled training data (80 points: 20 per class)
├── .python-version                        # Python version configuration
├── uv.lock                                # UV package manager lock file
│
├── global_0.1_degree_representation/      # Tile data (0.1° grid)
├── global_0.1_degree_tiff_all/            # Auxiliary TIFF files
├── output_mosaics/                        # Generated visualization mosaics
│   ├── kmeans_mosaic.tiff                 # K-means cluster map (GeoTIFF)
│   └── knn_mosaic.tiff                    # KNN class map (GeoTIFF)
└── spark_temp/                            # Spark temporary files
```

## 🔧 Technologies

- **Apache Spark 3.x**: Distributed computing framework
- **PySpark**: Python API for Spark MLlib and RDD operations
- **scikit-learn**: Machine learning (KNN, PCA)
- **NumPy/Pandas**: Data manipulation and baseline implementation
- **Rasterio**: Geospatial raster processing and CRS transformations
- **Matplotlib**: Visualization and georeferenced maps
- **PyProj**: Coordinate system transformations

## 🚀 K-means Implementations

The project implements and compares **5 different K-means approaches**:

1. **Local NumPy (Baseline)**: Single-machine implementation for comparison
2. **Spark MLlib**: Official distributed K-means from PySpark
3. **Custom RDD Implementation**: Low-level RDD-based distributed K-means
4. **Custom DataFrame (UDF)**: DataFrame API with user-defined functions
5. **Optimized DataFrame**: Optimized version using built-in Spark operations

## 📈 Key Features

### 1. Scalability Benchmarking

- **SizeUp testing** with data fractions: 10%, 25%, 50%, 75%, 100%
- Performance metrics: training time, WSSSE (Within-Set Sum of Squared Errors)
- Comparison of local vs. distributed approaches
- Ideal linear scale-up baseline for evaluation

### 2. Geospatial Analysis

- Cluster centroids analysis and pairwise distances
- PCA visualization of high-dimensional embeddings
- Georeferenced cluster maps with proper CRS handling
- GeoTIFF export for GIS software (QGIS, ArcGIS)

### 3. Interpretable Classification

- **Land cover classes**: Vegetation, Forest, Urban, Agricultural (4 classes)
- **Training data**: 80 manually labeled points (20 per class) from KNN_POINTS.geojson
- KNN classifier (k=5) trained on labeled embeddings
- Cross-tabulation of unsupervised clusters vs. supervised classes
- Confusion matrix and classification report

### 4. Visualization

- PCA projection of clusters in 2D space
- Centroid distance matrices and heatmaps
- Side-by-side comparison: K-means vs. KNN results
- Georeferenced mosaics with spatial context

## 📋 Workflow

### 1. Data Preparation (`generateEmbeddingParquet.ipynb`)

- Download Sentinel tiles from GeoTessera API
- Extract 128-dimensional embeddings per pixel
- Store metadata (GPS coordinates, CRS, transforms)
- Export to Parquet format for efficient Spark processing

### 2. Main Analysis (`knn_vs_kmeans_classification.ipynb`)

- Initialize Spark session with optimized configuration (~16GB memory)
- Load embeddings dataset (~2M points) and create feature vectors
- Load manually labeled training points from KNN_POINTS.geojson (80 points)
- Train KNN classifier (k=5) on labeled data with 4 classes
- Predict land cover classes for all ~2M points using KNN
- Train K-means clustering (k=4) with Spark MLlib
- Compare KNN classification vs K-means clustering results
- PCA 2D visualization with centroids and training points
- Generate georeferenced mosaics for both methods
- Export GeoTIFF files for GIS software

### 3. SizeUp Benchmark (`sizeup_scalability_test.ipynb`)

- Compare 5 K-means implementations
- Test with data fractions: 10%, 25%, 50%, 75%, 100%
- Measure training time and WSSSE for each model
- Plot scalability curves vs. ideal linear scale-up
- Identify performance bottlenecks

## 🎓 Results Summary

### Clustering Performance

- **K-means clusters**: 4 well-separated groups
- **Training time**: ~10-30 seconds for full dataset (varies by implementation)
- **WSSSE**: Quantified cluster compactness
- **PCA variance**: 2D projection captures significant variance

### Interpretable Classes

- **Vegetation**: Green areas
- **Urban**: Built-up areas and buildings
- **Agricultural**: Croplands and farmland
- **Forest**: Forests

### Scalability Insights

- Spark MLlib shows near-linear scale-up for large datasets
- Local NumPy becomes prohibitive beyond ~100K samples
- Custom RDD implementation offers flexibility with acceptable overhead
- Optimized DataFrame implementation balances performance and usability

## 🔬 Technical Details

### Spark Configuration

```python
spark = SparkSession.builder \
    .appName("Spark-Kmeans-Scalability") \
    .master("local[*]") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.local.dir", "spark_temp") \
    .getOrCreate()
```

### K-means Parameters

- **k**: 4 clusters
- **max_iter**: 10-20 iterations
- **seed**: 42 (for reproducibility)
- **features**: 128-dimensional embeddings

### Labeled Training Data

**Source**: `KNN_POINTS.geojson` (manually created)

- **Total points**: 80 (20 per class)
- **Classes**:
  - Vegetation (20 points)
  - Forest (20 points)
  - Urban (20 points)
  - Agricultural (20 points)
- **Format**: GeoJSON with polygon geometries (converted to centroids)
- **CRS**: EPSG:4326 → transformed to EPSG:32630 (UTM Zone 30N)

**KNN Configuration**:

- Algorithm: K-Nearest Neighbors (k=5)
- Distance metric: Euclidean (128-dimensional embedding space)
- Training/prediction: scikit-learn implementation

## 📦 Installation

### Requirements

- Python 3.8+
- Apache Spark 3.x
- Java 8 or 11 (required for Spark)
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Setup with UV (Recommended)

```bash
# Clone repository
git clone https://github.com/Yngvine/Spark-Kmeans-Scalability.git
cd Spark-Kmeans-Scalability

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Dependencies

- pyspark>=3.0.0
- numpy>=1.20.0
- pandas>=1.3.0
- matplotlib>=3.3.0
- scikit-learn>=0.24.0
- rasterio>=1.2.0
- pyproj>=3.0.0
- geopandas>=0.10.0
- scipy>=1.7.0

### Expected Outputs

- `output_mosaics/kmeans_mosaic.tiff`: Georeferenced K-means cluster map
- `output_mosaics/knn_mosaic.tiff`: Georeferenced KNN class map
- `GeoTessera_Pamplona_embeddings.parquet`: Processed embedding data
- Console output: Performance metrics and statistics
- Plots: PCA projections, confusion matrices, scalability curves

## 🗺️ GIS Integration

The exported GeoTIFF files can be opened in:

- **QGIS**: Free and open-source GIS software
- **ArcGIS**: Commercial GIS platform
- **Google Earth Engine**: Cloud-based geospatial analysis
- **Python**: Using `rasterio`, `geopandas`, etc.

### Load in QGIS

```
Layer → Add Raster Layer → output_mosaics/kmeans_mosaic.tiff
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Authors

- GitHub: [@Yngvine](https://github.com/Yngvine)
- GitHub: [@Ninjalice](https://github.com/Ninjalice)

## 🙏 Acknowledgments

- **GeoTessera Project** for providing Sentinel satellite embeddings
- **Apache Spark** community for the distributed computing framework
- **University of Cambridge** for the Tessera dataset infrastructure

## 📚 References

- Apache Spark MLlib Documentation
- GeoTessera Dataset
- Sentinel Satellite Program (ESA)
- K-means Clustering Algorithm

---

**Last Updated**: January 2026
