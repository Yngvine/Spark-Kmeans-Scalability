import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np
from rasterio.transform import from_bounds

def plot_scalability(data_points, training_times, wssse_values):
    """
    Visualize scalability results.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Training time vs data size
    axes[0].plot(data_points, training_times, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of Data Points', fontsize=12)
    axes[0].set_ylabel('Training Time (seconds)', fontsize=12)
    axes[0].set_title('K-means Scalability: Training Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: WSSSE vs data size
    axes[1].plot(data_points, wssse_values, 's-', color='orange', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of Data Points', fontsize=12)
    axes[1].set_ylabel('WSSSE', fontsize=12)
    axes[1].set_title('K-means Scalability: Clustering Quality', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def create_mosaic(tile_metadata, cluster_labels):
    """
    Reconstruct spatial maps for each tile and assemble them into a mosaic.
    """
    print("Reconstructing spatial maps for each tile...")

    tile_maps = []
    start_idx = 0

    for idx, tile in enumerate(tile_metadata):
        n_pixels = tile['n_pixels']
        
        # Extract labels for this tile
        tile_cluster_labels = cluster_labels[start_idx:start_idx + n_pixels]
        
        # Reshape back to spatial dimensions
        h, w = tile['shape'][0], tile['shape'][1]
        tile_cluster_map = tile_cluster_labels.reshape(h, w)
        
        tile_maps.append({
            'cluster_map': tile_cluster_map,
            'lat': tile['lat'],
            'lon': tile['lon'],
            'crs': tile['crs'],
            'transform': tile['transform'],
            'shape': tile['shape']
        })
        
        start_idx += n_pixels

    print(f"✓ Created spatial maps for {len(tile_maps)} tiles")
    
    # Calculate global bounds for all tiles
    all_bounds = []
    for tile in tile_maps:
        t = tile['transform']
        h, w = tile['shape'][0], tile['shape'][1]
        
        # Calculate tile bounds
        min_x = t.c
        max_y = t.f
        max_x = min_x + w * t.a
        min_y = max_y + h * t.e  # e is negative
        
        all_bounds.append((min_x, min_y, max_x, max_y))

    # Global bounding box
    min_x = min(b[0] for b in all_bounds)
    min_y = min(b[1] for b in all_bounds)
    max_x = max(b[2] for b in all_bounds)
    max_y = max(b[3] for b in all_bounds)

    print(f"Global bounds: ({min_x:.6f}, {min_y:.6f}, {max_x:.6f}, {max_y:.6f})")

    # Use resolution from first tile
    first_transform = tile_maps[0]['transform']
    pixel_size_x = first_transform.a
    pixel_size_y = -first_transform.e

    width = int((max_x - min_x) / pixel_size_x)
    height = int((max_y - min_y) / pixel_size_y)

    print(f"Mosaic dimensions: {width} x {height} pixels")

    # Initialize mosaic with -1 (no data)
    cluster_mosaic = np.full((height, width), -1, dtype=np.int32)

    # Place each tile in the mosaic
    print("\nAssembling mosaic...")
    for idx, tile in enumerate(tile_maps):
        cluster_map = tile['cluster_map'].astype(np.int32)
        tile_transform = tile['transform']
        
        # Calculate pixel offset in the mosaic
        tile_min_x = tile_transform.c
        tile_max_y = tile_transform.f
        
        col_offset = int((tile_min_x - min_x) / pixel_size_x)
        row_offset = int((max_y - tile_max_y) / pixel_size_y)
        
        tile_height, tile_width = cluster_map.shape
        
        # Place the tile in the mosaic
        cluster_mosaic[row_offset:row_offset+tile_height, col_offset:col_offset+tile_width] = cluster_map
        
    print(f"\n✓ Mosaic assembly complete!")
    
    output_transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
    extent = [min_x, max_x, min_y, max_y]
    return cluster_mosaic, extent, tile_maps[0]['crs'], output_transform

def plot_mosaic(mosaic, extent, k, title='Georeferenced Map - Clusters', label_prefix='Cluster', class_names=None):
    """
    Visualize the cluster mosaic.
    """
    min_x, max_x, min_y, max_y = extent
    
    # Create custom colormap with black for no data
    cmap_clusters = plt.get_cmap('inferno')
    # Use linspace to sample k colors from the colormap range [0, 1]
    # We start from 0.2 to avoid very dark colors that look like the background
    colors_list = [cmap_clusters(x) for x in np.linspace(0.2, 1.0, k)]
    colors_list.insert(0, (0, 0, 0, 1))  # Black for no-data (-1)

    custom_cmap = mcolors.ListedColormap(colors_list)
    bounds = list(range(-1, k + 1))
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))

    im = ax.imshow(mosaic, cmap=custom_cmap, norm=norm, interpolation='nearest',
                   extent=(min_x, max_x, min_y, max_y))

    ax.set_title(f'{title}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, ticks=np.arange(-0.5, k, 1))
    cbar.set_label('ID', fontsize=12)
    
    if class_names:
        if isinstance(class_names, dict):
            labels = [class_names[i] for i in range(k)]
        else:
            labels = class_names
        cbar.ax.set_yticklabels(['No Data'] + labels)
    else:
        cbar.ax.set_yticklabels(['No Data'] + [f'{label_prefix} {i}' for i in range(k)])

    # Grid and bbox outline
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                     linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax.add_patch(rect)

    plt.tight_layout()
    plt.show()

def plot_comparison(cluster_mosaic, knn_mosaic, extent, k, class_names):
    """
    Side-by-side comparison of K-means clusters vs KNN classes.
    """
    min_x, max_x, min_y, max_y = extent
    n_classes = len(class_names)
    
    # Colormaps
    # Clusters
    cmap_clusters = plt.get_cmap('inferno')
    colors_list = [cmap_clusters(x) for x in np.linspace(0.2, 1.0, k)]
    colors_list.insert(0, (0, 0, 0, 1))
    custom_cmap = mcolors.ListedColormap(colors_list)
    bounds = list(range(-1, k + 1))
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
    
    # Classes
    cmap_classes = plt.get_cmap('tab10').copy()
    colors_list_classes = [cmap_classes(i) for i in range(n_classes)]
    colors_list_classes.insert(0, (0, 0, 0, 1))
    custom_cmap_classes = mcolors.ListedColormap(colors_list_classes)
    bounds_classes = list(range(-1, n_classes + 1))
    norm_classes = mcolors.BoundaryNorm(bounds_classes, custom_cmap_classes.N)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Left: K-means clusters
    im1 = axes[0].imshow(cluster_mosaic, cmap=custom_cmap, norm=norm, 
                         interpolation='nearest', extent=(min_x, max_x, min_y, max_y))
    axes[0].set_title(f'K-means Clusters (k={k})', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Longitude', fontsize=12)
    axes[0].set_ylabel('Latitude', fontsize=12)
    axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    cbar1 = plt.colorbar(im1, ax=axes[0], ticks=np.arange(-0.5, k, 1))
    cbar1.set_label('Cluster ID', fontsize=12)
    cbar1.ax.set_yticklabels(['No Data'] + [str(i) for i in range(k)])

    # Right: KNN interpretable classes
    im2 = axes[1].imshow(knn_mosaic, cmap=custom_cmap_classes, norm=norm_classes,
                         interpolation='nearest', extent=(min_x, max_x, min_y, max_y))
    axes[1].set_title(f'KNN Interpretable Classes', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Longitude', fontsize=12)
    axes[1].set_ylabel('Latitude', fontsize=12)
    axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    cbar2 = plt.colorbar(im2, ax=axes[1], ticks=np.arange(-0.5, n_classes, 1))
    cbar2.set_label('Land Cover Class', fontsize=12)
    # Assuming class_names is a dict or list. If dict, convert to list ordered by index
    if isinstance(class_names, dict):
        class_names_list = [class_names[i] for i in range(n_classes)]
    else:
        class_names_list = class_names
        
    cbar2.ax.set_yticklabels(['No Data'] + class_names_list)

    plt.suptitle('Comparison: Unsupervised Clusters vs Interpretable Classes', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
