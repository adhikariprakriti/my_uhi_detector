import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

data_dir = 'data/landsat/'
processed_dir = 'data/processed/kmeans/'
results_dir = 'data/results/kmeans/'
visualizations_dir = 'visualization/kmeans/'

os.makedirs(processed_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)

red_band_path = os.path.join(data_dir, 'LC08_L1TP_020032_20240727_20240801_02_T1_B4.TIF')
nir_band_path = os.path.join(data_dir, 'LC08_L1TP_020032_20240727_20240801_02_T1_B5.TIF')
thermal_band_path = os.path.join(data_dir, 'LC08_L1TP_020032_20240727_20240801_02_T1_B10.TIF')
ndvi_output_path = os.path.join(processed_dir, 'ndvi.tif')
lst_output_path = os.path.join(processed_dir, 'lst.tif')
uhi_output_path = os.path.join(results_dir, 'uhi_kmeans_2_clusters.tif')

ndvi_plot_path = os.path.join(visualizations_dir, 'ndvi_map.png')
lst_plot_path = os.path.join(visualizations_dir, 'lst_map.png')
kmeans_plot_path = os.path.join(visualizations_dir, 'kmeans_result.png')
uhi_plot_path = os.path.join(visualizations_dir, 'uhi_detection.png')

def calculate_ndvi(red_band_path, nir_band_path, output_path):
    with rasterio.open(red_band_path) as red_src:
        red = red_src.read(1).astype(float)
        profile = red_src.profile

    with rasterio.open(nir_band_path) as nir_src:
        nir = nir_src.read(1).astype(float)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    ndvi = (nir - red) / (nir + red + epsilon)
    ndvi = np.clip(ndvi, -1, 1)  # Clip values to valid NDVI range
    ndvi[~np.isfinite(ndvi)] = 0  # Handle any remaining invalid values

    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(ndvi.astype(rasterio.float32), 1)

    return ndvi

def calculate_lst(thermal_band_path, output_path):
    with rasterio.open(thermal_band_path) as thermal_src:
        thermal = thermal_src.read(1).astype(float)
        profile = thermal_src.profile

    # Convert DN to radiance and then to temperature
    ML = 3.342e-4  # Multiplicative rescaling factor
    AL = 0.1      # Additive rescaling factor
    
    radiance = thermal * ML + AL
    lst = (radiance - 273.15)  # Convert to Celsius

    # Filter out invalid values
    lst[~np.isfinite(lst)] = np.nan
    
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(lst.astype(rasterio.float32), 1)

    return lst

def kmeans_clustering(lst, ndvi, n_clusters=5):
    lst_flat = lst.flatten()
    ndvi_flat = ndvi.flatten()
    valid_mask = (
        np.isfinite(lst_flat) & 
        np.isfinite(ndvi_flat) & 
        (lst_flat != 0) & 
        (ndvi_flat != 0)
    )
    valid_lst = lst_flat[valid_mask]
    valid_ndvi = ndvi_flat[valid_mask]
    
    features = np.vstack([valid_lst, valid_ndvi]).T
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300
    )
    
    cluster_labels = kmeans.fit_predict(features_scaled)
    
    # Create output image
    labels_image = np.full(lst_flat.shape, -1, dtype=np.int32)
    labels_image[valid_mask] = cluster_labels
    labels_image = labels_image.reshape(lst.shape)
    
    return {
        'labels': labels_image,
        'centers': kmeans.cluster_centers_,
        'scaler': scaler,
        'inertia': kmeans.inertia_
    }

def identify_uhi_cluster(cluster_info, lst):
    labels = cluster_info['labels']
    
    cluster_temps = []
    for i in range(len(cluster_info['centers'])):
        mask = (labels == i)
        if np.any(mask):
            mean_temp = np.nanmean(lst[mask])
            cluster_temps.append((i, mean_temp))
    
    uhi_cluster = max(cluster_temps, key=lambda x: x[1])[0]
    
    uhi_mask = (labels == uhi_cluster).astype(np.uint8)
    
    return uhi_mask, uhi_cluster

def save_plot(data, title, cmap, plot_path, vmin=None, vmax=None):
    plt.figure(figsize=(12, 8))
    
    # Handle different types of plots
    if title == 'K-means Clusters':
        # Create a masked array for better visualization
        masked_data = np.ma.masked_where(data == -1, data)
        im = plt.imshow(masked_data, cmap='viridis', aspect='auto')
    else:
        im = plt.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    plt.colorbar(im, label=title)
    plt.title(title)
    plt.axis('off')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Calculate NDVI
    print("Calculating NDVI...")
    ndvi = calculate_ndvi(red_band_path, nir_band_path, ndvi_output_path)
    save_plot(ndvi, 'NDVI', 'RdYlGn', ndvi_plot_path, vmin=-1, vmax=1)

    # Calculate LST
    print("Calculating LST...")
    lst = calculate_lst(thermal_band_path, lst_output_path)
    save_plot(lst, 'Land Surface Temperature (°C)', 'hot', lst_plot_path)

    # Perform clustering
    print("Performing K-means clustering...")
    cluster_info = kmeans_clustering(lst, ndvi, n_clusters=2)
    save_plot(cluster_info['labels'], 'K-means Clusters', 'coolwarm', kmeans_plot_path)

    # Identify UHI
    print("Identifying UHI...")
    uhi_mask, uhi_cluster = identify_uhi_cluster(cluster_info, lst)
    save_plot(uhi_mask, 'Urban Heat Island Detection', 'coolwarm', uhi_plot_path)

    # Save UHI result
    with rasterio.open(thermal_band_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1)

    with rasterio.open(uhi_output_path, 'w', **profile) as dst:
        dst.write(uhi_mask.astype(rasterio.uint8), 1)

    print("Processing complete!")

if __name__ == "__main__":
    main()