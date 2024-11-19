import os
import numpy as np
import rasterio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define directories for data and results
data_dir = 'data/landsat/'
processed_dir = 'data/processed/kmeans/'
results_dir = 'data/results/kmeans/'
visualizations_dir = 'visualization/kmeans/'

os.makedirs(processed_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(visualizations_dir, exist_ok=True)

# Define file paths
red_band_path = os.path.join(data_dir, 'LC08_L1TP_020032_20240727_20240801_02_T1_B4.TIF')
nir_band_path = os.path.join(data_dir, 'LC08_L1TP_020032_20240727_20240801_02_T1_B5.TIF')
thermal_band_path = os.path.join(data_dir, 'LC08_L1TP_020032_20240727_20240801_02_T1_B10.TIF')
ndvi_output_path = os.path.join(processed_dir, 'ndvi.tif')
lst_output_path = os.path.join(processed_dir, 'lst.tif')
uhi_output_path = os.path.join(results_dir, 'uhi_kmeans.tif')
ndvi_plot_path = os.path.join(visualizations_dir, 'ndvi_map.png')
lst_plot_path = os.path.join(visualizations_dir, 'lst_map.png')
kmeans_plot_path = os.path.join(visualizations_dir, 'kmeans_result.png')
uhi_plot_path = os.path.join(visualizations_dir, 'uhi_detection.png')

# Function to calculate NDVI
def calculate_ndvi(red_band_path, nir_band_path, output_path):
    with rasterio.open(red_band_path) as red_src:
        red = red_src.read(1).astype(float)
        red_meta = red_src.meta.copy()

    with rasterio.open(nir_band_path) as nir_src:
        nir = nir_src.read(1).astype(float)

    ndvi = (nir - red) / (nir + red + 1e-10)  # Avoid division by zero
    ndvi = np.clip(ndvi, -1, 1)  # NDVI range

    # Handle invalid values
    ndvi[~np.isfinite(ndvi)] = np.nan

    red_meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    with rasterio.open(output_path, 'w', **red_meta) as dst:
        dst.write(ndvi.astype(rasterio.float32), 1)

    # Debug: Print NDVI statistics
    print(f"NDVI - min: {np.nanmin(ndvi):.4f}, max: {np.nanmax(ndvi):.4f}, mean: {np.nanmean(ndvi):.4f}")

    return ndvi

def calculate_lst(thermal_band_path, output_path):
    with rasterio.open(thermal_band_path) as thermal_src:
        thermal = thermal_src.read(1).astype(float)
        profile = thermal_src.profile

    # Convert DN to radiance and then to temperature
    ML = 3.342e-4
    AL = 0.1

    radiance = thermal * ML + AL
    lst = (radiance - 273.15)
    lst[~np.isfinite(lst)] = np.nan

    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(lst.astype(rasterio.float32), 1)

    return lst
'''
# Function to calculate LST
def calculate_lst(thermal_band_path, output_path):
    with rasterio.open(thermal_band_path) as thermal_src:
        thermal = thermal_src.read(1).astype(float)
        thermal_meta = thermal_src.meta.copy()
    
    # Exclude pixels with zero or nodata values
    invalid_mask = (thermal == 0) | (thermal == thermal_src.nodata)
    thermal[invalid_mask] = np.nan
    
    # Constants for Landsat 8 Thermal Band (Band 10)
    ML = 0.0003342  # Radiance multiplicative scaling factor
    AL = 0.1        # Radiance additive scaling factor
    K1 = 774.8853   # Thermal conversion constant
    K2 = 1321.0789  # Thermal conversion constant

    # Calculate radiance
    radiance = ML * thermal + AL

    # Avoid negative or zero radiance values
    radiance[radiance <= 0] = np.nan

    # Calculate brightness temperature
    brightness_temp = K2 / np.log((K1 / radiance) + 1) - 273.15  # Convert to Celsius

    # Handle invalid values
    brightness_temp[~np.isfinite(brightness_temp)] = np.nan

    thermal_meta.update(dtype=rasterio.float32, count=1, nodata=np.nan)
    with rasterio.open(output_path, 'w', **thermal_meta) as dst:
        dst.write(brightness_temp.astype(rasterio.float32), 1)

    # Debug: Print LST statistics after cleaning
    print(f"LST - min: {np.nanmin(brightness_temp):.2f}°C, max: {np.nanmax(brightness_temp):.2f}°C, mean: {np.nanmean(brightness_temp):.2f}°C")

    return brightness_temp
'''

# Function to perform K-means clustering
def kmeans_clustering(lst, ndvi, n_clusters=5):
    lst_flat = lst.flatten()
    ndvi_flat = ndvi.flatten()
    valid_mask = np.isfinite(lst_flat) & np.isfinite(ndvi_flat)
    valid_lst = lst_flat[valid_mask]
    valid_ndvi = ndvi_flat[valid_mask]

    # Prepare features
    features = np.vstack((valid_lst, valid_ndvi)).T

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(features_scaled)

    # Create full-sized label image
    labels_image = np.full(lst.shape, -1, dtype=np.int32)
    labels_image.flat[valid_mask] = cluster_labels

    # Rescale cluster centers back to original feature ranges
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_info = {
        'labels': labels_image,
        'centers': cluster_centers,
        'inertia': kmeans.inertia_
    }

    # Identify UHI cluster
    uhi_cluster = identify_uhi_cluster(cluster_info, valid_lst, valid_ndvi, cluster_labels)

    # Create UHI mask
    uhi_mask = (labels_image == uhi_cluster).astype(np.uint8)

    return uhi_mask, cluster_info

# Function to identify the UHI cluster
def identify_uhi_cluster(cluster_info, valid_lst, valid_ndvi, cluster_labels):
    centers = cluster_info['centers']
    lst_centers = centers[:, 0]
    ndvi_centers = centers[:, 1]

    # Calculate cluster-level statistics
    cluster_temps = []
    for i in range(len(centers)):
        mask = (cluster_labels == i)
        mean_temp = np.mean(valid_lst[mask])
        mean_ndvi = np.mean(valid_ndvi[mask])
        std_temp = np.std(valid_lst[mask])  # Measure variability
        cluster_temps.append((i, mean_temp, mean_ndvi, std_temp))

    # Calculate UHI score (higher mean LST, lower mean NDVI, lower variability)
    uhi_scores = [
        (i, mean_temp - mean_ndvi - 0.1 * std_temp)  # Weighted scoring
        for i, mean_temp, mean_ndvi, std_temp in cluster_temps
    ]

    # Identify the cluster with the highest UHI score
    uhi_cluster = max(uhi_scores, key=lambda x: x[1])[0]

    # Debugging: Print cluster centers, scores, and selection
    print("Cluster Centers and Scores:")
    for i, (lst_c, ndvi_c, score) in enumerate(zip(lst_centers, ndvi_centers, [s[1] for s in uhi_scores])):
        print(f"Cluster {i}: LST Center = {lst_c:.2f}, NDVI Center = {ndvi_c:.2f}, Score = {score:.2f}")

    print(f"Identified UHI Cluster: {uhi_cluster}")
    return uhi_cluster

# Function to save plots
def save_plot(data, title, cmap, filename, vmin=None, vmax=None):
    plt.figure(figsize=(10, 8))
    plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
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

    # Perform K-means clustering
    n_clusters = 5  # You can adjust this number
    print("Performing K-means clustering...")
    uhi_mask, cluster_info = kmeans_clustering(lst, ndvi, n_clusters=n_clusters)
    save_plot(uhi_mask, 'UHI Detection via K-means', 'gray', uhi_plot_path)

    # Save the UHI mask to a GeoTIFF file
    with rasterio.open(thermal_band_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(uhi_output_path, 'w', **profile) as dst:
            dst.write(uhi_mask.astype(rasterio.uint8), 1)

    print(f"UHI mask saved to {uhi_output_path}")


    print("Processing complete!")

if __name__ == "__main__":
    main()
