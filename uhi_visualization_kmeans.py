import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
from shapely.geometry import mapping
from rasterio.mask import mask

def visualize_ndvi(ndvi_path, output_path):
    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1)
        ndvi[ndvi == src.nodata] = np.nan

    plt.figure(figsize=(10, 8))
    plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(label='NDVI')
    plt.title('NDVI')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"NDVI map saved to {output_path}")

def visualize_lst(lst_path, output_path):
    with rasterio.open(lst_path) as src:
        lst = src.read(1)
        lst[lst == src.nodata] = np.nan

    plt.figure(figsize=(10, 8))
    plt.imshow(lst, cmap='hot')
    plt.colorbar(label='Temperature (°C)')
    plt.title('Land Surface Temperature')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"LST map saved to {output_path}")

def visualize_uhi(uhi_mask_path, output_path, title='UHI Detection via K-means'):
    with rasterio.open(uhi_mask_path) as src:
        uhi_mask = src.read(1).astype(float)  # Convert to float to allow NaN assignments
        uhi_mask[uhi_mask == src.nodata] = np.nan  # Replace nodata values with NaN

    plt.figure(figsize=(10, 8))
    plt.imshow(uhi_mask, cmap='gray')
    plt.colorbar(label='UHI Mask')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"UHI detection map saved to {output_path}")


if __name__ == "__main__":
    results_dir = 'data/processed/kmeans/'
    visualizations_dir = 'visualization/kmeans/'
    uhi_results_dir = 'data/results/kmeans/'


    ndvi_path = os.path.join(results_dir, 'ndvi.tif')
    lst_path = os.path.join(results_dir, 'lst.tif')
    uhi_mask_path = os.path.join(uhi_results_dir, 'uhi_kmeans.tif')

    ndvi_plot_path = os.path.join(visualizations_dir, 'ndvi_map.png')
    lst_plot_path = os.path.join(visualizations_dir, 'lst_map.png')
    uhi_plot_path = os.path.join(visualizations_dir, 'uhi_detection.png')

    # Create visualizations
    visualize_ndvi(ndvi_path, ndvi_plot_path)
    visualize_lst(lst_path, lst_plot_path)
    visualize_uhi(uhi_mask_path, uhi_plot_path)
