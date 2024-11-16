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

    plt.figure(figsize=(10, 10))
    cmap = plt.cm.YlGn  # Match Random Forest colormap
    cmap.set_under(color='white')
    im = plt.imshow(ndvi, cmap=cmap, vmin=-1, vmax=1)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('NDVI')
    plt.title('Normalized Difference Vegetation Index (NDVI)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"NDVI map saved to {output_path}")


def visualize_lst(lst_path, output_path):
    with rasterio.open(lst_path) as src:
        lst = src.read(1)
    plt.figure(figsize=(10, 10))
    cmap = plt.cm.jet
    cmap.set_bad(color='white')
    im = plt.imshow(lst, cmap=cmap)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (°C)')
    plt.title('Land Surface Temperature (LST)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"LST map saved to {output_path}")


def visualize_uhi(uhi_mask_path, output_path, title='UHI Detection via K-means'):
    with rasterio.open(uhi_mask_path) as src:
        uhi_mask = src.read(1).astype(float) 
    plt.figure(figsize=(10, 10))
    cmap = plt.cm.RdBu_r 
    cmap.set_bad(color='white')
    masked_data = np.ma.masked_where(np.isnan(uhi_mask), uhi_mask)
    im = plt.imshow(masked_data, cmap=cmap)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Non-UHI', 'UHI'])
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
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
