import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping
from folium import raster_layers
from branca.element import Template, MacroElement
from PIL import Image
import pyproj
from pyproj import Transformer
from matplotlib import cm
from matplotlib.colors import Normalize


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


def overlay_uhi_on_osm(uhi_mask_path, folium_map_output_path):
    # Read the UHI mask
    with rasterio.open(uhi_mask_path) as src:
        uhi_mask = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        bounds = src.bounds

    # Handle invalid values
    uhi_mask = uhi_mask.astype(float)
    uhi_mask[uhi_mask == src.nodata] = np.nan

    # Get the bounds in lat/lon (WGS84)
    transformer = Transformer.from_crs(src_crs, 'EPSG:4326', always_xy=True)
    left, bottom = transformer.transform(bounds.left, bounds.bottom)
    right, top = transformer.transform(bounds.right, bounds.top)
    bounds_latlon = [[bottom, left], [top, right]]

    # Calculate the center of the map
    center_lat = (bottom + top) / 2
    center_lon = (left + right) / 2

    # Mask NaN values
    masked_data = np.ma.masked_where(np.isnan(uhi_mask), uhi_mask)

    # Normalize data for visualization
    norm = Normalize(vmin=0, vmax=1)

    # Apply colormap
    cmap = cm.get_cmap('RdBu_r')
    colored_data = cmap(norm(masked_data))

    # Convert to 8-bit unsigned integers
    colored_data = (colored_data * 255).astype(np.uint8)

    # Create an image from the array
    image = Image.fromarray(colored_data)
    image_path = 'uhi_overlay.png'
    image.save(image_path)

    # Create a Folium map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')

    # Add the image overlay
    folium.raster_layers.ImageOverlay(
        name='UHI Detection',
        image=image_path,
        bounds=bounds_latlon,
        opacity=0.6,
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(m)

    # Add layer control
    folium.LayerControl().add_to(m)

    # Save the map to an HTML file
    m.save(folium_map_output_path)
    print(f"Map has been saved to {folium_map_output_path}")

    # Optionally, remove the image file if not needed
    os.remove(image_path)

def normalize(array):
    """Normalize the array for colormap application."""
    array_min, array_max = np.nanmin(array), np.nanmax(array)
    denom = array_max - array_min
    if denom == 0 or np.isnan(denom):
        return np.zeros_like(array)
    return (array - array_min) / denom

if __name__ == "__main__":
    results_dir = 'data/processed/kmeans/'
    visualizations_dir = 'visualization/kmeans/'
    uhi_results_dir = 'data/results/kmeans/'
    osm_boundary_path = 'data/osm/dayton_ohio.shp'

    ndvi_path = os.path.join(results_dir, 'ndvi.tif')
    lst_path = os.path.join(results_dir, 'lst.tif')
    uhi_mask_path = os.path.join(uhi_results_dir, 'uhi_kmeans.tif')

    ndvi_plot_path = os.path.join(visualizations_dir, 'ndvi_map.png')
    lst_plot_path = os.path.join(visualizations_dir, 'lst_map.png')
    uhi_plot_path = os.path.join(visualizations_dir, 'uhi_detection.png')
    folium_map_output_path = os.path.join(visualizations_dir, 'uhi_kmeans_map.html')

    # Create visualizations
    visualize_ndvi(ndvi_path, ndvi_plot_path)
    visualize_lst(lst_path, lst_plot_path)
    visualize_uhi(uhi_mask_path, uhi_plot_path)

    # Create interactive Folium map
    overlay_uhi_on_osm(uhi_mask_path, folium_map_output_path)