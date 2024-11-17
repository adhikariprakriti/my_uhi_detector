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


def visualize_kmeans_with_osm(uhi_mask_path, osm_boundary_path, folium_map_output_path):
    # Read UHI mask raster
    with rasterio.open(uhi_mask_path) as src:
        uhi_mask = src.read(1)
        uhi_profile = src.profile
        uhi_transform = src.transform
        uhi_crs = src.crs

    # Define destination CRS (Web Mercator)
    dst_crs = 'EPSG:3857'

    # Calculate transformation for reprojecting UHI mask
    transform, width, height = calculate_default_transform(
        uhi_crs, dst_crs, src.width, src.height, *src.bounds
    )
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    # Reproject UHI mask to Web Mercator
    uhi_mask_reprojected = np.empty((height, width), dtype=np.int32)
    reproject(
        source=uhi_mask,
        destination=uhi_mask_reprojected,
        src_transform=uhi_transform,
        src_crs=uhi_crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.nearest
    )

    # Save reprojected UHI mask
    uhi_mask_reprojected_path = 'data/processed/kmeans/uhi_mask_reprojected.tif'
    os.makedirs(os.path.dirname(uhi_mask_reprojected_path), exist_ok=True)
    with rasterio.open(uhi_mask_reprojected_path, 'w', **kwargs) as dst:
        dst.write(uhi_mask_reprojected, 1)

    # Reproject OSM boundary to Web Mercator
    osm_boundary = gpd.read_file(osm_boundary_path)
    osm_boundary = osm_boundary.to_crs(dst_crs)

    # Calculate bounds in lat/lon (EPSG:4326)
    bounds = rasterio.transform.array_bounds(height, width, transform)
    minx, miny, maxx, maxy = bounds
    transformer = pyproj.Transformer.from_crs(dst_crs, 'EPSG:4326', always_xy=True)
    min_lon, min_lat = transformer.transform(minx, miny)
    max_lon, max_lat = transformer.transform(maxx, maxy)

    # Initialize Folium map at the center
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')

    # Create RGBA image for UHI mask
    rgba_image = np.zeros((height, width, 4), dtype=np.uint8)
    rgba_image[uhi_mask_reprojected == 0] = [0, 0, 255, 100]  # Blue for Non-UHI
    rgba_image[uhi_mask_reprojected == 1] = [255, 0, 0, 100]  # Red for UHI

    # Convert to PIL Image and save
    img = Image.fromarray(rgba_image, 'RGBA')
    uhi_mask_png = 'data/processed/kmeans/uhi_mask.png'
    img.save(uhi_mask_png)

    # Use absolute path for the image overlay
    uhi_mask_png_path = os.path.abspath(uhi_mask_png)

    # Add UHI mask overlay to Folium map
    folium.raster_layers.ImageOverlay(
        name='K-Means UHI Mask',
        image=uhi_mask_png_path,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=0.7,
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(folium_map)

    # Add OSM boundary
    folium.GeoJson(
        osm_boundary,
        name='OSM Boundary',
        style_function=lambda x: {
            'fillColor': 'none',
            'color': 'black',
            'weight': 2,
            'opacity': 1
        }
    ).add_to(folium_map)

    # Add legend for UHI mask
    template = """
    {% macro html(this, kwargs) %}
    <div style="position: fixed; bottom: 50px; left: 50px; width: 200px; height: 90px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color: white;">
        <b>K-Means UHI Legend</b><br>
        <i style="background:blue;opacity:0.7;width:12px;height:12px;display:inline-block;"></i> Non-UHI<br>
        <i style="background:red;opacity:0.7;width:12px;height:12px;display:inline-block;"></i> UHI
    </div>
    {% endmacro %}
    """
    macro = MacroElement()
    macro._template = Template(template)
    folium_map.get_root().add_child(macro)

    # Add layer control and save map
    folium.LayerControl().add_to(folium_map)
    folium_map.save(folium_map_output_path)
    print(f"Folium map with K-means UHI overlay saved to {folium_map_output_path}")


def visualize_with_folium(uhi_mask_path, folium_map_output_path):
    # Read the UHI mask
    with rasterio.open(uhi_mask_path) as src:
        uhi_mask = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        src_bounds = src.bounds

    # Reproject UHI mask to EPSG:3857 (Web Mercator)
    dst_crs = 'EPSG:3857'
    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    uhi_mask_reprojected = np.zeros((height, width), dtype=np.uint8)
    with rasterio.open('temp_uhi_reprojected.tif', 'w', **kwargs) as dst:
        reproject(
            source=uhi_mask,
            destination=uhi_mask_reprojected,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)

    # Get bounds in lat/lon for Folium
    left, bottom = rasterio.transform.xy(transform, height, 0, offset='ul')
    right, top = rasterio.transform.xy(transform, 0, width, offset='ur')
    bounds = [[bottom, left], [top, right]]

    # Create Folium map centered on the UHI mask
    center_lat = (bounds[0][0] + bounds[1][0]) / 2
    center_lon = (bounds[0][1] + bounds[1][1]) / 2
    folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='OpenStreetMap')

    # Prepare UHI mask overlay
    uhi_mask_rgba = np.zeros((uhi_mask_reprojected.shape[0], uhi_mask_reprojected.shape[1], 4), dtype=np.uint8)
    uhi_mask_rgba[..., 0] = np.where(uhi_mask_reprojected == 1, 255, 0)  # Red for UHI
    uhi_mask_rgba[..., 2] = np.where(uhi_mask_reprojected == 0, 255, 0)  # Blue for non-UHI
    uhi_mask_rgba[..., 3] = np.where(uhi_mask_reprojected >= 0, 100, 0)  # Alpha channel

    # Save overlay image
    uhi_mask_image_path = 'data/processed/kmeans/uhi_mask_overlay.png'
    Image.fromarray(uhi_mask_rgba).save(uhi_mask_image_path)

    # Overlay the UHI mask on the Folium map
    folium.raster_layers.ImageOverlay(
        name='UHI Detection',
        image=uhi_mask_image_path,
        bounds=bounds,
        opacity=0.6,
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(folium_map)

    # Add layer control
    folium.LayerControl().add_to(folium_map)

    # Save the map
    folium_map.save(folium_map_output_path)
    print(f"Interactive Folium map saved to {folium_map_output_path}")

    # Clean up temporary files
    os.remove('temp_uhi_reprojected.tif')
    os.remove(uhi_mask_image_path)

if __name__ == "__main__":
    results_dir = 'data/processed/kmeans/'
    visualizations_dir = 'visualization/kmeans/'
    uhi_results_dir = 'data/results/kmeans/'
    osm_boundary_path = 'data/osm/columbus_boundary.shp'

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
    visualize_kmeans_with_osm(uhi_mask_path, osm_boundary_path, folium_map_output_path)
    visualize_with_folium(uhi_mask_path, folium_map_output_path)