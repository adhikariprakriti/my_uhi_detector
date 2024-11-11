import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import folium
from folium import raster_layers
from folium.plugins import FloatImage
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import box
from PIL import Image
import matplotlib.colors as mcolors
from branca.element import Template, MacroElement
import pyproj
import rasterio.features  # Added for geometry_mask

def visualize_ndvi(ndvi_path, output_path):
    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1)
    plt.figure(figsize=(10, 10))
    cmap = plt.cm.YlGn
    cmap.set_under(color='white')
    im = plt.imshow(ndvi, cmap=cmap, vmin=-1, vmax=1)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('NDVI')
    plt.title('Normalized Difference Vegetation Index (NDVI)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"NDVI visualization saved to {output_path}")

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
    print(f"LST visualization saved to {output_path}")

def visualize_uhi(uhi_mask_path, output_path, title='UHI Detection'):
    with rasterio.open(uhi_mask_path) as src:
        uhi_mask = src.read(1)
    plt.figure(figsize=(10, 10))
    cmap = plt.cm.RdBu_r
    cmap.set_bad(color='white')
    masked_data = np.ma.masked_where(uhi_mask == -1, uhi_mask)
    im = plt.imshow(masked_data, cmap=cmap)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Non-UHI', 'UHI'])
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"UHI mask visualization saved to {output_path}")

def visualize_with_folium(uhi_mask_path, building_geojson_path, folium_map_output_path):
    # Read UHI mask raster
    with rasterio.open(uhi_mask_path) as src:
        uhi_mask = src.read(1)
        uhi_profile = src.profile
        uhi_transform = src.transform
        uhi_crs = src.crs

    # Define destination CRS (Web Mercator)
    dst_crs = 'EPSG:3857'
    
    # Calculate transformation
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
    
    # Reproject UHI mask
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
    uhi_mask_reprojected_path = 'data/processed/random_forest/uhi_mask_reprojected.tif'
    os.makedirs(os.path.dirname(uhi_mask_reprojected_path), exist_ok=True)
    with rasterio.open(uhi_mask_reprojected_path, 'w', **kwargs) as dst:
        dst.write(uhi_mask_reprojected, 1)
    
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
    uhi_mask_png = 'data/processed/random_forest/uhi_mask.png'
    img.save(uhi_mask_png)

    # Use absolute path for the image overlay
    uhi_mask_png_path = os.path.abspath(uhi_mask_png)

    # Add UHI mask overlay to Folium map
    folium.raster_layers.ImageOverlay(
        name='UHI Mask',
        image=uhi_mask_png_path,
        bounds=[[min_lat, min_lon], [max_lat, max_lon]],
        opacity=0.7,
        interactive=True,
        cross_origin=False,
        zindex=1
    ).add_to(folium_map)

    # Read building footprints from GeoJSON
    buildings = gpd.read_file(building_geojson_path)
    buildings = buildings.to_crs('EPSG:4326')
    buildings_geojson = buildings.__geo_interface__
    
    folium.GeoJson(
        buildings_geojson,
        name='Buildings',
        style_function=lambda x: {
            'fillColor': 'gray',
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.5
        },
        tooltip=folium.GeoJsonTooltip(fields=[], aliases=[], localize=True)
    ).add_to(folium_map)
    
    # Add legend
    template = """
    {% macro html(this, kwargs) %}
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 150px; height: 90px; 
        border:2px solid grey; z-index:9999; font-size:14px;
        background-color: white;
        ">
        &nbsp;<b>UHI Mask Legend</b><br>
        &nbsp;<i class="fa fa-square" style="color:blue"></i>&nbsp;Non-UHI<br>
        &nbsp;<i class="fa fa-square" style="color:red"></i>&nbsp;UHI
    </div>
    {% endmacro %}
    """
    macro = MacroElement()
    macro._template = Template(template)
    folium_map.get_root().add_child(macro)
    
    # Add layer control and save map
    folium.LayerControl().add_to(folium_map)
    folium_map.save(folium_map_output_path)
    print(f"Folium map saved to {folium_map_output_path}")


def analyze_buildings_uhi(uhi_mask_path, building_geojson_path, analysis_output_path):
    with rasterio.open(uhi_mask_path) as src:
        uhi_mask = src.read(1)
        uhi_transform = src.transform
        uhi_crs = src.crs
    
    buildings = gpd.read_file(building_geojson_path)
    buildings = buildings.to_crs(uhi_crs)
    uhi_area = uhi_mask == 1
    buildings_in_uhi = 0
    total_buildings = len(buildings)
    print("Analyzing building overlaps with UHI areas...")
    for idx, building in buildings.iterrows():
        building_geom = building.geometry
        if building_geom is None or building_geom.is_empty:
            continue
        building_mask = rasterio.features.geometry_mask(
            [building_geom],
            out_shape=uhi_mask.shape,
            transform=uhi_transform,
            invert=True
        )
        if np.any(uhi_area & building_mask):
            buildings_in_uhi += 1
        if (idx + 1) % 500 == 0:
            print(f"Processed {idx + 1}/{total_buildings} buildings...")
    percentage = (buildings_in_uhi / total_buildings) * 100 if total_buildings > 0 else 0
    with open(analysis_output_path, 'w') as f:
        f.write(f"Total Buildings: {total_buildings}\n")
        f.write(f"Buildings in UHI Areas: {buildings_in_uhi}\n")
        f.write(f"Percentage of Buildings in UHI Areas: {percentage:.2f}%\n")
    print(f"UHI and Buildings Analysis saved to {analysis_output_path}")
    print(f"Percentage of Buildings in UHI Areas: {percentage:.2f}%")

def main():
    data_dir = 'data/landsat/'
    processed_dir = 'data/processed/random_forest/'
    results_dir = 'data/results/random_forest/'
    visualizations_dir = 'visualization/random_forest/'
    building_geojson_path = 'data/buildings/buildings_columbus.geojson' 
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    os.makedirs(os.path.dirname(building_geojson_path), exist_ok=True)
    
    ndvi_path = os.path.join(processed_dir, 'ndvi.tif')
    lst_path = os.path.join(processed_dir, 'lst.tif')
    uhi_mask_path = os.path.join(results_dir, 'uhi_detection_rf.tif')
    
    ndvi_visual_path = os.path.join(visualizations_dir, 'ndvi.png')
    lst_visual_path = os.path.join(visualizations_dir, 'lst.png')
    uhi_visual_path = os.path.join(visualizations_dir, 'uhi_detection_rf.png')
    folium_map_output = os.path.join(visualizations_dir, 'uhi_detection_map.html')
    analysis_output_path = os.path.join(results_dir, 'buildings_uhi_analysis.txt')
    
    print("Visualizing NDVI...")
    visualize_ndvi(ndvi_path, ndvi_visual_path)
    
    print("Visualizing LST...")
    visualize_lst(lst_path, lst_visual_path)
    
    print("Visualizing UHI Mask...")
    visualize_uhi(uhi_mask_path, uhi_visual_path, title='Urban Heat Island Detection (Random Forest)')
    
    print("Creating interactive Folium map...")
    visualize_with_folium(uhi_mask_path, building_geojson_path, folium_map_output)

    print("Visualization completed")
    
    # print("Analyzing relationship between buildings and UHI...")
    # analyze_buildings_uhi(uhi_mask_path, building_geojson_path, analysis_output_path)
    
    # print("\nVisualization and Analysis Completed.")

if __name__ == "__main__":
    main()
