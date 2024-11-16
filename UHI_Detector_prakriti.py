## Import Libraries
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

from rasterio.plot import show
import folium
import os


# from google.colab import drive
# drive.mount('/content/drive')


project_path = '/content/drive/MyDrive/my uhi detector'
os.chdir(project_path)

# ## Step 1: Define Paths and Create Output Directories
# data_dir = f"{project_path}/data/landsat/"
# processed_dir = f"{project_path}/data/processed/normal"
# results_dir = f"{project_path}/data/results/normal"

data_dir = 'data/landsat/'
processed_dir = 'data/processed/normal'
results_dir = 'data/results/normal'

results_dir = 'data/results/normal/'
visualizations_dir = 'visualization/normal/'


os.makedirs(processed_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# File paths for input and output

red_band_path = os.path.join(data_dir, 'LC08_L1TP_020032_20240727_20240801_02_T1_B4.TIF')
nir_band_path = os.path.join(data_dir, 'LC08_L1TP_020032_20240727_20240801_02_T1_B5.TIF')
thermal_band_path = os.path.join(data_dir, 'LC08_L1TP_020032_20240727_20240801_02_T1_B10.TIF')
ndvi_output_path = os.path.join(processed_dir, 'ndvi.png')
lst_output_path = os.path.join(processed_dir, 'lst.png')
uhi_output_path = os.path.join(results_dir, 'uhi_detection.png')# ## Step 1: Define Paths and Create Output Directories


os.makedirs(processed_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# File paths for input and output

red_band_path = os.path.join(data_dir, 'LC08_L1TP_020032_20240727_20240801_02_T1_B4.TIF')
nir_band_path = os.path.join(data_dir, 'LC08_L1TP_020032_20240727_20240801_02_T1_B5.TIF')
thermal_band_path = os.path.join(data_dir, 'LC08_L1TP_020032_20240727_20240801_02_T1_B10.TIF')
ndvi_output_path = os.path.join(processed_dir, 'ndvi.tif')
lst_output_path = os.path.join(processed_dir, 'lst.tif')
uhi_output_path = os.path.join(results_dir, 'uhi_detection.tif')


# ## Step 2: Calculate NDVI
def calculate_ndvi(red_band_path, nir_band_path, output_path):
    with rasterio.open(red_band_path) as red_src:
        red = red_src.read(1).astype(float)
        profile = red_src.profile

    with rasterio.open(nir_band_path) as nir_src:
        nir = nir_src.read(1).astype(float)

    epsilon = 1e-10
    ndvi = (nir - red) / (nir + red + epsilon)
    ndvi = np.clip(ndvi, -1, 1)  # Clip values to valid NDVI range
    ndvi[~np.isfinite(ndvi)] = 0  # Handle any remaining invalid values

    # Save NDVI to GeoTIFF
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(ndvi.astype(rasterio.float32), 1)

    return ndvi

# Calculate and display NDVI
ndvi = calculate_ndvi(red_band_path, nir_band_path, ndvi_output_path)
plt.imshow(ndvi, cmap='RdYlGn')
plt.title('NDVI Map')
plt.colorbar()
plt.show()


# ## Step 3: Calculate Land Surface Temperature (LST)
def calculate_lst(thermal_band_path, output_path):
    with rasterio.open(thermal_band_path) as thermal_src:
        thermal = thermal_src.read(1).astype(float) * 0.1  # Convert to Kelvin
        profile = thermal_src.profile

    # Convert LST to Celsius
    lst = thermal - 273.15

    # Save LST to GeoTIFF
    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(lst.astype(rasterio.float32), 1)

    return lst

# Calculate and display LST
lst = calculate_lst(thermal_band_path, lst_output_path)
plt.imshow(lst, cmap='hot')
plt.title('Land Surface Temperature (LST)')
plt.colorbar()
plt.show()

print("LST shape:", lst.shape)
print("NDVI shape:", ndvi.shape)

print(ndvi_output_path)



def read_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Read the first band
    return data

# Paths to your NDVI and LST images
ndvi_path = f"{project_path}/{ndvi_output_path}"
lst_path = f"{project_path}/{lst_output_path}"
print(ndvi_path)
print(lst_path)

# Read the rasters
ndvi = read_raster(ndvi_path)
lst = read_raster(lst_path)

# You can adjust this threshold according to your analysis needs
ndvi_threshold = 0.2  # Common threshold to identify vegetated areas
hotspots = (ndvi < ndvi_threshold) & (lst > np.percentile(lst, 90))  # Hot areas with low vegetation

# Plotting the UHI effect based on LST where NDVI is low
plt.figure(figsize=(10, 6))
plt.title('Urban Heat Island (UHI) Effect')
# plt.imshow(lst, cmap='hot', interpolation='none')
plt.imshow(lst, cmap='plasma', vmin=0, vmax=3500, interpolation='none')

plt.colorbar(label='Temperature')
plt.contour(hotspots, colors='pink', linewidths=1)  # Contour for hotspots
plt.grid(False)
plt.show()


# Convert hotspots to float32 for saving in TIFF format
hotspots = hotspots.astype('float32')

# Set up profile for the new GeoTIFF
# new_profile = lst.profile
# new_profile.update(
#     dtype=rasterio.float32,
#     count=1,
#     compress='lzw'
# )

with rasterio.open(lst_path) as lst_src:
        lst = lst_src.read(1).astype(float) * 0.1  # Convert to Kelvin
        new_profile = lst_src.profile


# Path to save the new GeoTIFF
output_tif_path = f"{project_path}/data/results/hotspots.tif"

# Writing the data
with rasterio.open(output_tif_path, 'w', **new_profile) as dst:
    dst.write(hotspots, 1)

print(f"Hotspots TIFF saved to {output_tif_path}")


# Path to your UHI TIFF file
tif_path = f"{project_path}/data/results/hotspots.tif"

# Open the TIFF file using rasterio
with rasterio.open(tif_path) as src:
    bounds = src.bounds
    data = src.read(1)  # Read the first band for plotting

    # Calculate the center of the map
    lon = (bounds.left + bounds.right) / 2
    lat = (bounds.bottom + bounds.top) / 2

# Create a folium map at the calculated center
m = folium.Map(location=[lat, lon], zoom_start=11)

# Overlay the TIFF using folium's raster_layers
folium.raster_layers.ImageOverlay(
    image=data,
    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
    colormap=lambda x: (x, 0, 1-x, x),  # You can customize the colormap as needed
    name='UHI Effect'
).add_to(m)

# Add layer control to toggle layers
folium.LayerControl().add_to(m)

# Display the map
m.save('UHI_Map.html')
m

