import requests
import json
import geopandas as gpd
from shapely.geometry import shape

def get_geojson(city, state):
    url = f"https://nominatim.openstreetmap.org/search?city={city}&state={state}&format=geojson"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error fetching data: {response.status_code}")

def save_geojson(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    city = "Dayton"
    state = "Ohio"
    filename = "dayton_ohio.geojson"
    
    geojson_data = get_geojson(city, state)
    
    # Save GeoJSON data
    save_geojson(geojson_data, filename)
    
    # Convert GeoJSON to Shapefile

    gdf = gpd.GeoDataFrame.from_features(geojson_data["features"])
    shp_filename = filename.replace(".geojson", ".shp")
    gdf.to_file(shp_filename)
    
    print(f"Shapefile data for {city}, {state} saved to {shp_filename}")
    save_geojson(geojson_data, filename)
    print(f"GeoJSON data for {city}, {state} saved to {filename}")