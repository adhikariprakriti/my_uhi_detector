import requests
import geopandas as gpd
import os
import osmnx as ox

def get_state_boundary(state_name):
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    relation
      ["admin_level"="4"]
      ["name"="{state_name}"]
      ["boundary"="administrative"];
    (._;>;);
    out body;
    """
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    return data

def save_state_shapefile(state_name, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    data = get_state_boundary(state_name)
    
    # Convert OSM data to GeoDataFrame
    gdf = ox.graph_from_place(state_name, network_type='all')
    
    # Define the output file path
    shp_filename = os.path.join(output_directory, f"{state_name}_boundary.shp")
    gdf.to_file(shp_filename)
    print(f"Shapefile saved to {shp_filename}")

if __name__ == "__main__":
    state = "Ohio"
    osm_path = "./data/osm/"
    
    save_state_shapefile(state, osm_path)