import requests
import json

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
    city = "Columbus"
    state = "Ohio"
    filename = "columbus_ohio.geojson"
    
    geojson_data = get_geojson(city, state)
    save_geojson(geojson_data, filename)
    print(f"GeoJSON data for {city}, {state} saved to {filename}")