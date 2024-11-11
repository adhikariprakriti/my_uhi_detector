import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def calculate_ndvi(red_band_path, nir_band_path, output_path):
    with rasterio.open(red_band_path) as red_src:
        red = red_src.read(1).astype(float)
        profile = red_src.profile

    with rasterio.open(nir_band_path) as nir_src:
        nir = nir_src.read(1).astype(float)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    ndvi = (nir - red) / (nir + red + epsilon)
    ndvi = np.clip(ndvi, -1, 1)
    ndvi[~np.isfinite(ndvi)] = 0

    profile.update(dtype=rasterio.float32, count=1)
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(ndvi.astype(rasterio.float32), 1)

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

def random_forest_uhi_detection(lst, ndvi):
    """
    Implements UHI detection using Random Forest algorithm
    """
    def prepare_features(lst, ndvi):
        # Flatten arrays
        lst_flat = lst.flatten()
        ndvi_flat = ndvi.flatten()
        
        # Create valid mask
        valid_mask = (
            np.isfinite(lst_flat) & 
            np.isfinite(ndvi_flat) & 
            (lst_flat != 0)
        )
        
        # Extract valid data points
        X = np.column_stack((
            lst_flat[valid_mask],
            ndvi_flat[valid_mask]
        ))
        
        return X, valid_mask

    def create_labels(X):
        # Create labels using temperature threshold
        lst_values = X[:, 0]
        lst_mean = np.mean(lst_values)
        lst_std = np.std(lst_values)
        
        # Define UHI thresholds
        uhi_threshold = lst_mean + lst_std
        
        # Create labels (1 for UHI, 0 for non-UHI)
        y = (lst_values >= uhi_threshold).astype(int)
        
        return y, uhi_threshold

    # Prepare data
    print("Preparing features...")
    X, valid_mask = prepare_features(lst, ndvi)
    y, uhi_threshold = create_labels(X)
    
    # Split data for training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train Random Forest
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    predictions = rf_model.predict(X)
    
    # Create output mask
    uhi_mask = np.full(lst.shape, -1, dtype=np.int32)
    uhi_mask.flat[valid_mask] = predictions
    
    # Calculate feature importance
    feature_importance = {
        'LST': rf_model.feature_importances_[0],
        'NDVI': rf_model.feature_importances_[1]
    }
    
    results = {
        'uhi_mask': uhi_mask,
        'model': rf_model,
        'threshold': uhi_threshold,
        'feature_importance': feature_importance,
        'accuracy': rf_model.score(X_test, y_test)
    }
    
    return results

def visualize_results(uhi_mask, output_path, title='UHI Detection'):
    """
    Visualize UHI detection results
    """
    plt.figure(figsize=(10, 10))  # Square figure to maintain aspect ratio
    
    # Create custom colormap
    cmap = plt.cm.RdBu_r
    cmap.set_bad(color='white')  # Set color for masked areas
    
    # Plot the UHI mask
    masked_data = np.ma.masked_where(uhi_mask == -1, uhi_mask)
    im = plt.imshow(masked_data, cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Non-UHI', 'UHI'])
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_ndvi(ndvi, output_path):
    """
    Visualize NDVI
    """
    plt.figure(figsize=(10, 10))  # Square figure to maintain aspect ratio
    cmap = plt.cm.YlGn
    cmap.set_under(color='white')  # For low NDVI values

    im = plt.imshow(ndvi, cmap=cmap, vmin=-1, vmax=1)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('NDVI')
    
    plt.title('Normalized Difference Vegetation Index (NDVI)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def visualize_lst(lst, output_path):
    """
    Visualize Land Surface Temperature (LST)
    """
    plt.figure(figsize=(10, 10))  # Square figure to maintain aspect ratio
    cmap = plt.cm.jet
    cmap.set_bad(color='white')  # For NaN values

    im = plt.imshow(lst, cmap=cmap)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (°C)')
    
    plt.title('Land Surface Temperature (LST)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Directory setup (keep your existing paths)
    data_dir = 'data/landsat/'
    processed_dir = 'data/processed/random_forest'
    results_dir = 'data/results/random_forest/'
    visualizations_dir = 'visualization/random_forest/'
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # File paths (keep your existing paths)
    red_band_path = os.path.join(data_dir, 'LC08_L1TP_020032_20240727_20240801_02_T1_B4.TIF')
    nir_band_path = os.path.join(data_dir, 'LC08_L1TP_020032_20240727_20240801_02_T1_B5.TIF')
    thermal_band_path = os.path.join(data_dir, 'LC08_L1TP_020032_20240727_20240801_02_T1_B10.TIF')
    
    # Calculate NDVI and LST
    print("Calculating NDVI...")
    ndvi_output_path = os.path.join(processed_dir, 'ndvi.tif')
    ndvi = calculate_ndvi(red_band_path, nir_band_path, ndvi_output_path)
    
    print("Calculating LST...")
    lst_output_path = os.path.join(processed_dir, 'lst.tif')
    lst = calculate_lst(thermal_band_path, lst_output_path)
    
    # Perform UHI detection
    print("Performing UHI detection...")
    results = random_forest_uhi_detection(lst, ndvi)
    
    # Save UHI mask
    uhi_output_path = os.path.join(results_dir, 'uhi_detection_rf.tif')
    with rasterio.open(thermal_band_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.int32, count=1)
        
        with rasterio.open(uhi_output_path, 'w', **profile) as dst:
            dst.write(results['uhi_mask'].astype(rasterio.int32), 1)
    
    # Visualize and save results
    print("Visualizing results...")
    uhi_visual_path = os.path.join(visualizations_dir, 'uhi_detection_rf.png')
    visualize_results(results['uhi_mask'], uhi_visual_path, title='Urban Heat Island Detection (Random Forest)')
    
    ndvi_visual_path = os.path.join(visualizations_dir, 'ndvi.png')
    visualize_ndvi(ndvi, ndvi_visual_path)
    
    lst_visual_path = os.path.join(visualizations_dir, 'lst.png')
    visualize_lst(lst, lst_visual_path)
    
    # Print results
    print("\nResults:")
    print(f"Model Accuracy: {results['accuracy']:.2f}")
    print("\nFeature Importance:")
    for feature, importance in results['feature_importance'].items():
        print(f"{feature}: {importance:.3f}")
    print(f"\nUHI Temperature Threshold: {results['threshold']:.2f}°C")

if __name__ == "__main__":
    main()
