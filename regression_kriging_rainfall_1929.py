import geopandas as gpd
import rasterio
import numpy as np
from rasterio.features import geometry_mask, rasterize
from scipy.spatial import cKDTree
from pykrige.rk import RegressionKriging
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

print("Loading data...")

# Set your base path based on the system you are using
base_path = "/home/filbra/Desktop/QGIS Project/RhECAST/Model/"

# Load point data for precipitation in 1929
ws_1929 = gpd.read_file(f"{base_path}Output/gpkg/w_s_1929.gpkg")

# Load computational region (ROI)
roi = gpd.read_file(f"{base_path}Input/gpkg/roi/italy_roi.gpkg")

# Load the coastline data
coast = gpd.read_file(f"{base_path}Input/gpkg/roi/coastline.gpkg")

# Load the DEM as a raster dataset
dem = rasterio.open(f"{base_path}Input/gtiff/italy_dem_200.tif")

# Read DEM data as a numpy array
dem_data = dem.read(1)

# Calculate the gradient in the x and y directions
dx, dy = np.gradient(dem_data, dem.transform[0], dem.transform[4])

# Calculate slope in degrees
slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180.0 / np.pi)

# Calculate aspect in degrees (0-360)
aspect = np.arctan2(-dy, dx) * (180.0 / np.pi)
aspect = np.where(aspect < 0, 360.0 + aspect, aspect)

# Clip slope and aspect by ROI using geometry_mask
roi_mask = geometry_mask(roi.geometry, transform=dem.transform, invert=True, out_shape=dem_data.shape)
slope_clipped = np.where(roi_mask, slope, np.nan)
aspect_clipped = np.where(roi_mask, aspect, np.nan)

# Save the clipped slope and aspect rasters (optional)
slope_meta = dem.meta.copy()
slope_meta.update({"driver": "GTiff", "dtype": "float32"})
with rasterio.open(f"{base_path}Output/images/slope.tif", "w", **slope_meta) as dest:
    dest.write(slope_clipped.astype(np.float32), 1)

aspect_meta = dem.meta.copy()
aspect_meta.update({"driver": "GTiff", "dtype": "float32"})
with rasterio.open(f"{base_path}Output/images/aspect.tif", "w", **aspect_meta) as dest:
    dest.write(aspect_clipped.astype(np.float32), 1)

print("Slope and aspect calculation and export complete.")

# Convert the coastline layer to a raster at the same resolution as the DEM
out_shape = dem.shape
transform = dem.transform

# Rasterize the coastline
coast_mask = rasterize(
    [(geom, 1) for geom in coast.geometry if geom.is_valid],
    out_shape=out_shape,
    transform=transform,
    fill=0,
    all_touched=True,
    dtype='int32'
)

# Find indices where the coastline is present
coastline_indices = np.column_stack(np.where(coast_mask == 1))

# Create a KDTree for fast distance computation
tree = cKDTree(coastline_indices)

# Get the indices of each pixel
pixel_indices = np.indices(coast_mask.shape).reshape(2, -1).T

# Calculate the Euclidean distance from each pixel to the nearest coastline point in pixels
distances_pixels = tree.query(pixel_indices)[0].reshape(coast_mask.shape)

# Convert pixel distances to meters using the DEM's transform
pixel_size = dem.transform[0]  # The pixel size in meters (assuming square pixels)

# Convert distances from pixels to meters
distances_meters = distances_pixels * pixel_size

# Save the distance raster in meters to a file (optional)
out_meta = dem.meta.copy()
out_meta.update({"driver": "GTiff", "dtype": "float32"})

with rasterio.open(f"{base_path}Output/images/coastline_eu_dist.tif", "w", **out_meta) as dest:
    dest.write(distances_meters.astype(np.float32), 1)

print("Euclidean distance calculation and export in meters complete.")

# Ensure that ws_1929 points are in the same CRS as the DEM
if ws_1929.crs != dem.crs:
    ws_1929 = ws_1929.to_crs(dem.crs)

# Function to calculate the row, col index from geometry
def get_raster_index(geom, transform):
    x, y = geom.x, geom.y
    row, col = ~transform * (x, y)
    return int(row), int(col)

# Sample DEM values for ws_1929 points
def sample_raster_values(geom, raster_data, transform):
    row, col = get_raster_index(geom, transform)
    if 0 <= row < raster_data.shape[0] and 0 <= col < raster_data.shape[1]:
        return raster_data[row, col]
    else:
        return np.nan  # Return NaN if out of bounds

ws_1929['dem'] = ws_1929.geometry.apply(lambda geom: sample_raster_values(geom, dem_data, dem.transform))

# Calculate the Euclidean distance from the coastline for each point in ws_1929
ws_1929['dist_to_coast'] = ws_1929.geometry.apply(lambda geom: sample_raster_values(geom, distances_meters, dem.transform))

# Calculate slope and aspect for ws_1929 points
ws_1929['slope'] = ws_1929.geometry.apply(lambda geom: sample_raster_values(geom, slope_clipped, dem.transform))
ws_1929['aspect'] = ws_1929.geometry.apply(lambda geom: sample_raster_values(geom, aspect_clipped, dem.transform))

# Extract TOT_mm values from the ws_1929 points
y = ws_1929['TOT_mm'].values

# Prepare the feature matrix (DEM, distance to coastline, slope, and aspect)
X = np.vstack([ws_1929['dem'], ws_1929['dist_to_coast'], ws_1929['slope'], ws_1929['aspect']]).T

# Remove rows with NaN values from X and y
mask = ~np.isnan(X).any(axis=1)
X = X[mask]
y = y[mask]

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert coordinates to a NumPy array
coordinates = np.array(ws_1929.geometry.apply(lambda geom: (geom.x, geom.y)).tolist())

# Apply the same mask to the coordinates
coordinates = coordinates[mask]

# Set up K-Fold cross-validation (using 5 folds as an example)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store validation metrics
val_mae_list = []
val_mse_list = []
val_r2_list = []

# Perform K-Fold Cross-Validation
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    coords_train, coords_test = coordinates[train_index], coordinates[test_index]
    
    # Linear regression model
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    
    # Perform regression kriging
    rk = RegressionKriging(regression_model=lm, n_closest_points=10)
    rk.fit(X_train, coords_train, y_train)
    
    # Predict using regression kriging on the validation data
    test_predicted = rk.predict(X_test, coords_test)
    
    # Calculate and store validation metrics
    val_mae_list.append(mean_absolute_error(y_test, test_predicted))
    val_mse_list.append(mean_squared_error(y_test, test_predicted))
    val_r2_list.append(r2_score(y_test, test_predicted))

# Calculate average validation metrics across all folds
avg_val_mae = np.mean(val_mae_list)
avg_val_mse = np.mean(val_mse_list)
avg_val_r2 = np.mean(val_r2_list)

print(f"Cross-Validation Mean Absolute Error: {avg_val_mae}")
print(f"Cross-Validation Mean Squared Error: {avg_val_mse}")
print(f"Cross-Validation R^2 Score: {avg_val_r2}")

# Continue with your grid predictions and other operations as in your original code...

# Mask the DEM and distances using the ROI
masked_dem = np.where(roi_mask, dem_data, np.nan)
masked_distances = np.where(roi_mask, distances_meters, np.nan)

# Generate a grid of coordinates corresponding to the DEM resolution within the ROI
rows, cols = np.indices(dem.shape)
coords = np.column_stack([cols.flatten(), rows.flatten()])
xs, ys = rasterio.transform.xy(dem.transform, coords[:, 1], coords[:, 0])
grid_coords = np.column_stack([xs, ys])

# Mask the grid coordinates with the ROI
valid_grid_coords = grid_coords[roi_mask.flatten()]

# Predict values for the entire grid within the ROI
valid_dem_values = masked_dem[roi_mask].reshape(-1, 1)
valid_distances_values = masked_distances[roi_mask].reshape(-1, 1)
valid_slope_values = slope_clipped[roi_mask].reshape(-1, 1)
valid_aspect_values = aspect_clipped[roi_mask].reshape(-1, 1)

valid_X_grid = np.hstack([valid_dem_values, valid_distances_values, valid_slope_values, valid_aspect_values])

# Impute NaN values with the mean of each column in valid_X_grid
imputer = SimpleImputer(strategy="mean")
valid_X_grid_scaled = imputer.fit_transform(valid_X_grid)

# Normalize the grid features using the scaler fitted earlier
valid_X_grid_scaled = scaler.transform(valid_X_grid_scaled)

# Perform predictions for the grid
grid_predictions = rk.predict(valid_X_grid_scaled, valid_grid_coords)

# Initialize a new array for the predicted values, filled with NaNs initially
predicted_raster = np.full(dem.shape, np.nan, dtype=np.float32)

# Place the predictions into the raster at the valid grid points
predicted_raster[roi_mask] = grid_predictions

# Export the raster to a GeoTIFF file
out_meta.update({
    "dtype": "float32",
    "count": 1,  # Single band
})

with rasterio.open(f"{base_path}Output/images/kriging_predictions.tif", "w", **out_meta) as dest:
    dest.write(predicted_raster, 1)

print("Kriging prediction raster exported successfully.")
