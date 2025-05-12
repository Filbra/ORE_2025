import geopandas as gpd
import pandas as pd
from pandas.api.types import CategoricalDtype

## Set the base path for file operations

###UBUNTU
base_path = "/home/filbra/Desktop/QGIS Project/RhECAST/Model/"

##Windows
##base_path = "C:/Users/filbr/Desktop/QGIS Projects_rog/RhECAST/Model/"

###SVANTE
##base_path = "/home/filbrand/rhecast/"

# Load GPKG files
gpkg_m = gpd.read_file(f"{base_path}Input/gpkg/lulc/mun_1929.gpkg")
gpkg_p = gpd.read_file(f"{base_path}Input/gpkg/lulc/prov_1929.gpkg")
gpkg_ws = gpd.read_file(f"{base_path}Input//gpkg/weather_stations_1929.gpkg")

print("gpkg data loaded")

# Load CSV files
lulc_m = pd.read_csv(f"{base_path}Input/csv/MUN_1929_lulc.csv")
lulc_p = pd.read_csv(f"{base_path}Input/csv/PROV_1929_lulc.csv") 
w_s = pd.read_csv(f"{base_path}Input/csv/w_stations.csv")

print("csv data loaded")

# Ensure that 'Name' columns in both GPKG and CSV dataframes are of the same type
gpkg_m['Name'] = gpkg_m['Name'].astype(str)
lulc_m['Name'] = lulc_m['Name'].astype(str)
gpkg_p['Name'] = gpkg_p['Name'].astype(str)
lulc_p['Name'] = lulc_p['Name'].astype(str)
gpkg_ws['Name'] = gpkg_ws['Name'].astype(str)
w_s['Name'] = w_s['Name'].astype(str)

# Ensure that 'Name' columns in both gpkg and lulc are of the same type
gpkg_m['Name'] = gpkg_m['Name'].astype(str)
lulc_m['Name'] = lulc_m['Name'].astype(str)

# Function to convert specified columns to numeric in a DataFrame
def convert_columns_to_numeric(df, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')  # Use 'coerce' to handle errors by setting invalid values to NaN
    return df

# Define the columns to be converted
columns_to_convert = ["POP_TOT", "Cropland", "Agroforestry", 
                      "Grassland", "Orchard", "Managed_forest", 
                      "Mixed_forest", "Uncultivated_productive_area", "Cattle", "cows", 
                      "oxens", "Equines", "Swine", "Ovines", "Caprines"]

# Apply the function to lulc_m
lulc_m = convert_columns_to_numeric(lulc_m, columns_to_convert)

# Loop through all columns in the DataFrame 'lulc_p', converting them to numeric except for the 'Name' and 'REG' columns
for col in lulc_p.columns:
    if col not in ["Name", "REG"]:  # Skip the "Name" and "REG" columns
        lulc_p[col] = pd.to_numeric(lulc_p[col], errors='coerce')  # Convert to numeric, coerce errors to NaN

# Loop through all columns in the DataFrame 'w_s', converting them to numeric except for the 'Name' and 'Reg' columns
for col in w_s.columns:
    if col not in ["Name", "Reg"]:  # Skip the "Name" and "Reg" columns
        w_s[col] = pd.to_numeric(w_s[col], errors='coerce')  # Convert to numeric, coerce errors to NaN

# Merge operations, assuming the 'Name' column exists in all dataframes and is the key for merging
gpkg_m = gpkg_m.merge(lulc_m, on='Name', how='left')
gpkg_p = gpkg_p.merge(lulc_p, on='Name', how='left')
gpkg_ws = gpkg_ws.merge(w_s, on='Name', how='left')

# Replace NA values with 0 in all columns for gpkg_m, gpkg_p, and gpkg_ws
gpkg_m = gpkg_m.fillna(0)
gpkg_p = gpkg_p.fillna(0)
gpkg_ws = gpkg_ws.fillna(0)

# Calculate Area (ha) for each polygon in gpkg_m and in gpkg_p
gpkg_m['Area_ha'] = gpkg_m.geometry.area / 10000
gpkg_p['Area_ha'] = gpkg_p.geometry.area / 10000

# Calculate Unproductive_area (ha), thus quantifies the extent of urban and non-agro-forested land for each polygon in gpkg_m
gpkg_m['Unproductive_area'] = gpkg_m['Area_ha'] - (
    gpkg_m['Cropland'] + gpkg_m['Grassland'] + 
    gpkg_m['Agroforestry'] + gpkg_m['Orchard'] + 
    gpkg_m['Managed_forest'] + gpkg_m['Mixed_forest'] + 
    gpkg_m['Uncultivated_productive_area']
)

# Fields Calculation in gpkg_p

gpkg_p['AGRI_WORKFORCE'] = gpkg_p['Land_owners']+ gpkg_p['Tenants']+gpkg_p['Sharecroppers']+gpkg_p['Day_laborers']

gpkg_p['Cattle'] = gpkg_p['calves']+ gpkg_p['heifers']+gpkg_p['cows']+gpkg_p['oxens']+gpkg_p['bulls']
gpkg_p['LIVESTOCK'] = gpkg_p['Equines']+ gpkg_p['Swine']+gpkg_p['Ovines']+gpkg_p['Caprines']

gpkg_p['Agroforestry_tot_area_ha'] = gpkg_p['AF_arable_land']+ gpkg_p['AF_grassland']+gpkg_p['AF_pastures']
gpkg_p['Grassland'] = gpkg_p['S_grassland']+ gpkg_p['AF_grassland']
gpkg_p['pasture'] = gpkg_p['S_pastures']+ gpkg_p['AF_pastures']
gpkg_p['Agricultural_Area_ha'] = gpkg_p['non_irrigated_arable_land_ha']+gpkg_p['AF_arable_land']+gpkg_p['Grassland']+gpkg_p['pasture']+gpkg_p['Fruit_tree_plantations']
gpkg_p['AGRI_and_FOREST_LAND_ha'] = gpkg_p['Agricultural_Area_ha']+gpkg_p['Managed_Forest']+gpkg_p['Mixed_forest']+gpkg_p['Uncultivated_productive_area']
gpkg_p['Unproductive_area'] = gpkg_p['Area_ha'] - gpkg_p['AGRI_and_FOREST_LAND_ha']

gpkg_p['wheat_1929_tot_q'] = gpkg_p['wheat_S_1929_tot_q'] + gpkg_p['wheat_AF_1929_tot_q']	
gpkg_p['wheat_S_1929_yield'] = gpkg_p['wheat_S_1929_tot_q'] / gpkg_p['wheat_S_area_ha']
gpkg_p['wheat_AF_1929_yield'] = gpkg_p['wheat_AF_1929_tot_q'] / gpkg_p['wheat_AF_area_ha']
gpkg_p['wheat_area_ha'] = gpkg_p['wheat_S_area_ha'] + gpkg_p['wheat_AF_area_ha']

gpkg_p['rice_1929_tot_q'] = gpkg_p['rice_S_1929_tot_q'] + gpkg_p['rice_AF_1929_tot_q']	
gpkg_p['rice_S_1929_yield'] = gpkg_p['rice_S_1929_tot_q'] / gpkg_p['rice_S_area_ha']
gpkg_p['rice_AF_1929_yield'] = gpkg_p['rice_AF_1929_tot_q'] / gpkg_p['rice_AF_area_ha']
gpkg_p['rice_area_ha'] = gpkg_p['rice_S_area_ha'] + gpkg_p['rice_AF_area_ha']

gpkg_p['corn_1929_tot_q'] = gpkg_p['corn_S_1929_tot_q'] + gpkg_p['corn_AF_1929_tot_q']	
gpkg_p['corn_S_1929_yield'] = gpkg_p['corn_S_1929_tot_q'] / gpkg_p['corn_S_area_ha']
gpkg_p['corn_AF_1929_yield'] = gpkg_p['corn_AF_1929_tot_q'] / gpkg_p['corn_AF_area_ha']
gpkg_p['corn_area_ha'] = gpkg_p['corn_S_area_ha'] + gpkg_p['corn_AF_area_ha']

gpkg_p['other_cereals_1929_tot_q'] = gpkg_p['other_cereals_S_1929_tot_q'] + gpkg_p['other_cereals_AF_1929_tot_q']	
gpkg_p['other_cereals_S_1929_yield'] = gpkg_p['other_cereals_S_1929_tot_q'] / gpkg_p['other_cereals_S_area_ha']
gpkg_p['other_cereals_AF_1929_yield'] = gpkg_p['other_cereals_AF_1929_tot_q'] / gpkg_p['other_cereals_AF_area_ha']
gpkg_p['other_cereals_area_ha'] = gpkg_p['other_cereals_S_area_ha'] + gpkg_p['other_cereals_AF_area_ha']

gpkg_p['Cereals_1929_tot_q'] = gpkg_p['wheat_1929_tot_q'] + gpkg_p['rice_1929_tot_q']+ gpkg_p['corn_1929_tot_q']+ gpkg_p['other_cereals_1929_tot_q']
gpkg_p['Cereals_area_ha'] = gpkg_p['wheat_area_ha'] + gpkg_p['rice_area_ha']+ gpkg_p['corn_area_ha']+ gpkg_p['other_cereals_area_ha']

gpkg_p['Forage_crops_1929_yield'] = gpkg_p['Forage_crops_1929_tot_q'] / gpkg_p['Forage_crops_area_ha']

# Ensure CRS is set to EPSG:32632 before exporting
gpkg_m = gpkg_m.set_crs("EPSG:32632", allow_override=True)
gpkg_p = gpkg_p.set_crs("EPSG:32632", allow_override=True)
gpkg_ws = gpkg_ws.set_crs("EPSG:32632", allow_override=True)

# Specify the final order of columns for gpkg_m, including 'geometry'

final_columns_order_m = [
    'UID','REG','PROV', 'Name', 'POP_TOT', 'Area_ha', 'Unproductive_area',
    'Cropland', 'Agroforestry', 'Grassland',
    'Orchard', 'Managed_forest', 'Mixed_forest',
    'Uncultivated_productive_area', 'Cattle', 'cows', 'oxens', 'Equines',
    'Swine', 'Ovines', 'Caprines', 'geometry'  # Ensure 'geometry' is included
]

# Reordering the columns in gpkg_m

gpkg_m = gpkg_m[final_columns_order_m]

# Specify the final order of columns for gpkg_p, including 'geometry'

final_columns_order_p = [
    'UID','REG','Name', 'POP_TOT', 'Area_ha', 'AGRI_WORKFORCE','Land_owners','Tenants',
    'Sharecroppers','Day_laborers','Other_agri_wf','LIVESTOCK','Cattle','calves','heifers','cows','oxens','bulls','Equines','Swine',
    'Ovines', 'Caprines', 'AGRI_and_FOREST_LAND_ha', 'Agricultural_Area_ha','Agroforestry_tot_area_ha','non_irrigated_arable_land_ha',
    'AF_arable_land','Grassland', 'S_grassland','AF_grassland','pasture','S_pastures','AF_pastures','Fruit_tree_plantations','Managed_Forest',
    'Mixed_forest','Uncultivated_productive_area','Unproductive_area','Cereals_area_ha','Cash_crops_area_ha','Other_crops_area_ha','Forage_crops_area_ha',
    'Fallows_area_ha','Marginal_land_area_ha','Grapes_S_area_ha','Grapes_S_1929_mean_n_ha','Grapes_AF_area_ha','Grapes_AF_1929_mean_n_ha',
    'Grapes_Spread_area_ha','Grapes_Spread_1929_mean_n_ha','AF_vines_support_tree_area_ha','AF_vines_support_tree_n_ha','Olive_T__S_area_ha',
    'Olive_T__S_1929_mean_n_ha','Olive_T__AF_area_ha','Olive_T__AF_1929_mean_n_ha','Citrus_T__S_area_ha','Citrus_T__S_1929_mean_n_ha',
    'Citrus_T__AF_area_ha','Citrus_T__AF_1929_mean_n_ha','Mulberry_S_area_ha','Mulberry_S_1929_mean_n_ha','Mulberry_AF_area_ha',
    'Mulberry_AF_1929_mean_n_ha','Mulberry_Spread_area_ha','Mulberry_Spread_1929_mean_n_ha','Mix_Orchards__S_area_ha','Mix_Orchards__S_1929_mean_n_ha',
    'Mix_Orchards__AF_area_ha','Mix_Orchards__AF_1929_mean_n_ha','Mix_Orchards_Spread_area_ha','Mix_Orchards_Spread_1929_mean_n_ha','Cereals_1929_tot_q',
    'wheat_area_ha','wheat_S_area_ha','wheat_AF_area_ha','wheat_1929_tot_q','wheat_S_1929_tot_q','wheat_AF_1929_tot_q','wheat_S_1929_yield',
    'wheat_AF_1929_yield','rice_area_ha','rice_S_area_ha','rice_AF_area_ha','rice_1929_tot_q','rice_S_1929_tot_q','rice_AF_1929_tot_q','rice_S_1929_yield',
    'rice_AF_1929_yield','corn_area_ha','corn_S_area_ha','corn_AF_area_ha','corn_1929_tot_q','corn_S_1929_tot_q','corn_AF_1929_tot_q','corn_S_1929_yield',
    'corn_AF_1929_yield','other_cereals_area_ha','other_cereals_S_area_ha','other_cereals_AF_area_ha','other_cereals_1929_tot_q','other_cereals_S_1929_tot_q',
    'other_cereals_AF_1929_tot_q','other_cereals_S_1929_yield','other_cereals_AF_1929_yield','Forage_crops_1929_tot_q','Forage_crops_1929_yield','geometry'  # Ensure 'geometry' is included
]


# Reordering the columns in gpkg_p

gpkg_p = gpkg_p[final_columns_order_p]

# Before exporting gpkg_m, exclude polygons where "POP_TOT" = 0
gpkg_m = gpkg_m[gpkg_m["POP_TOT"] != 0]
#gpkg_p = gpkg_p[gpkg_p["POP_TOT"] != 0]

# Export merged data back to GPKG with the specified CRS
gpkg_m.to_file(f"{base_path}Output/gpkg/lulc_m_1929.gpkg", layer='lulc_m_1929', driver="GPKG", if_exists='replace')
print("lulc_m_1929: Done!")
gpkg_p.to_file(f"{base_path}Output/gpkg/lulc_p_1929.gpkg", layer='lulc_p_1929', driver="GPKG", if_exists='replace')
print("lulc_p_1929: Done!")
gpkg_ws.to_file(f"{base_path}Output/gpkg/w_s_1929.gpkg", layer='w_s_1929', driver="GPKG", if_exists='replace')
print("w_s_1929: Done!")

