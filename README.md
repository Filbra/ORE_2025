#ORE_2025 – Python Scripts
These Python scripts are part of the RhECAST project and are available via Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15389080.svg)](https://doi.org/10.5281/zenodo.15389080).

#Scripts Included

- csv_to_gpkg_prov.py
Converts Catasto Agrario 1929 data from CSV format into GeoPackage (.gpkg) format.

- regression_kriging_rainfall_1929.py
Performs regression kriging to analyze rainfall data for the year 1929, based on records from the Catasto Agrario.

#Associated Data
The datasets used by these scripts are publicly available on Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15388896.svg)](https://doi.org/10.5281/zenodo.15388896)

#Description:
This repository contains Python scripts developed under the RhECAST research project for the geospatial integration and historical reconstruction of land use, agroecological practices, and climatic variables in Italy during the interwar period (specifically the year 1929). The scripts facilitate the processing of historical census and meteorological data in combination with modern geospatial datasets to support environmental history and sustainability research.
The tools were designed and tested in a Ubuntu 22.04 LTS environment and make use of modern geospatial Python libraries. The overall objective is to produce harmonised geospatial datasets (GeoPackage and GeoTIFF formats) suitable for spatial analysis, landscape modelling, and visualisation in QGIS or other GIS platforms.

Core scripts:

- csv_to_gpkg_prov.py:
    - Merges historical agricultural census data (CSV) with municipal and provincial boundary datasets (GeoPackage).
    - Computes derived variables such as unproductive land, crop yields, and livestock densities.
    - Outputs geospatially enriched .gpkg layers for municipal (lulc_m_1929) and provincial (lulc_p_1929) levels.

- regression_kriging_rainfall_1929.py:
    - Performs terrain analysis (slope, aspect, elevation) using a 200m-resolution DEM.
    - Computes Euclidean distance from coastline features.
    - Applies regression kriging to interpolate historical rainfall data based on topographic predictors and station observations.
    - Outputs high-resolution rainfall prediction rasters.

Technical stack:
  -OS: Ubuntu 22.04 LTS
  - Python ≥ 3.10
  - Key libraries: geopandas, rasterio, pandas, numpy, scikit-learn, matplotlib, pykrige

Usage:
The scripts are modular and adaptable. Users should adjust base_path variables to match their local file system, and ensure input data are correctly structured. Outputs are stored in Output/gpkg/ and Output/images/ subdirectories.
