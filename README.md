# Land cover classification using Google Earth Engine
This project shows how I used Google Earth Engine to classify land cover in a small area near Lund using satellite images. I wanted to identify four types of land: water, built‑up areas, vegetation, and bare soil.
- **Data:** I used Sentinel‑2 imagery from summer 2024 and training labels from ESA’s WorldCover map.
- **Features:** The script calculates several indices (NDVI, NDWI, MNDWI, NDBI) in addition to using the raw satellite bands.
- **Model:** I trained a Random Forest classifier on a random sample of pixels from each class, split into training and testing sets.
- **Accuracy:** The model achieved about 88% overall accuracy on the test data.
- **Output:** The script generates a land‑cover map for the area of interest and calculates the area of each class in square kilometres.

## Running the Script:

Open land_cover_classification.js in the Google Earth Engine Code Editor.

Define your area of interest (AOI) in the script.

Run the script to see the classified map and accuracy statistics.

Optionally, use the export code at the bottom to download the results as a GeoTIFF.
