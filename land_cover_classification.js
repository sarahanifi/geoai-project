var aoi = geometry;   // use the polygon you drew
// Use your drawn AOI
var region = aoi;

// Load Sentinel-2 surface reflectance data (summer 2024)
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region)
  .filterDate('2024-06-01', '2024-08-31')
  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median()
  .clip(region);

// Calculate NDVI as a simple test
var ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI');

// Add NDVI layer to the map
Map.centerObject(region, 10);
Map.addLayer(ndvi, {min:0, max:1, palette:['white','green']}, 'NDVI');
// ========= 1) REGION =========
var region = aoi;  // uses your drawn polygon

// ========= 2) SENTINEL-2 (L2A) CLOUD-MASK + COMPOSITE =========
function maskS2clouds(img) {
  var scl = img.select('SCL');
  var mask = scl.neq(3) // cloud shadow
              .and(scl.neq(8))  // medium probability clouds
              .and(scl.neq(9))  // high probability clouds
              .and(scl.neq(10)) // thin cirrus
              .and(scl.neq(11)); // snow/ice
  return img.updateMask(mask)
            .select(['B2','B3','B4','B8','B11','B12','SCL'])
            .divide(10000);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(region)
  .filterDate('2024-06-01', '2024-09-30') // summer, fewer clouds
  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds);

var comp = s2.median().clip(region);

// ========= 3) FEATURES (BANDS + INDICES) =========
var ndvi = comp.normalizedDifference(['B8','B4']).rename('NDVI');
var ndwi = comp.normalizedDifference(['B3','B8']).rename('NDWI');
var mndwi = comp.normalizedDifference(['B3','B11']).rename('MNDWI');
var ndbi = comp.normalizedDifference(['B11','B8']).rename('NDBI');

var feats = comp.select(['B2','B3','B4','B8','B11','B12'])
                .addBands([ndvi, ndwi, mndwi, ndbi]);

var bandNames = ['B2','B3','B4','B8','B11','B12','NDVI','NDWI','MNDWI','NDBI'];

// ========= 4) LABELS FROM ESA WORLDCOVER (SIMPLIFIED TO 4 CLASSES) =========
// If v200 errors, change to 'ESA/WorldCover/v100'
var wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map').clip(region);

// Map ESA classes to 4 super-classes: 0 Water, 1 Built, 2 Vegetation, 3 Bare
var label = wc.remap(
  [80, 50, 10, 20, 30, 40, 95],  // Water, Built, Trees, Shrubs, Grass, Crops, Bare
  [ 0,  1,  2,  2,   2,   2,   3]
).rename('label');

// ========= 5) SAMPLE, SPLIT, TRAIN RF =========
var samples = feats.addBands(label).stratifiedSample({
  numPoints: 4000,               // enough but still quick
  classBand: 'label',
  region: region,
  scale: 10,
  classValues: [0,1,2,3],
  classPoints: [800,800,2000,400],  // more vegetation samples
  geometries: true,
  seed: 42
});

var split = 0.7;
var withRand = samples.randomColumn('rand', 42);
var train = withRand.filter(ee.Filter.lt('rand', split));
var test  = withRand.filter(ee.Filter.gte('rand', split));

var rf = ee.Classifier.smileRandomForest({numberOfTrees: 200})
  .train({features: train, classProperty: 'label', inputProperties: bandNames});

// ========= 6) CLASSIFY + ACCURACY =========
var classified = feats.select(bandNames).classify(rf).clip(region);

var testClassified = test.classify(rf);
var cm = testClassified.errorMatrix('label', 'classification');
print('Confusion Matrix', cm);
print('Overall Accuracy', cm.accuracy());

// ========= 7) DISPLAY =========
Map.centerObject(region, 11);
var palette = ['3F88C5','D00000','2E7D32','C2B280']; // water, built, vegetation, bare
Map.addLayer(classified, {min:0, max:3, palette: palette}, 'Land cover (RF)');

// ========= 8) AREA BY CLASS (km²) =========
var areaImg = ee.Image.pixelArea().divide(1e6).addBands(classified);
var areas = areaImg.reduceRegion({
  reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'class'}),
  geometry: region, scale: 10, maxPixels: 1e13
});
print('Area (km²) by class', areas);

// ========= 9) EXPORT (OPTIONAL) =========
Export.image.toDrive({
  image: classified.toByte(),
  description: 'RF_LandCover_10m',
  region: region,
  scale: 10,
  maxPixels: 1e13
});
Export.image.toDrive({
  image: classified.toByte().clip(region),   // 👈 Clip to AOI polygon
  description: 'RF_LandCover_10m',
  region: region,
  scale: 10,
  maxPixels: 1e13
});
