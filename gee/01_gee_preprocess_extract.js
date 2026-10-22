
// ==========================================================
// GEE Workflow | Moroccan Dryland Vegetation Study
// Author: Yuwen Pang (pangy@si.edu)
// ==========================================================
// Supporting JavaScript workflow for the manuscript:
// "Field-calibrated satellite indicators of live vegetation
// dynamics in sparsely vegetated Moroccan dryland steppes."
//
// Processes multi-sensor satellite imagery and extracts spectral
// indices for subsequent field-calibrated vegetation analyses.
// Update study-specific GEE asset paths before running.
// Model calibration, validation, and trend analyses use Python.
// ==========================================================

var WORKING_AREA = ee.FeatureCollection('projects/YOUR_PROJECT/assets/YOUR_STUDY_AREA');
var STATIONS = ee.FeatureCollection('projects/YOUR_PROJECT/assets/YOUR_STATION_POINTS');
// The station asset must have attributes that identify stations and habitat classes.
// Do not publish confidential survey coordinates or IDs without permission.
var AOI = WORKING_AREA.geometry();
var OUT_FOLDER = 'GEE-Export-CSVs';
var BUFFER_M = 50;
var TREE_THRESHOLD = 10; // retain <10% tree cover
var CLOUD_THRESHOLD = 10;

var INDEX_BANDS = ['NDVI','EVI','SAVI','MSAVI','NDMI','NBR','NBR2','NDWI','MNDWI'];

// Verify STATIONS are point geometries. If they are already 50-m polygons, use STATIONS directly.
var STATION_BUFFERS = STATIONS.map(function(f) {
  f = ee.Feature(f);
  return f.buffer(BUFFER_M).copyProperties(f, f.propertyNames());
});

// Reiner et al. (2023): Africa 2019 % tree canopy cover, 100 m.
var TREE_COVER = ee.Image('projects/sat-io/open-datasets/PS_AFRICA_TREECOVER_2019_100m_V10')
  .rename('tree_cover_pct');
var OPEN_MASK = TREE_COVER.gte(0).and(TREE_COVER.lt(TREE_THRESHOLD));
function applyTreeMask(img) { return ee.Image(img).updateMask(OPEN_MASK); }

// ---------- Landsat Collection 2 Level 2 ----------
function maskL7(img) {
  var qa = img.select('QA_PIXEL');
  var m = qa.bitwiseAnd(1 << 0).eq(0) // fill
    .and(qa.bitwiseAnd(1 << 1).eq(0)) // dilated cloud
    .and(qa.bitwiseAnd(1 << 3).eq(0)) // cloud
    .and(qa.bitwiseAnd(1 << 4).eq(0)) // cloud shadow
    .and(qa.bitwiseAnd(1 << 5).eq(0)); // snow
  return img.updateMask(m);
}
function maskL89(img) {
  var qa = img.select('QA_PIXEL');
  var m = qa.bitwiseAnd(1 << 0).eq(0)
    .and(qa.bitwiseAnd(1 << 1).eq(0))
    .and(qa.bitwiseAnd(1 << 2).eq(0)) // high-confidence cirrus
    .and(qa.bitwiseAnd(1 << 3).eq(0))
    .and(qa.bitwiseAnd(1 << 4).eq(0))
    .and(qa.bitwiseAnd(1 << 5).eq(0));
  return img.updateMask(m);
}
function renameScaleLandsat(img, sensor) {
  var from = (sensor === 'LE07')
    ? ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7']
    : ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7'];
  var to = ['BLUE','GREEN','RED','NIR','SWIR1','SWIR2'];
  return img.select(from, to).multiply(0.0000275).add(-0.2)
    .copyProperties(img, ['system:time_start','system:index'])
    .set('platform', sensor);
}

// ---------- Sentinel-2 SR Harmonized ----------
function maskS2(img) {
  var scl = img.select('SCL');
  var m = scl.neq(3)  // cloud shadow
    .and(scl.neq(8))  // medium cloud probability
    .and(scl.neq(9))  // high cloud probability
    .and(scl.neq(10)) // cirrus
    .and(scl.neq(11)); // snow/ice
  return img.updateMask(m);
}
function renameScaleS2(img) {
  // Explicit 0.0001 scaling is required before EVI/SAVI/MSAVI.
  return img.select(
      ['B2','B3','B4','B8','B11','B12'],
      ['BLUE','GREEN','RED','NIR','SWIR1','SWIR2'])
    .multiply(0.0001)
    .copyProperties(img, ['system:time_start','system:index'])
    .set('platform', 'Sentinel-2');
}

// ---------- MODIS MOD09A1 v6.1 ----------
function maskMODIS(img) {
  var qa = img.select('StateQA');
  // Correct StateQA interpretation: bit 2 = cloud shadow, bit 10 = internal cloud,
  // bit 11 = internal fire. The legacy code incorrectly labeled bit 11 as shadow.
  var noShadow = qa.bitwiseAnd(1 << 2).eq(0);
  var noCloud  = qa.bitwiseAnd(1 << 10).eq(0);
  var noFire   = qa.bitwiseAnd(1 << 11).eq(0);
  return img.updateMask(noShadow.and(noCloud).and(noFire));
}
function renameScaleMODIS(img) {
  // Cross-sensor SWIR correspondence: b06 ~1.6 um (SWIR1), b07 ~2.1 um (SWIR2).
  return img.select(
      ['sur_refl_b03','sur_refl_b04','sur_refl_b01','sur_refl_b02','sur_refl_b06','sur_refl_b07'],
      ['BLUE','GREEN','RED','NIR','SWIR1','SWIR2'])
    .multiply(0.0001)
    .copyProperties(img, ['system:time_start','system:index'])
    .set('platform', 'Terra MODIS');
}

// ---------- Indices ----------
function addIndices(img) {
  var B = img.select('BLUE'), G = img.select('GREEN'), R = img.select('RED');
  var N = img.select('NIR'), S1 = img.select('SWIR1'), S2 = img.select('SWIR2');
  var ndvi = N.subtract(R).divide(N.add(R)).rename('NDVI');
  var evi = N.subtract(R).multiply(2.5)
    .divide(N.add(R.multiply(6)).subtract(B.multiply(7.5)).add(1)).rename('EVI');
  var savi = N.subtract(R).multiply(1.5).divide(N.add(R).add(0.5)).rename('SAVI');
  var t = N.multiply(2).add(1);
  var msavi = t.subtract(t.pow(2).subtract(N.subtract(R).multiply(8)).max(0).sqrt())
    .divide(2).rename('MSAVI');
  var ndmi = N.subtract(S1).divide(N.add(S1)).rename('NDMI');
  var nbr = N.subtract(S2).divide(N.add(S2)).rename('NBR');
  var nbr2 = S1.subtract(S2).divide(S1.add(S2)).rename('NBR2');
  var ndwi = G.subtract(N).divide(G.add(N)).rename('NDWI');
  var mndwi = G.subtract(S1).divide(G.add(S1)).rename('MNDWI');
  return img.addBands([ndvi,evi,savi,msavi,ndmi,nbr,nbr2,ndwi,mndwi]);
}

function prepLandsat(id, sensor) {
  var masker = (sensor === 'LE07') ? maskL7 : maskL89;
  return ee.ImageCollection(id)
    .filterBounds(AOI).filterDate('2016-01-01','2026-01-01')
    .filter(ee.Filter.lte('CLOUD_COVER', CLOUD_THRESHOLD))
    .map(masker)
    .map(function(img){ return renameScaleLandsat(img, sensor); })
    .map(addIndices).map(applyTreeMask).select(INDEX_BANDS);
}

var landsat = prepLandsat('LANDSAT/LE07/C02/T1_L2','LE07')
  .merge(prepLandsat('LANDSAT/LC08/C02/T1_L2','LC08'))
  .merge(prepLandsat('LANDSAT/LC09/C02/T1_L2','LC09'))
  .sort('system:time_start');

var sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(AOI).filterDate('2017-01-01','2026-01-01')
  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_THRESHOLD))
  .map(maskS2).map(renameScaleS2).map(addIndices).map(applyTreeMask)
  .select(INDEX_BANDS).sort('system:time_start');

var modis = ee.ImageCollection('MODIS/061/MOD09A1')
  .filterBounds(AOI).filterDate('2016-01-01','2026-01-01')
  .map(maskMODIS).map(renameScaleMODIS).map(addIndices).map(applyTreeMask)
  .select(INDEX_BANDS).sort('system:time_start');

print('Landsat images', landsat.size());
print('Sentinel-2 images', sentinel2.size());
print('MODIS composites', modis.size());

// No temporal compositing is done here. Each image/composite is reduced independently.
// Python later selects the closest observation within +/-8 days of each field survey.
var anyValid = ee.Filter.or(
  ee.Filter.notNull(['NDVI']), ee.Filter.notNull(['EVI']), ee.Filter.notNull(['SAVI']),
  ee.Filter.notNull(['MSAVI']), ee.Filter.notNull(['NDMI']), ee.Filter.notNull(['NBR']),
  ee.Filter.notNull(['NBR2']), ee.Filter.notNull(['NDWI']), ee.Filter.notNull(['MNDWI'])
);

function extract(ic, scale, sensorName) {
  ic = ee.ImageCollection(ic);
  var n = ic.size();
  var lst = ic.toList(n);
  var seq = ee.List.sequence(0, n.subtract(1));
  var out = ee.FeatureCollection(seq.iterate(function(i, acc) {
    var img = ee.Image(lst.get(ee.Number(i)));
    var date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd');
    var reduced = img.reduceRegions({
      collection: STATION_BUFFERS,
      reducer: ee.Reducer.mean(),
      scale: scale,
      tileScale: 4
    }).map(function(f) {
      return ee.Feature(f).set({
        date: date,
        sensor: sensorName,
        platform: img.get('platform'),
        scene_id: img.get('system:index')
      });
    });
    return ee.FeatureCollection(acc).merge(reduced);
  }, ee.FeatureCollection([])));
  return out.filter(anyValid);
}

var landsatTable = extract(landsat, 30, 'Landsat');
var sentinelTable = extract(sentinel2, 10, 'Sentinel2');
var modisTable = extract(modis, 500, 'MODIS');

Export.table.toDrive({collection: landsatTable, description: 'Landsat_FieldStations_2016_2025_revised', folder: OUT_FOLDER, fileFormat: 'CSV'});
Export.table.toDrive({collection: sentinelTable, description: 'Sentinel2_FieldStations_2017_2025_revised', folder: OUT_FOLDER, fileFormat: 'CSV'});
Export.table.toDrive({collection: modisTable, description: 'MODIS_FieldStations_2016_2025_revised', folder: OUT_FOLDER, fileFormat: 'CSV'});

// Optional QC table for Comment #13.
var treeQC = TREE_COVER.addBands(TREE_COVER.gte(TREE_THRESHOLD).rename('tree_ge_10pct'))
  .reduceRegions({
    collection: STATION_BUFFERS,
    reducer: ee.Reducer.mean().combine({reducer2: ee.Reducer.max(), sharedInputs: true}),
    scale: 100,
    tileScale: 4
  });
Export.table.toDrive({collection: treeQC, description: 'FieldStations_treecover_QC_2019_100m', folder: OUT_FOLDER, fileFormat: 'CSV'});

Map.centerObject(WORKING_AREA, 7);
Map.addLayer(TREE_COVER.clip(AOI), {min:0,max:30}, 'Tree cover % 2019', false);
Map.addLayer(STATIONS, {}, 'Station points', true);
Map.addLayer(STATION_BUFFERS, {}, '50-m buffers', false);
