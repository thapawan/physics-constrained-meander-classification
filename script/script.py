```python
import ee
import pandas as pd
import numpy as np # Added for quantile clip

# Initialize Earth Engine (ensure you have authenticated your GEE account)
# ee.Authenticate() # Uncomment and run once if not authenticated
ee.Initialize()

# ==============================================
# 1. HARMONIZED DATA EXTRACTION (10m, Annual)
# ==============================================
def get_harmonized_metrics_10m(feature):
    """Extract 10m resolution metrics with Sentinel-2 and SMAP-enhanced erodibility."""
    geom = feature.geometry()
    year = ee.Number(feature.get('year'))
    river = feature.get('river')

    # Sentinel-2 MSI (2015-2024)
    s2 = ee.ImageCollection("COPERNICUS/S2_SR") \
       .filterBounds(geom) \
       .filterDate(f'{year}-01-01', f'{year}-12-31') \
       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
       .median()

    # MNDWI at 10m (Green + SWIR1)
    # B3: Green (10m), B11: SWIR1 (20m, resampled to 10m by GEE for normalizedDifference)
    mndwi = s2.normalizedDifference().rename('MNDWI')
    centerline = mndwi.gt(0.1).selfMask().reduceToVectors(
        geometry=geom.buffer(5000),
        scale=10,
        geometryType='line',
        maxPixels=1e10
    ).first()

    # NDVI trend at 10m (NIR + Red)
    # B8: NIR (10m), B4: Red (10m)
    ndvi_trend = s2.normalizedDifference().rename('NDVI')

    # Slope at 10m (USGS 3DEP resampled)
    slope = ee.Terrain.slope(ee.Image("USGS/3DEP/10m")).rename('slope')

    # Climate data resampled to 10m
    # CHIRPS: 0.05 deg (~5km)
    precip = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
       .filterDate(f'{year}-01-01', f'{year}-12-31') \
       .sum() \
       .resample('bicubic').reproject('EPSG:4326', scale=10)

    # ERA5-Land: ~9km grid spacing
    temp = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY") \
       .filterDate(f'{year}-01-01', f'{year}-12-31') \
       .mean() \
       .resample('bicubic').reproject('EPSG:4326', scale=10)

    # Human impacts (GDAT V1 DAMS)
    # Filter dams within 5km buffer, calculate distance to river geometry
    dam_dist = ee.FeatureCollection("projects/sat-io/open-datasets/GDAT/GDAT_V1_DAMS") \
       .filterBounds(geom.buffer(5000)) \
       .map(lambda d: d.set('dist', d.geometry().distance(geom))) \
       .sort('dist').first().get('dist')

    # NLCD land cover (resampled to 10m from 30m)
    nlcd = ee.ImageCollection("USGS/NLCD") \
       .filter(ee.Filter.eq('system:index', str(year))) \
       .first() \
       .select(['developed', 'cultivated', 'forest']) \
       .resample('mode').reproject('EPSG:4326', scale=10)

    # Soil Erodibility (SSURGO 10m + SMAP moisture)
    # SSURGO: varying scales, resampled to 10m
    kfactor = ee.ImageCollection("USDA/NRCS/SSURGO/v2") \
       .filterBounds(geom) \
       .first().select('kwfact') \
       .resample('bicubic').reproject('EPSG:4326', scale=10)
    
    # SMAP: 9km, resampled to 10m
    smap = ee.ImageCollection("NASA/SMAP/SPL4SMGP/007") \
       .filterDate(f'{year}-01-01', f'{year}-12-31') \
       .mean().select('sm_surface') \
       .resample('bicubic').reproject('EPSG:4326', scale=10)
    
    # Enhance kfactor with SMAP soil moisture
    kfactor = kfactor.multiply(smap)

    # Extract metrics (50m buffer for bank properties)
    # Use centerline.buffer(50) to get properties along the river banks
    metrics = ee.Image.cat([mndwi, ndvi_trend, slope, precip, temp, nlcd, kfactor]) \
       .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=centerline.buffer(50),
            scale=10,
            maxPixels=1e10
        )

    return ee.Feature(centerline).set({
        'river': river,
        'year': year,
        'mndwi': metrics.get('MNDWI'),
        'ndvi_trend': metrics.get('NDVI'),
        'slope_deg': metrics.get('slope'),
        'precip_mm': metrics.get('precipitation'), # Note: ERA5-Land precip is 'total_precipitation_sum'
        'temp_C': metrics.get('temperature_2m'), # Note: ERA5-Land temp is 'temperature_2m'
        'dam_dist_km': ee.Number(dam_dist).divide(1000),
        'urban_pct': metrics.get('developed'),
        'ag_pct': metrics.get('cultivated'),
        'riparian_pct': metrics.get('forest'),
        'kfactor': metrics.get('kwfact'),
        'discharge_proxy': metrics.get('precipitation') * 1.5  # Adjusted for 10m scale
    })

# ==============================================
# 2. PHYSICS-BASED NORMALIZATION (10m)
# ==============================================
def physics_normalize_10m(df):
    """Normalize 10m data with adjusted physical constraints."""
    # Discharge proxy (0-1 by basin)
    # Clip to avoid division by zero or extremely small values
    df['discharge_norm'] = df.groupby('river')['discharge_proxy'].transform(
        lambda x: (x - x.min()) / (x.quantile(0.97) - x.min()).clip(lower=1e-6)).fillna(0)
    
    # Human impacts
    # Clip to avoid extreme values from very close dams or 0 distance
    df['dam_impact'] = 1 - (df['dam_dist_km'] / df['dam_dist_km'].quantile(0.97)).clip(upper=1)
    df['urban_impact'] = df['urban_pct'] / 100
    df['riparian_impact'] = 1 - (df['riparian_pct'] / 100)
    
    # Geomorphic variables
    # Curvature proxy from MNDWI, normalized by 97th percentile
    # Add a small epsilon to avoid division by zero if MNDWI is exactly 0
    df['curvature'] = 1 / (df.groupby('river')['mndwi'].transform('mean') + 1e-6)
    df['curvature_norm'] = (df['curvature'] / df['curvature'].quantile(0.97)).clip(upper=1)
    df['slope_norm'] = (df['slope_deg'] / 12.0).clip(upper=1) # Adjusted max slope for 10m
    
    # Climate 
    for col in ['precip_mm', 'temp_C']:
        # Handle cases where std might be zero (e.g., constant values)
        df[f'{col}_norm'] = df.groupby('river')[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std()!= 0 else 0).fillna(0)
    
    # Soil erodibility
    df['kfactor_norm'] = (df['kfactor'].clip(0.05, 0.75) / 0.75).clip(upper=1) # Tightened bounds for 10m
    
    return df

# ==============================================
# 3-5. CLASSIFICATION & MODELING (Placeholder)
# ==============================================
# This section would contain the actual LSTM model definition,
# CA rules implementation, training, and prediction logic.
# For demonstration, a simple placeholder classification function is included.
def classify_meanders(df):
    """Placeholder for meander classification based on simplified rules."""
    # This is a highly simplified example. Real CA rules would be complex.
    # Assigns a 'class_code' based on some arbitrary thresholds for demonstration.
    conditions = [
        (df['curvature_norm'] < 0.2) & (df['slope_norm'] < 0.1),
        (df['curvature_norm'] >= 0.2) & (df['curvature_norm'] < 0.5) & (df['slope_norm'] < 0.3),
        (df['curvature_norm'] >= 0.5) & (df['curvature_norm'] < 0.8) & (df['urban_impact'] < 0.3),
        (df['curvature_norm'] >= 0.8) & (df['kfactor_norm'] > 0.5) & (df['urban_impact'] < 0.5),
        (df['curvature_norm'] >= 0.8) & (df['kfactor_norm'] > 0.7)
    ]
    choices =  # Incipient, Developing, Mature, Dynamic, Complex
    df['class_code'] = np.select(conditions, choices, default=-1)
    return df

# ==============================================
# EXECUTION PIPELINE (10m)
# ==============================================
def process_rivers_10m(river_geoms, years=range(2015, 2025)):
    """Run the 10m pipeline for multiple rivers."""
    all_data =
    for name, geom in river_geoms.items():
        for year in years:
            # Wrap in try-except for robustness in GEE calls
            try:
                feature_data = get_harmonized_metrics_10m(
                    ee.Feature(geom, {'river': name, 'year': year})
                )
                all_data.append(feature_data.getInfo()['properties'])
            except Exception as e:
                print(f"Error processing {name} for year {year}: {e}")
                continue
    
    df = pd.DataFrame(all_data)
    
    # Ensure necessary columns exist before normalization and classification
    required_cols = ['mndwi', 'ndvi_trend', 'slope_deg', 'precip_mm', 'temp_C', 
                     'dam_dist_km', 'urban_pct', 'ag_pct', 'riparian_pct', 'kfactor', 'discharge_proxy']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan # Or appropriate default value

    df = physics_normalize_10m(df)
    df = classify_meanders(df)  # Reuse CA rules placeholder
    
    # Prepare sequences for LSTM (assuming 'class_code' is the target)
    X =
    y =
    # Group by river and then by year to maintain temporal sequence for LSTM
    for river_name, river_df in df.groupby('river'):
        # Sort by year to ensure correct temporal order for LSTM
        river_df_sorted = river_df.sort_values('year')
        
        # Extract features for LSTM input
        # Ensure these columns are present after normalization
        features = river_df_sorted[[
            'curvature_norm', 'discharge_norm', 'slope_norm',
            'urban_impact', 'dam_impact', 'kfactor_norm'
        ]].values
        
        # Ensure there's at least one time step for the sequence
        if len(features) > 0:
            X.append(features)
            # Assuming 'class_code' is the target for the entire sequence or the last step
            # For simplicity, taking the class code of the last entry in the sequence
            y.append(river_df_sorted['class_code'].iloc[-1])
    
    return df, X, y

# Example usage for 3 AL rivers
al_rivers = {
    'Sipsey': ee.Geometry.LineString([[-87.8, 33.2], [-87.5, 33.0]]),
    'Black_Warrior': ee.Geometry.LineString([[-87.6, 33.2], [-87.0, 32.8]]),
    'Alabama': ee.Geometry.LineString([[-86.8, 32.5], [-87.5, 31.0]])
}

# Commented out to prevent actual GEE calls during a simple script run
# al_df, al_X, al_y = process_rivers_10m(al_rivers)
# print("Processed Alabama Rivers Data:")
# print(al_df.head())
# print(f"Number of sequences (X): {len(al_X)}")
# print(f"Number of targets (y): {len(al_y)}")


# For 13 SE states (batch export) - This part is for GEE export, not direct local processing
# se_states = ee.FeatureCollection("TIGER/2018/States") \
#    .filter(ee.Filter.inList('STUSPS',))
# se_rivers = ee.FeatureCollection("WWF/HydroSHEDS/v1/FreeFlowingRivers") \
#    .filterBounds(se_states) \
#    .filter(ee.Filter.gte('ORD_FLOW', 5))

# # Export task - This would initiate a long-running task in GEE
# task = ee.batch.Export.table.toDrive(
#     collection=se_rivers.map(lambda f: get_harmonized_metrics_10m(
#         f.set({'year': 2020, 'river': f.get('REACH_ID')})
#     )),
#     description='SE_Rivers_10m_2020',
#     fileFormat='CSV'
# )
# # task.start() # Uncomment to start the GEE export task
