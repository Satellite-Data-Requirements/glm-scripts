#-----------------------------------------------------------------------------------------------------------
# WMO RA III and RA IV SDR Data Processing and Visualization Task Force 
# Target Product(s): GLM FED, TOE and FMA
# Script: Aggregate 1 min grids into a time series and accumulate in time + plot (reprojection)
# References: 
# https://github.com/deeplycloudy/glmtools/blob/master/examples/aggregate_and_plot.ipynb
# https://github.com/deeplycloudy/glmtools/blob/eee269127a8c6471379f331bd9de3b4659867211/glmtools/io/imagery.py
# https://github.com/deeplycloudy/glmtools/blob/eee269127a8c6471379f331bd9de3b4659867211/glmtools/plot/grid.py
# https://github.com/joaohenry23/GOES
# Author: Diego Souza
# Contributors: Alejandro Sierra (LANOT - Mexico), Joao Huaman (SENAMHI - Peru)
# Date: Jan-24-2023
#-----------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------
# REQUIRED MODULES
#-----------------------------------------------------------------------------------------------------------
import os                                                                 # Miscellaneous operating system interfaces
import xarray as xr                                                       # N-D labeled arrays and datasets in Python
import numpy as np                                                        # Fundamental package for scientific computing
import pandas as pd                                                       # Data analysis and manipulation tool
import matplotlib.pyplot as plt                                           # Plotting library
import matplotlib.colors                                                  # Matplotlib colors 
import matplotlib.cm                                                      # Builtin colormaps, colormap handling utilities
import cartopy, cartopy.crs as ccrs                                       # Plot maps
import cartopy.feature as cfeature                                        # Common drawing and filtering operations
import cartopy.io.shapereader as shpreader                                # Import shapefiles
import time as t                                                          # Time access and conversion 
import pyproj as pyproj                                                   # Python interface to PROJ (cartographic projections and coordinate transformations library)
from pyresample import utils                                              # Pyresample utils module
from pyresample.geometry import SwathDefinition                           # Geometry definitions
from pyresample.kd_tree import resample_nearest                           # Resampling of Swath Data
from osgeo import osr                                                     # Python bindings for GDAL
from osgeo import gdal                                                    # Python bindings for GDAL
from netCDF4 import Dataset                                               # Read / Write NetCDF4 files
from datetime import datetime, timedelta                                  # Basic Dates and time types
from glmtools.io.imagery import open_glm_time_series, aggregate           # glmtools utilities
from glmtools.plot.values import display_params                           # glmtools utilities  
from utilities import geo2grid, latlon2xy, convertExtent2GOESProjection   # Our own utilities  
from utilities import download_CMI, download_GLM                          # Our function for download

#-----------------------------------------------------------------------------------------------------------
# Start the time counter
start_time = t.time()  

# Desired extent
extent = [-92.0, 29.0, -89.0, 32.0] # Min lon, Min lat, Max lon, Max lat
#extent = [-59.0, -34.0, -53.0, -28.0] # Min lon, Min lat, Max lon, Max lat

#-----------------------------------------------------------------------------------------------------------
# AGGREGATE THE GLM GRIDDED FILES
#-----------------------------------------------------------------------------------------------------------

# List of GLM gridded files
fns = [os.path.join('C:/VLAB/glmtools-master/scripts/GLM_grids/2020/Nov/25/', fn)
                    for fn in 
                    ('OR_GLM-L2-GLMF-M3_G16_s20203301815000_e20203301816000_c20230200853280.nc',
                     'OR_GLM-L2-GLMF-M3_G16_s20203301816000_e20203301817000_c20230200853410.nc',
                     'OR_GLM-L2-GLMF-M3_G16_s20203301817000_e20203301818000_c20230200853510.nc',
                     'OR_GLM-L2-GLMF-M3_G16_s20203301818000_e20203301819000_c20230200854000.nc',
                     'OR_GLM-L2-GLMF-M3_G16_s20203301819000_e20203301820000_c20230200854110.nc'
                    )
      ]

# Initial datetime
yyyymmddhhmn = '202011251815'

# Minutes to aggregate
accum_mins = 5

# Start date, duration and end date
yyyy = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%Y')
mm = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%m')
dd = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%d')
hh = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%H')
mn = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%M')
startdate = datetime(int(yyyy), int(mm), int(dd), int(hh), int(mn))
duration = timedelta(0, 60*accum_mins)
enddate = startdate + duration

# Open the time series
glm = open_glm_time_series(fns)
#print(glm)

# Aggregate the files according to the parameters
agglm = aggregate(glm, accum_mins,[startdate,enddate])

# Set the time coordinate to the start time of each time bin (could also choose mid)
agglm['time_bins'] = [v.left for v in agglm.time_bins.values]
glm_grids = agglm.rename({'time_bins':'time'})
#print(glm_grids)
#print(glm_grids.attrs)

# Reading global attributes (time and date)
start = glm_grids.attrs['time_coverage_start']
end = glm_grids.attrs['time_coverage_end']

#-----------------------------------------------------------------------------------------------------------
# RETRIEVE THE FED, TOE AND MFA
#-----------------------------------------------------------------------------------------------------------

# Field selection
# Options: 'flash_extent_density', 'minimum_flash_area','total_energy', 'group_extent_density', 'average_group_area', 'group_centroid_density'           

# FED
f = 'flash_extent_density'        
# Extract the data from that field
glm_sel = glm_grids[f].sel(time=startdate)
fed = glm_sel.data
fed[fed==0] = np.nan # If value is zero, set as NaN

# TOE
f = 'total_energy'        
# Extract the data from that field
glm_sel = glm_grids[f].sel(time=startdate)
toe = glm_sel.data
toe[toe==0] = np.nan # If value is zero, set as NaN

# MFA
f = 'minimum_flash_area'        
# Extract the data from that field
glm_sel = glm_grids[f].sel(time=startdate)
mfa = glm_sel.data
mfa[mfa==0] = np.nan # If value is zero, set as NaN

#-----------------------------------------------------------------------------------------------------------
# RETRIEVE PROJECTION INFORMATION FROM THE AGGREGATED FILE
#-----------------------------------------------------------------------------------------------------------

# Retrieve projection information
glmx = glm_grids.x.data[:] # GOES fixed grid projection x-coordinate (rad)
glmy = glm_grids.y.data[:] # GOES fixed grid projection y-coordinate (rad) 
proj_var = glm_grids['goes_imager_projection'] 
satheight = proj_var.perspective_point_height
x = glmx * satheight # rad x * altitude = meters
y = glmy * satheight # rad y * altitude = meters
glm_xlim = x.min(), x.max() # x minimum
glm_ylim = y.min(), y.max() # x maximum
satlon = proj_var.longitude_of_projection_origin # Central Longitude
satsweep = proj_var.sweep_angle_axis # Sweep Angle Axis

#-----------------------------------------------------------------------------------------------------------
# REPROJECT THE THE FED, TOE AND MFA DATASETS
#-----------------------------------------------------------------------------------------------------------

# Calculating latitudes and longitudes (from radians to decimals)
x, y = np.meshgrid(x, y)
proj = pyproj.Proj(proj='geos', h=satheight, lon_0=satlon, sweep=satsweep)
lons, lats = proj(x, y, inverse=True)
lons = np.where((lons>=-360.0)&(lons<=360.0)&(lats>=-90.0)&(lats<=90.0),lons,-999.99).astype(np.float32)
lats = np.where((lons>=-360.0)&(lons<=360.0)&(lats>=-90.0)&(lats<=90.0),lats,-999.99).astype(np.float32)

# Desired extent
area_extent = [extent[0], extent[2], extent[1], extent[3]] # Min lon, Max lon, Min lat, Max lat
PixResol=2.0

nx = int( np.ceil( (area_extent[1] - area_extent[0])*111.0/float(PixResol) ) )
ny = int( np.ceil( (area_extent[3] - area_extent[2])*111.0/float(PixResol) ) )
Lons = np.arange(nx+1,dtype=np.float32)*(float(PixResol)/111.0)+area_extent[0]
Lats = np.arange(ny+1,dtype=np.float32)*(float(PixResol)/111.0)*(-1.0)+area_extent[3]
LonCenCyl, LatCenCyl = np.meshgrid(Lons, Lats)
#print(LonCenCyl)
#print(LatCenCyl)

# Calculate midpoint in X
#Lons = midpoint_in_x(LonCenCyl, fmt=np.float32)
Field = LonCenCyl
Field = np.array(Field,dtype=np.float32)
Field = np.column_stack((Field, np.full([Field.shape[0],1],-999.99,dtype=np.float32)))
right = np.column_stack((Field[:,1:], np.full([Field.shape[0],1],-999.99,dtype=np.float32)))
left = np.column_stack((np.full([Field.shape[0],1],-999.99,dtype=np.float32), Field[:,:-1]))
left2 = np.column_stack((np.full([Field.shape[0],2],-999.99,dtype=np.float32), Field[:,:-2]))
midpoint = np.where((Field>-400.0)&(left<-400.0),Field-(right-Field)/2.0,-999.99)
midpoint = np.where((Field>-400.0)&(left>-400.0),(left+Field)/2.0,midpoint)
Lons = np.where((Field<-400.0)&(left>-400.0),left+(left-left2)/2.0,midpoint)

# Calculate midpoint in Y
#Lats = midpoint_in_y(LatCenCyl, fmt=np.float32)
Field = LatCenCyl
Field = np.array(Field,dtype=np.float32)
Field = np.vstack((Field, np.full([1,Field.shape[1]],-999.99,dtype=np.float32)))
lower = np.vstack((Field[1:,:], np.full([1,Field.shape[1]],-999.99,dtype=np.float32)))
upper = np.vstack((np.full([1,Field.shape[1]],-999.99,dtype=np.float32), Field[:-1,:]))
upper2 = np.vstack((np.full([2,Field.shape[1]],-999.99,dtype=np.float32), Field[:-2,:]))
midpoint = np.where((Field>-400.0)&(upper<-400.0),Field-(lower-Field)/2.0,-999.99)
midpoint = np.where((Field>-400.0)&(upper>-400.0),(upper+Field)/2.0,midpoint)
Lats = np.where((Field<-400.0)&(upper>-400.0),upper+(upper-upper2)/2.0,midpoint)

# Calculate midpoint in X
#LonCorCyl = midpoint_in_y(Lons, fmt=np.float32)
Field = Lons
Field = np.array(Field,dtype=np.float32)
Field = np.column_stack((Field, np.full([Field.shape[0],1],-999.99,dtype=np.float32)))
right = np.column_stack((Field[:,1:], np.full([Field.shape[0],1],-999.99,dtype=np.float32)))
left = np.column_stack((np.full([Field.shape[0],1],-999.99,dtype=np.float32), Field[:,:-1]))
left2 = np.column_stack((np.full([Field.shape[0],2],-999.99,dtype=np.float32), Field[:,:-2]))
midpoint = np.where((Field>-400.0)&(left<-400.0),Field-(right-Field)/2.0,-999.99)
midpoint = np.where((Field>-400.0)&(left>-400.0),(left+Field)/2.0,midpoint)
LonCorCyl = np.where((Field<-400.0)&(left>-400.0),left+(left-left2)/2.0,midpoint)

# Calculate midpoint in Y
#LatCorCyl = midpoint_in_x(Lats, fmt=np.float32)
Field = Lats
Field = np.array(Field,dtype=np.float32)
Field = np.vstack((Field, np.full([1,Field.shape[1]],-999.99,dtype=np.float32)))
lower = np.vstack((Field[1:,:], np.full([1,Field.shape[1]],-999.99,dtype=np.float32)))
upper = np.vstack((np.full([1,Field.shape[1]],-999.99,dtype=np.float32), Field[:-1,:]))
upper2 = np.vstack((np.full([2,Field.shape[1]],-999.99,dtype=np.float32), Field[:-2,:]))
midpoint = np.where((Field>-400.0)&(upper<-400.0),Field-(lower-Field)/2.0,-999.99)
midpoint = np.where((Field>-400.0)&(upper>-400.0),(upper+Field)/2.0,midpoint)
LatCorCyl = np.where((Field<-400.0)&(upper>-400.0),upper+(upper-upper2)/2.0,midpoint)
#print(LonCorCyl)
#print(LatCorCyl)

# Proj parameters
Prj = pyproj.Proj('+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=6378.137 +b=6378.137 +units=km')
AreaID = 'cyl'
AreaName = 'cyl'
ProjID = 'cyl'
Proj4Args = '+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=6378.137 +b=6378.137 +units=km'

ny, nx = LonCenCyl.shape
SW = Prj(LonCenCyl.min(), LatCenCyl.min())
NE = Prj(LonCenCyl.max(), LatCenCyl.max())
area_extent = [SW[0], SW[1], NE[0], NE[1]]

# Area nd swath definitions
AreaDef = utils.get_area_def(AreaID, AreaName, ProjID, Proj4Args, nx, ny, area_extent)
SwathDef = SwathDefinition(lons=lons, lats=lats)

# Read the FED, TOE and MFA as numpy arrays
fed = np.ascontiguousarray(fed, dtype=np.float32)
toe = np.ascontiguousarray(toe, dtype=np.float32)
mfa = np.ascontiguousarray(mfa, dtype=np.float32)

# Reproject the FED, TOE and MFA to the Cylindrical Equidistant projection
fedCyl = resample_nearest(SwathDef, fed, AreaDef, radius_of_influence=6000,
                          fill_value=np.nan, epsilon=3, reduce_data=True)

toeCyl = resample_nearest(SwathDef, toe, AreaDef, radius_of_influence=6000,
                          fill_value=np.nan, epsilon=3, reduce_data=True)

mfaCyl = resample_nearest(SwathDef, mfa, AreaDef, radius_of_influence=6000,
                          fill_value=np.nan, epsilon=3, reduce_data=True)

#-----------------------------------------------------------------------------------------------------------
# EXTRA: DOWNLOAD AND REPROJECT AN ABI FILE 
#-----------------------------------------------------------------------------------------------------------

# Desired extent
#extent = [-92.0, 29.0, -89.0, 32.0] # Min lon, Min lat, Max lon, Max lat

# Create a dir to store the samples
input = "samples"; os.makedirs(input, exist_ok=True)

# Download the ABI file
abi_file = download_CMI('202011251810', 13, input)

#-----------------------------------------------------------------------------------------------------------

# Variable
var = 'CMI'

# Open the file
img = gdal.Open(f'NETCDF:{input}/{abi_file}.nc:' + var)

# Read the header metadata
metadata = img.GetMetadata()
scale = float(metadata.get(var + '#scale_factor'))
offset = float(metadata.get(var + '#add_offset'))
undef = float(metadata.get(var + '#_FillValue'))
dtime = metadata.get('NC_GLOBAL#time_coverage_start')

# Load the data
ds = img.ReadAsArray(0, 0, img.RasterXSize, img.RasterYSize).astype(float)

# Apply the scale, offset and convert to celsius
ds = (ds * scale + offset) - 273.15

# Read the original file projection and configure the output projection
source_prj = osr.SpatialReference()
source_prj.ImportFromProj4(img.GetProjectionRef())

target_prj = osr.SpatialReference()
target_prj.ImportFromProj4("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")

# Reproject the data
GeoT = img.GetGeoTransform()
driver = gdal.GetDriverByName('MEM')
raw = driver.Create('raw', ds.shape[0], ds.shape[1], 1, gdal.GDT_Float32)
raw.SetGeoTransform(GeoT)
raw.GetRasterBand(1).WriteArray(ds)

# Define the parameters of the output file  
options = gdal.WarpOptions(format = 'netCDF', 
          srcSRS = source_prj, 
          dstSRS = target_prj,
          outputBounds = (extent[0], extent[3], extent[2], extent[1]), 
          outputBoundsSRS = target_prj, 
          outputType = gdal.GDT_Float32, 
          srcNodata = undef, 
          dstNodata = 'nan', 
          xRes = 0.02, 
          yRes = 0.02, 
          resampleAlg = gdal.GRA_NearestNeighbour)

print(options)

# Write the reprojected file on disk
gdal.Warp(f'{input}/{abi_file}_ret.nc', raw, options=options)
#-----------------------------------------------------------------------------------------------------------
# Open the reprojected GOES-R image
file = Dataset(f'{input}/{abi_file}_ret.nc')

# Get the pixel values
abi_data = file.variables['Band1'][:]
#print(abi_data.shape)

#-----------------------------------------------------------------------------------------------------------
# PLOT THE DATASETS (MULTIPANEL)
#-----------------------------------------------------------------------------------------------------------

# Choose the plot size (width x height, in inches)
fig, axs = plt.subplots(1,3, figsize=(19,7), sharex = False, sharey = False, gridspec_kw ={'hspace':0.15, 'wspace':0.15}, 
subplot_kw=dict(projection=ccrs.PlateCarree())) # 1 row x 3 columns  

# Define the image extent
img_extent = [extent[0], extent[2], extent[1], extent[3]]
#-----------------------------------------------------------------------------------------------------------
# Create a custom color scale:
# HEX values got from: https://imagecolorpicker.com/:
colors = ["#0000b8", "#0702c1", "#0f05cb", "#1808d6", "#1f0bdf", 
          "#280eeb", "#2f10f4", "#3813fe", "#2d49ff", "#1e92ff", 
          "#12dfff", "#5bfdb6", "#d5ff3c", "#ffad12", "#f73611", 
          "#cc0e4e", "#f9e5e7"]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
#cmap.set_over('#')
#cmap.set_under('#')

# Plot the image
vmin = 1
vmax = 256
norm = matplotlib.colors.LogNorm(vmin, vmax, clip='False')

# Plot the abi image
axs[0].imshow(abi_data, vmin=-50, vmax=80, origin='upper', extent=img_extent, cmap='gray_r', alpha = 1.0)
# Plot the glm image
img1 = axs[0].imshow(fedCyl, vmin=vmin, vmax=vmax, norm=norm, origin='upper', transform=ccrs.PlateCarree(), extent=img_extent, cmap=cmap)

# Add a land mask
axs[0].add_feature(cfeature.LAND, facecolor='#202020')
# Add an ocean mask
axs[0].add_feature(cfeature.OCEAN, facecolor='#000000')

# Add a shapefile
shapefile = list(shpreader.Reader('ne_10m_admin_1_states_provinces.shp').geometries())
axs[0].add_geometries(shapefile, ccrs.PlateCarree(), edgecolor='gray',facecolor='none', linewidth=0.3)
 
# Add coastlines, borders and gridlines
axs[0].coastlines(resolution='50m', color='white', linewidth=0.8)
axs[0].add_feature(cartopy.feature.BORDERS, edgecolor='white', linewidth=0.5)
#axs[0].gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5)
gl = axs[0].gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 1), ylocs=np.arange(-90, 90, 1), draw_labels=True)
gl.top_labels = False
gl.right_labels = False

# Add a colorbar
cb = plt.colorbar(img1, ax=axs[0], label='Flash Extent Density - FED (count)', extend='neither', orientation='horizontal', pad=0.05, fraction=0.05)
ticks = [0, 2, 4, 8, 16, 32, 64, 128, 256]
cb.set_ticks(ticks)
cb.set_ticklabels([f"{t:g}" for t in ticks])
cb.minorticks_off()
	
# Add a title
axs[0].set_title(f'GOES-16 GLM FED {start} - {end}', fontweight='bold', fontsize=6, loc='left')
axs[0].set_title('Reg.: ' + str(extent) , fontsize=5, loc='right')
#-----------------------------------------------------------------------------------------------------------
# Get the regional pixel values
toeCyl = toeCyl * 1000000

# Plot the image
vmin = 0.01
vmax = 1500
norm = matplotlib.colors.LogNorm(vmin, vmax, clip='False')

# Plot the abi image
axs[1].imshow(abi_data, vmin=-50, vmax=80, origin='upper', extent=img_extent, cmap='gray_r', alpha = 1.0)
# Plot the image
img2 = axs[1].imshow(toeCyl, vmin=vmin, vmax=vmax, norm=norm, origin='upper', transform=ccrs.PlateCarree(), extent=img_extent, cmap='magma')

# Add a land mask
axs[1].add_feature(cfeature.LAND, facecolor='#202020')
# Add an ocean mask
axs[1].add_feature(cfeature.OCEAN, facecolor='#000000')

# Add a shapefile
shapefile = list(shpreader.Reader('ne_10m_admin_1_states_provinces.shp').geometries())
axs[1].add_geometries(shapefile, ccrs.PlateCarree(), edgecolor='gray',facecolor='none', linewidth=0.3)
 
# Add coastlines, borders and gridlines
axs[1].coastlines(resolution='50m', color='white', linewidth=0.8)
axs[1].add_feature(cartopy.feature.BORDERS, edgecolor='white', linewidth=0.5)
#axs[1].gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5)
gl = axs[1].gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 1), ylocs=np.arange(-90, 90, 1), draw_labels=True)
gl.top_labels = False
gl.right_labels = False

# Add a colorbar
cb = plt.colorbar(img2, ax=axs[1], label='Total Optical Energy - TOE (fJ)', extend='neither', orientation='horizontal', pad=0.05, fraction=0.05)
ticks = [1, 5, 10, 25, 50, 150, 500, 1500]
cb.set_ticks(ticks)
cb.set_ticklabels([f"{t:g}" for t in ticks])
cb.minorticks_off()
 	
# Add a title
axs[1].set_title(f'GOES-16 GLM TOE {start} - {end}', fontweight='bold', fontsize=6, loc='left')
axs[1].set_title('Reg.: ' + str(extent) , fontsize=5, loc='right')
#-----------------------------------------------------------------------------------------------------------
# Plot the image
vmin = 64
vmax = 2500
norm = matplotlib.colors.LogNorm(vmin, vmax, clip='False')

# Plot the abi image
axs[2].imshow(abi_data, vmin=-50, vmax=80, origin='upper', extent=img_extent, cmap='gray_r', alpha = 1.0)
# Plot the image
img3 = axs[2].imshow(mfaCyl, vmin=vmin, vmax=vmax, norm=norm, origin='upper', transform=ccrs.PlateCarree(), extent=img_extent, cmap='viridis_r')

# Add a land mask
axs[2].add_feature(cfeature.LAND, facecolor='#202020')
# Add an ocean mask
axs[2].add_feature(cfeature.OCEAN, facecolor='#000000')

# Add a shapefile
shapefile = list(shpreader.Reader('ne_10m_admin_1_states_provinces.shp').geometries())
axs[2].add_geometries(shapefile, ccrs.PlateCarree(), edgecolor='gray',facecolor='none', linewidth=0.3)
 
# Add coastlines, borders and gridlines
axs[2].coastlines(resolution='50m', color='white', linewidth=0.8)
axs[2].add_feature(cartopy.feature.BORDERS, edgecolor='white', linewidth=0.5)
#axs[2].gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5)
gl = axs[2].gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 1), ylocs=np.arange(-90, 90, 1), draw_labels=True)
gl.top_labels = False
gl.right_labels = False

# Add a colorbar
cb = plt.colorbar(img3, ax=axs[2], label='Minimum Flash Area - MFA (km²)', extend='neither', orientation='horizontal', pad=0.05, fraction=0.05)
ticks = [60, 120, 300, 600, 1200, 2000]
cb.set_ticks(ticks)
cb.set_ticklabels([f"{t:g}" for t in ticks])
cb.minorticks_off()
	
# Add a title
axs[2].set_title(f'GOES-16 GLM MFA {start} - {end}', fontweight='bold', fontsize=6, loc='left')
axs[2].set_title('Reg.: ' + str(extent) , fontsize=5, loc='right')
 
#-----------------------------------------------------------------------------------------------------------
# SAVE AND VISUALIZE THE IMAGE
#-----------------------------------------------------------------------------------------------------------

# Create a dir to store the imagery
output = "output"; os.makedirs(output, exist_ok=True)

# Save the image
plt.savefig('output/image_01.png', bbox_inches='tight', pad_inches=0, dpi=100)

# End the time counter
print('\nTotal Processing Time:', round((t.time() - start_time),2), 'seconds.') 

# Show the image
plt.show()
