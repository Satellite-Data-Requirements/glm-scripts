#-----------------------------------------------------------------------------------------------------------
# WMO RA III and RA IV SDR Data Processing and Visualization Task Force 
# Target Product(s): GLM FED, TOE and FMA
# Script: Aggregate 1 min grids into a time series and accumulate in time 
# References: 
# https://github.com/deeplycloudy/glmtools/blob/master/examples/aggregate_and_plot.ipynb
# https://github.com/deeplycloudy/glmtools/blob/eee269127a8c6471379f331bd9de3b4659867211/glmtools/io/imagery.py
# https://github.com/deeplycloudy/glmtools/blob/eee269127a8c6471379f331bd9de3b4659867211/glmtools/plot/grid.py
# Author: Diego Souza
# Date: Jan-12-2023
#
# Installation and usage:
#
# 1. Clone the repository: https://github.com/deeplycloudy/glmtools
# 
# 2. On the terminal, access the main dir and execute the following commands:
#
# conda env create -f environment.yml
# conda activate glmval
# conda install -c conda-forge matplotlib dask cartopy boto3 gdal
# conda install git pip
# pip install git+https://github.com/deeplycloudy/lmatools.git
# pip install git+https://github.com/deeplycloudy/stormdrain.git
# pip install -e .
#
# 3. Download and execute the scripts 
# Note: In this preliminary test, a "scripts" folder have been created inside the main dir
#-----------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------
# REQUIRED MODULES
#-----------------------------------------------------------------------------------------------------------
import os                                                                 # Miscellaneous operating system interfaces
import xarray as xr                                                       # N-D labeled arrays and datasets in Python
import numpy as np                                                        # Fundamental package for scientific computing
import pandas as pd                                                       # Data analysis and manipulation tool
import matplotlib.pyplot as plt                                           # Plotting library
import cartopy, cartopy.crs as ccrs                                       # Plot maps
import cartopy.feature as cfeature                                        # Common drawing and filtering operations
from netCDF4 import Dataset                                               # Read / Write NetCDF4 files
from datetime import datetime, timedelta                                  # Basic Dates and time types
from glmtools.io.imagery import open_glm_time_series, aggregate           # glmtools utilities
from glmtools.plot.values import display_params                           # glmtools utilities  
from utilities import geo2grid, latlon2xy, convertExtent2GOESProjection   # Our own utilities  
from utilities import download_CMI, download_GLM                          # Our function for download

#-----------------------------------------------------------------------------------------------------------
# AGGREGATE THE GLM GRIDDED FILES
#-----------------------------------------------------------------------------------------------------------
# List of GLM gridded files
fns = [os.path.join('C:/VLAB/glmtools-master/scripts/GLM_grids/2023/Jan/10/', fn)
                    for fn in 
                    ('OR_GLM-L2-GLMF-M3_G16_s20230101700000_e20230101701000_c20230131555150.nc',
                     'OR_GLM-L2-GLMF-M3_G16_s20230101701000_e20230101702000_c20230131555430.nc',
                     'OR_GLM-L2-GLMF-M3_G16_s20230101702000_e20230101703000_c20230131555550.nc',
                     'OR_GLM-L2-GLMF-M3_G16_s20230101703000_e20230101704000_c20230131556050.nc',
                     'OR_GLM-L2-GLMF-M3_G16_s20230101704000_e20230101705000_c20230131556180.nc',
                    )
      ]

# Initial datetime
yyyymmddhhmn = '202301101700'

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

#-----------------------------------------------------------------------------------------------------------
# RETRIEVE INFORMATION FROM THE AGGREGATED FILE
#-----------------------------------------------------------------------------------------------------------

# Retrieve projection information
glmx = glm_grids.x.data[:]
glmy = glm_grids.y.data[:]
proj_var = glm_grids['goes_imager_projection']
x = glmx * proj_var.perspective_point_height
y = glmy * proj_var.perspective_point_height
glm_xlim = x.min(), x.max()
glm_ylim = y.min(), y.max()

# Reading some global attributes
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
# REGIONAL PLOT (if you want the full disk, comment this section and change the img_extent)
#-----------------------------------------------------------------------------------------------------------
# Open the GOES-R image
file = Dataset(fns[0])

# Desired extent
extent = [-59.0, -30.0, -45.0, -15.0] # Min lon, Max lon, Min lat, Max lat

# Convert lat/lon to grid-coordinates
lly, llx = geo2grid(extent[1], extent[0], file)
ury, urx = geo2grid(extent[3], extent[2], file)

# Compute data-extent in GOES projection-coordinates
img_extent = convertExtent2GOESProjection(extent)
#img_extent = (x.min(),x.max(),y.min(),y.max())

#-----------------------------------------------------------------------------------------------------------
# EXTRA: DOWNLOAD AN ABI FILE 
#-----------------------------------------------------------------------------------------------------------

# Create a dir to store the samples
input = "samples"; os.makedirs(input, exist_ok=True)

# Download the ABI file
abi_file = download_CMI(yyyymmddhhmn, 13, input)

# Open the GOES-R image
file = Dataset(f'{input}/{abi_file}.nc')
                         
# Get the pixel values
abi_data = file.variables['CMI'][ury:lly, llx:urx] - 273.15     

#-----------------------------------------------------------------------------------------------------------
# PLOT THE DATASETS (MULTIPANEL)
#-----------------------------------------------------------------------------------------------------------

# Choose the plot size (width x height, in inches)
fig, axs = plt.subplots(1,3, figsize=(17,6), sharex = False, sharey = False, gridspec_kw ={'hspace':0.01, 'wspace':0.01}, 
subplot_kw=dict(projection=ccrs.Geostationary(central_longitude=-75.0, satellite_height=35786023.0))) # 1 row x 3 columns    
#-----------------------------------------------------------------------------------------------------------
# Get the regional pixel values
fed = fed[ury:lly, llx:urx]  

# Plot the abi image
axs[0].imshow(abi_data, vmin=-50, vmax=80, origin='upper', extent=img_extent, cmap='gray_r', alpha = 0.3)
# Plot the glm image
img1 = axs[0].imshow(fed, vmin=0, vmax=32, origin='upper', extent=img_extent, cmap='jet')

# Add a land mask
axs[0].add_feature(cfeature.LAND, facecolor='#202020')
# Add an ocean mask
axs[0].add_feature(cfeature.OCEAN, facecolor='#000000')
 
# Add coastlines, borders and gridlines
axs[0].coastlines(resolution='110m', color='white', linewidth=0.8)
axs[0].add_feature(cartopy.feature.BORDERS, edgecolor='white', linewidth=0.5)
#axs[0].gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5)
gl = axs[0].gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=False)
gl.top_labels = False
gl.right_labels = False

# Add a colorbar
fig.colorbar(img1, ax=axs[0], label='Flash Extent Density - FED (count)', extend='max', orientation='horizontal', pad=0.01, fraction=0.05)
 	
# Add a title
axs[0].set_title(f'GOES-16 GLM FED {start} - {end}', fontweight='bold', fontsize=6, loc='left')
axs[0].set_title('Reg.: ' + str(extent) , fontsize=5, loc='right')
#-----------------------------------------------------------------------------------------------------------
# Get the regional pixel values
toe = toe[ury:lly, llx:urx]  
toe = toe * 1000000

# Plot the abi image
axs[1].imshow(abi_data, vmin=-50, vmax=80, origin='upper', extent=img_extent, cmap='gray_r', alpha = 0.3)
# Plot the image
img2 = axs[1].imshow(toe, vmin=0, vmax=50, origin='upper', extent=img_extent, cmap='plasma')

# Add a land mask
axs[1].add_feature(cfeature.LAND, facecolor='#202020')
# Add an ocean mask
axs[1].add_feature(cfeature.OCEAN, facecolor='#000000')
 
# Add coastlines, borders and gridlines
axs[1].coastlines(resolution='110m', color='white', linewidth=0.8)
axs[1].add_feature(cartopy.feature.BORDERS, edgecolor='white', linewidth=0.5)
#axs[1].gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5)
gl = axs[1].gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=False)
gl.top_labels = False
gl.right_labels = False

# Add a colorbar
fig.colorbar(img2, ax=axs[1], label='Total Energy - TOE (mJ)', extend='max', orientation='horizontal', pad=0.01, fraction=0.05)
 	
# Add a title
axs[1].set_title(f'GOES-16 GLM TOE {start} - {end}', fontweight='bold', fontsize=6, loc='left')
axs[1].set_title('Reg.: ' + str(extent) , fontsize=5, loc='right')
#-----------------------------------------------------------------------------------------------------------
# Get the regional pixel values
mfa = mfa[ury:lly, llx:urx]  

# Plot the abi image
axs[2].imshow(abi_data, vmin=-50, vmax=80, origin='upper', extent=img_extent, cmap='gray_r', alpha = 0.3)
# Plot the image
img3 = axs[2].imshow(mfa, vmin=0, vmax=2000, origin='upper', extent=img_extent, cmap='viridis_r')

# Add a land mask
axs[2].add_feature(cfeature.LAND, facecolor='#202020')
# Add an ocean mask
axs[2].add_feature(cfeature.OCEAN, facecolor='#000000')
 
# Add coastlines, borders and gridlines
axs[2].coastlines(resolution='110m', color='white', linewidth=0.8)
axs[2].add_feature(cartopy.feature.BORDERS, edgecolor='white', linewidth=0.5)
#axs[2].gridlines(color='white', alpha=0.5, linestyle='--', linewidth=0.5)
gl = axs[2].gridlines(crs=ccrs.PlateCarree(), color='gray', alpha=1.0, linestyle='--', linewidth=0.25, xlocs=np.arange(-180, 180, 5), ylocs=np.arange(-90, 90, 5), draw_labels=False)
gl.top_labels = False
gl.right_labels = False

# Add a colorbar
fig.colorbar(img3, ax=axs[2], label='Minimum Flash Area - MFA (km²)', extend='max', orientation='horizontal', pad=0.01, fraction=0.05)
 	
# Add a title
axs[2].set_title(f'GOES-16 GLM MFA {start} - {end}', fontweight='bold', fontsize=6, loc='left')
axs[2].set_title('Reg.: ' + str(extent) , fontsize=5, loc='right')
#-----------------------------------------------------------------------------------------------------------    

#-----------------------------------------------------------------------------------------------------------
# SAVE AND VISUALIZE THE IMAGE
#-----------------------------------------------------------------------------------------------------------

# Create a dir to store the imagery
output = "output"; os.makedirs(output, exist_ok=True)

# Save the image
plt.savefig('output/image_01.png', bbox_inches='tight', pad_inches=0, dpi=100)

# Show the image
plt.show()