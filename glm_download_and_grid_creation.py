#-----------------------------------------------------------------------------------------------------------
# WMO RA III and RA IV SDR Data Processing and Visualization Task Force 
# Target Product(s): GLM FED, TOE and FMA
# Script: Create grids on the ABI fixed grid
# Reference: https://github.com/deeplycloudy/glmtools/blob/master/examples/plot_glm_test_data.ipynb
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
#
# Execute the glm_download_and_grid_creation.py and then glm_grid_aggregation_and_plot
#-----------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------
# REQUIRED MODULES
#-----------------------------------------------------------------------------------------------------------
import os                                              # Miscellaneous operating system interfaces
import subprocess                                      # Spawn new processes
import glob                                            # Unix style pathname pattern expansion
import tempfile                                        # Generate temporary files and directories
import numpy as np                                     # Fundamental package for scientific computing
import pandas as pd                                    # Data analysis and manipulation tool
import glmtools                                        # glmtools utilities
from glmtools.io.glm import GLMDataset                 # glmtools utilities
from glmtools.test.common import get_sample_data_path  # glmtools utilities
from datetime import datetime, timedelta               # Basic Dates and time types
from utilities import download_CMI, download_GLM       # Functions to download ABI and GLM data

#-----------------------------------------------------------------------------------------------------------
# DOWNLOAD GLM SAMPLES FROM AMAZON WEB SERVICES
#-----------------------------------------------------------------------------------------------------------

# Create a dir to store the samples
input = "samples"; os.makedirs(input, exist_ok=True)

# Initial datetime to process
yyyymmddhhmn = '202301101700'

# Minutes do download (after the initial file)
minutes = '5'

# Initial time and date
yyyy = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%Y')
mm = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%m')
dd = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%d')
hh = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%H')
mn = datetime.strptime(yyyymmddhhmn, '%Y%m%d%H%M').strftime('%M')

date_ini = str(datetime(int(yyyy),int(mm),int(dd),int(hh),int(mn)))
date_end = str(datetime(int(yyyy),int(mm),int(dd),int(hh),int(mn)) + timedelta(minutes=int(minutes)))
date_loop = date_ini

# List with the sample files
samples = []

# GLM download loop
while (date_loop <= date_end):
 
    # Date structure
    yyyymmddhhmnss = datetime.strptime(date_loop, '%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S')

    # Download the file
    file_glm = download_GLM(yyyymmddhhmnss, input)
    
    # Add the downloaded file to the sample list
    samples.append(os.path.abspath(input+'//'+file_glm)+'.nc')
    
    # Increment the date_ini
    date_loop = str(datetime.strptime(date_loop, '%Y-%m-%d %H:%M:%S') + timedelta(seconds=20))
 
#print(samples)

#-----------------------------------------------------------------------------------------------------------
# IF YOU ALREADY HAVE GLM SAMPLES, PUT THEM IN THE 'SAMPLES' LIST
#-----------------------------------------------------------------------------------------------------------
'''
sample_path = get_sample_data_path()
samples = [
    "OR_GLM-L2-LCFA_G16_s20181830433000_e20181830433200_c20181830433231.nc",
    "OR_GLM-L2-LCFA_G16_s20181830433200_e20181830433400_c20181830433424.nc",
    "OR_GLM-L2-LCFA_G16_s20181830433400_e20181830434000_c20181830434029.nc",
]
samples = [os.path.join(sample_path, s) for s in samples]

print(samples)
'''

#-----------------------------------------------------------------------------------------------------------
# CREATE THE GLM GRIDS
#-----------------------------------------------------------------------------------------------------------

# GLM grid directory
grid_dir = "GLM_grids"; os.makedirs(grid_dir, exist_ok=True)

# GLM tools path
glmtools_path = os.path.abspath(glmtools.__path__[0])

# Set the start time and duration
startdate = datetime(int(yyyy), int(mm), int(dd), int(hh), int(mn))
duration = timedelta(0, 60*(int(minutes)))
enddate = startdate + duration

# Execute the "make_GLM_grids.py" with the arguments    
cmd = "python C:/VLAB/glmtools-master/examples/grid/make_GLM_grids.py"
cmd += " -o {1}/{{start_time:%Y/%b/%d}}/{{dataset_name}}"
cmd += " --fixed_grid --split_events --float_output"
cmd += " --goes_position=east --goes_sector=full"
cmd += " --ctr_lat=0.0 --ctr_lon=-75.0 --dx=2.0 --dy=2.0"
cmd += " --start={3} --end={4} {2}"
cmd = cmd.format(glmtools_path, grid_dir, ' '.join(samples),
                startdate.isoformat(), enddate.isoformat())
#print (cmd)
out_bytes = subprocess.check_output(cmd.split())

#-----------------------------------------------------------------------------------------------------------
# LIST THE GLM GRIDDED FILES
#-----------------------------------------------------------------------------------------------------------

grid_dir_base = grid_dir
nc_files = glob.glob(os.path.join(grid_dir_base, startdate.strftime('%Y/%b/%d'),'*.nc'))
print(nc_files)
#-----------------------------------------------------------------------------------------------------------