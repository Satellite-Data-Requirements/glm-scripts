![image](https://user-images.githubusercontent.com/54595784/213831880-7b7c7da6-8c2e-45c8-89ec-ae978f61d141.png)

## Installation and usage:

**1.** Clone the repository: https://github.com/deeplycloudy/glmtools

**2.** On the terminal, access the main dir and execute the following commands:

  conda env create -f environment.yml

  conda activate glmval

  conda install -c conda-forge matplotlib dask cartopy boto3 gdal pyresample

  conda install git pip

  pip install git+https://github.com/deeplycloudy/lmatools.git

  pip install git+https://github.com/deeplycloudy/stormdrain.git

  pip install -e .

**3.** Download and execute the scripts, in this order:

First: glm_download_and_grid_creation.py 
(Note: Select the desired time, date and the number of minutes you would like to download. Also, necessary changing the location of your make_GLM_grids.py script)

Second: glm_grid_aggregation_and_plot.py
(Note: List the gridded files and select the extent you would like to plot)

Extra: glm_grid_aggregation_and_plot_reprojection.py
The same plot but in the cylindrical equidistant projection
(Note: List the gridded files and select the extent you would like to plot)

**Note:** In this preliminary test, a "scripts" folder have been created inside the main dir
