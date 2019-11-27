############################################################
####   GCM - Sea Water Potential Temperature (Surface)  ####
############################################################

### This part only should be run when using Spyder on fiji (to change the libraries directory to the appropriate one) and should be run only once

#import sys
#sys.path.remove('/usr/global/AnacondaPython/lib/python36.zip')
#sys.path.append('/usr/global/AnacondaPython/envs/ees3/lib/python36.zip')
#sys.path.remove('/usr/global/AnacondaPython/lib/python3.6')
#sys.path.append('/usr/global/AnacondaPython/envs/ees3/lib/python3.6')
#sys.path.remove('/usr/global/AnacondaPython/lib/python3.6/site-packages')
#sys.path.append('/usr/global/AnacondaPython/envs/ees36/lib/python3.6/site-packages') # Appends to main directory of libraries this directory where complete list of libraries are available, so we can import netCDF4 and Basemap
#sys.path.remove('/usr/global/AnacondaPython/lib/python3.6/lib-dynload')
#sys.path.append('/usr/global/AnacondaPython/envs/ees36/lib/python3.6/lib-dynload')
#sys.path.remove('/usr/global/AnacondaPython/lib/python3.6/site-packages/IPython/extensions')
#sys.path.append('/usr/global/AnacondaPython/envs/ees36/lib/python3.6/site-packages/IPython/extensions')
#sys.path.append('/usr/global/AnacondaPython/envs/ees36/lib/python3.6/site-packages/mpl_toolkits/')
##sys.path.append('/usr/global/AnacondaPython/envs/ees36/share/basemap/')
#sys.prefix='/usr/global/AnacondaPython/envs/ees36'

###############################################################

import os
import numpy as np
from numpy import zeros, ones, empty, nan, shape
from numpy import isnan, nanmean, nanmax, nanmin
import numpy.ma as ma
import matplotlib
import matplotlib.mlab as ml
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import math
from netCDF4 import MFDataset, Dataset, num2date, date2num, date2index
from mpl_toolkits.basemap import Basemap, cm, maskoceans

#os.chdir('/usr/global/AnacondaPython/envs/ees36/lib/python3.6/site-packages/') # Go to the directory that complete list of libraries are available, so we can import netCDF4 and Basemap

#GCM_Names = ['GFDL-ESM2M', 'CanESM2','CESM1-BGC','CMCC-CESM','CNRM-CM5','GFDL-ESM2G','GFDL-ESM2M','GISS-E2-H-CC','GISS-E2-R-CC','HadGEM2-CC','HadGEM2-ES','IPSL-CM5A-LR','IPSL-CM5A-MR','IPSL-CM5B-LR','MIROC-ESM','MIROC-ESM-CHEM','MPI-ESM-LR','MPI-ESM-MR','MRI-ESM1','IPSL-CM5B-LR','NorESM1-ME']
GCM_Names = ['GFDL-ESM2M', 'GFDL-ESM2G', 'HadGEM2-ES','IPSL-CM5A-MR', 'IPSL-CM5A-LR', 'MIROC-ESM', 'MIROC-ESM-CHEM', 'CESM1-BGC', 'CMCC-CESM', 'CanESM2', 'GISS-E2-H-CC', 'GISS-E2-R-CC', 'MPI-ESM-MR', 'MPI-ESM-LR', 'NorESM1-ME']

dir_pwd = os.getcwd() # Gets the current directory (and in which the code is placed)
dir_data_in1 = ('/data2/scratch/cabre/CMIP5/CMIP5_models/ocean_physics/') # Directory to raed raw data from
dir_data_out = (dir_pwd + '/Python Manipulated Data/') # Directory to save processed data
dir_figs = (dir_pwd + '/Figures2/') # Directory to save processed data

### Variables to be edited by user ###
start_date_cal_hist='1980' # Start Date for Calculations # Define as string, not number
end_date_cal_hist='1999' # End Date for Calculations
start_date_cal_rcp85='2080' # Start Date for Calculations # Define as string, not number
end_date_cal_rcp85='2099' # End Date for Calculations
Var_name='thetao' # The variable name to be read from .nc files
Var_plot_name='Sea Water Potential Temperature' # Name of variable to be used for plots only
#Var_plot_unit='Unit = °C' # Unit of variable to be used for plots only

### Regrdridding calculations ###
# creating new coordinate grid, same which was used in interpolation in data processing code
lat_n_regrid, lon_n_regrid = 180, 360 # Number of Lat and Lon elements in the regridded data
lon_min_regrid, lon_max_regrid = 0, 360 # Min and Max value of Lon in the regridded data
lat_min_regrid, lat_max_regrid = -90, 90 # Min and Max value of Lat in the regridded data
####creating arrays of regridded lats and lons ###
#### Latitude Bounds ####
Lat_regrid=zeros ((lat_n_regrid,1));
Lat_bound_regrid = zeros ((lat_n_regrid,2)); Lat_bound_regrid[0,0]=-90;  Lat_bound_regrid[0,1]=Lat_bound_regrid[0,0] + (180/lat_n_regrid); Lat_regrid[0,0]=(Lat_bound_regrid[0,0]+Lat_bound_regrid[0,1])/2
for ii in range(1,lat_n_regrid):
    Lat_bound_regrid[ii,0]=Lat_bound_regrid[ii-1,1]
    Lat_bound_regrid[ii,1]=Lat_bound_regrid[ii,0] +  (180/lat_n_regrid)
    Lat_regrid[ii,0]=(Lat_bound_regrid[ii,0]+Lat_bound_regrid[ii,1])/2

#### Longitude Bounds ####
Lon_regrid=zeros ((lon_n_regrid,1));
Lon_bound_regrid = zeros ((lon_n_regrid,2)); Lon_bound_regrid[0,0]=0;  Lon_bound_regrid[0,1]=Lon_bound_regrid[0,0] + (360/lon_n_regrid); Lon_regrid[0,0]=(Lon_bound_regrid[0,0]+Lon_bound_regrid[0,1])/2
for ii in range(1,lon_n_regrid):
    Lon_bound_regrid[ii,0]=Lon_bound_regrid[ii-1,1]
    Lon_bound_regrid[ii,1]=Lon_bound_regrid[ii,0] +  (360/lon_n_regrid)
    Lon_regrid[ii,0]=(Lon_bound_regrid[ii,0]+Lon_bound_regrid[ii,1])/2

Lon_img, Lat_img = np.meshgrid(Lon_regrid, Lat_regrid)

land_mask = empty ((lat_n_regrid, lon_n_regrid)) * nan
ocean_mask= maskoceans(Lon_img-180, Lat_img, land_mask)
for ii in range(lat_n_regrid):
    for jj in range(lon_n_regrid):
        if ma.is_masked(ocean_mask[ii,jj]):
            land_mask[ii,jj]=0 # land_mask=0 means the grid cell is not on land (it's in ocean)
        else:
            land_mask[ii,jj]=1 # land_mask=0 means the grid cell is on land (it's not in ocean)
land_mask2=land_mask # The created land_mask's longitude is from -180°-180° - following lines transfer it to 0°-360°
land_mask=empty((180,360)) *nan
land_mask[:,0:int(land_mask.shape[1]/2)]=land_mask2[:,int(land_mask.shape[1]/2):]
land_mask[:,int(land_mask.shape[1]/2):]=land_mask2[:,0:int(land_mask.shape[1]/2)]

Multimodel_Variable_Surface_Ave_Regrid_hist=zeros((len(GCM_Names), lat_n_regrid, lon_n_regrid))# Multimodel surface average of specified variable, regridded
Multimodel_Variable_Surface_Ave_Regrid_rcp85=zeros((len(GCM_Names), lat_n_regrid, lon_n_regrid))
# the following two variables are not masked over land and will be used for plotting only - because all of the variables are regridded over land and ocean - for further calculations and averaging, the land values should be deleted using "land_mask" created above
Multimodel_Variable_Surface_Ave_Regrid_hist_plt=zeros((len(GCM_Names), lat_n_regrid, lon_n_regrid))# Multimodel surface average of specified variable, regridded
Multimodel_Variable_Surface_Ave_Regrid_rcp85_plt=zeros((len(GCM_Names), lat_n_regrid, lon_n_regrid))
#Multimodel_Time_AllYears=zeros(( ((int(end_date_cal)-int(start_date_cal)+1)*12), len(GCM_Names) ))# Stores the corresponding date of each time step of the Variable, to ensure that the code has read the correct years
Multimodel_Time_AllYears_hist=[]
Multimodel_Time_AllYears_rcp85=[]
Multimodel_InputFileNames_hist=[]
Multimodel_InputFileNames_rcp85=[]

######################################
### Historical Period Calculations ###
######################################
for M_i in range(len(GCM_Names)): # M_i=0
    
    GCM=GCM_Names[M_i]
    
    dir_data_in2=(dir_data_in1+ GCM + '/historical/mo/')
    Input_File_Names = [xx for xx in sorted(os.listdir(dir_data_in2)) if xx.startswith(Var_name) and xx.endswith(".nc")] # List all the files in the directory that are .nc and end with year number (to avoid concated files)
    if GCM=='HadGEM2-ES' or GCM=='MIROC-ESM': # Some models have only concated data which the file name ends with concat.nc, so the year characters in their file name is different
        Input_File_Names = [xx for xx in Input_File_Names if ( int(xx[-23:-19])>=int(start_date_cal_hist) and int(xx[-23:-19])<=int(end_date_cal_hist) ) or ( int(xx[-16:-12])>=int(start_date_cal_hist) and int(xx[-16:-12])<=int(end_date_cal_hist) ) or ( int(xx[-23:-19])<=int(start_date_cal_hist) and int(xx[-16:-12])>=int(end_date_cal_hist) )] # Keep only the files that the time range is in the specified time interval
    else:
        Input_File_Names = [xx for xx in Input_File_Names if not xx.endswith("concat.nc") ] # Some models have both decadal files and a concated file, which may result in duplication
        Input_File_Names = [xx for xx in Input_File_Names if ( int(xx[-16:-12])>=int(start_date_cal_hist) and int(xx[-16:-12])<=int(end_date_cal_hist) ) or ( int(xx[-9:-5])>=int(start_date_cal_hist) and int(xx[-9:-5])<=int(end_date_cal_hist) ) or ( int(xx[-16:-12])<=int(start_date_cal_hist) and int(xx[-9:-5])>=int(end_date_cal_hist) )] # Keep only the files that the time range is in the specified time interval

    dir_data_in_file=(dir_data_in2 +Input_File_Names[0])
    dset = Dataset(dir_data_in_file)
    ## dset.variables  # Shows the variables in the .nc file
    
    if GCM=='HadGEM2-ES' or GCM=='MIROC-ESM' :
        lat_char='LAT'
        lon_char='LON'
        time_char='TIME'
        Var_char='THETAO'
    else:
        lat_char='lat'
        lon_char='lon'
        time_char='time'
        Var_char='thetao'
    
    # Reading lat and lon values from the first file
    Lat=np.asarray(dset.variables[lat_char][:])
    Lon=np.asarray(dset.variables[lon_char][:])
    if len(Lat.shape)==2:
        # 1 for curvelinear
        curvilinear=1
    else:
        # 0 for regular
        curvilinear=0 
    
    Variable_Surface_AllYears=[]
    Time_AllYears=[]   

    for F_i in  range(len(Input_File_Names[:])): # F_i=0
        
        try:#open netcdf file #MFDataset function can read multiple netcdf files, note * - wildcard which is used to do it                
            dir_data_in_file=(dir_data_in2 +Input_File_Names[F_i])
            dset = Dataset(dir_data_in_file)
            ## dset.variables  # Shows the variables in the .nc file
        except:
            print ('There is no such'+ GCM)
            continue
     
        # append time to variable
        Time_dset=dset.variables[time_char]
        # append thetao data to variable
        Var_dset = dset.variables[Var_char]
        # next lines of code are looking for specified dates
        start_date_i=0 # Conter of start_date (to be calculated below)
        end_date_i=0 # Conter of end_date (to be calculated below)
        text = Time_dset.units.split(' ') # Splits the Time_dset.UNITS wherever there's a ' ' separator
        Data_start_date=text[2][:4] # From TEXT, the (2+1)th row and the first 4 chars show the begenning year of data 
        # the loop goes through all the time indeces and checks if the time equals desired value
        ## This loop finds which row in the Time_dset variable is in the desired start_date_cal and end_date_cal range
        for ii in range(len(Time_dset[:])): 
            # converting time to date 
            date=str(num2date(Time_dset[ii],units=Time_dset.units,calendar=Time_dset.calendar))
            date_sp = date.split('-') # Splits the TIMES.UNITS wherever there's a '-' separator
            # if date_sp[0] which is year equals specified start year, append index of the year to start_date variable
            if (int(date_sp[0])>=int(start_date_cal_hist) and int(date_sp[0])<=int(end_date_cal_hist)): # Splits the DATE wherever there's a '-' separator and saves it as date_yr . Now the first row of WORDS would show the year
                Variable_Surface_AllYears.append(np.asarray(Var_dset[ii,0,:,:])) # Variable(time, lev, rlat, rlon) - Variable of all the desired years, only on the surface
                Time_AllYears.append(date)
    #dset.close() # closing netcdf file    
    Variable_Surface_AllYears = np.asarray(Variable_Surface_AllYears) # Converts the appended list to an array
    
    Variable_Surface_Ave=nanmean(Variable_Surface_AllYears,0) # Averages over the index 0 (over time)
    Variable_Surface_Ave_mask=ma.masked_where(Variable_Surface_Ave >= 1e19, Variable_Surface_Ave) # Masking grids with nodata(presented by value=1E20 by GCM)
    
    Data_regrid=Variable_Surface_Ave # Dataset (matrix) to be regrided
    data_vec=[] # Data to be regrided, converted into a vector, excluding nan/missing arrays
    lon_vec=[] # Lon converted to a vector
    lat_vec=[]
    for ii in range(Data_regrid.shape[0]): # Loop over rows
        for jj in range(Data_regrid.shape[1]): # Loop over columns
            # creating variables, used to calculate averages for each lat lon grid point
            if Data_regrid[ii,jj] < 1e19: # Values equal to 1e20 are empty arrays # Only selecting grids with available data
                data_vec.append(Data_regrid[ii,jj])
                # appending lat lons, which will be used in regriding the data
                # check for curvelinear coordinates
                if curvilinear==1:
                    # check if the GCM is GFDL, as GFDL lons start at -270 and go to 90
                    if GCM[:4]=='GFDL':
                        if Lon[ii,jj]<=0:
                            # converting -270 - 90 lon range to 0 - 360 lon range
                            lon_vec.append(Lon[ii,jj]+360)
                        else:  
                            lon_vec.append(Lon[ii,jj])
                    else:
                        lon_vec.append(Lon[ii,jj])
                    lat_vec.append(Lat[ii,jj])
                else:
                    lon_vec.append(Lon[jj])
                    lat_vec.append(Lat[ii]) 
                    
    # converting to numpy arrays, as regriding does not work with lists
    lon_vec = np.asarray(lon_vec)
    lat_vec = np.asarray(lat_vec)
    data_vec = np.asarray(data_vec)
    lon_lat_vec=zeros((lon_vec.shape[0],2)) # Combining lat and lon vectors into one vector
    lon_lat_vec[:,0]=lon_vec
    lon_lat_vec[:,1]=lat_vec
    
    # converting the old grid to a new, using nearest neighbour interpolation
    #Data_regrid = ml.griddata(lon_vec, lat_vec, data_vec, Lon_regrid, Lat_regrid, interp='linear')
    Data_regrid = griddata(lon_lat_vec, data_vec, (Lon_img, Lat_img), method='nearest') # Options: method='linear' , method='cubic'

    Variable_Surface_Ave_Regrid=np.asarray(Data_regrid) # Converts to numpy array - fills empty (masked) gridcells with nan
    Multimodel_Variable_Surface_Ave_Regrid_hist_plt[M_i,:,:]=Variable_Surface_Ave_Regrid
    
    Variable_Surface_Ave_Regrid [ land_mask == 1] = nan # masking over land, so grid cells that fall on land area will be deleted
    Multimodel_Variable_Surface_Ave_Regrid_hist[M_i,:,:]=Variable_Surface_Ave_Regrid
    #Multimodel_Time_AllYears[M_i,:]=Time_AllYears
    Multimodel_Time_AllYears_hist.append(Time_AllYears)
    Multimodel_InputFileNames_hist.append(Input_File_Names)

    ################################################
    ### Ploting for each model (with pcolormesh) ###
    Plot_Var=Multimodel_Variable_Surface_Ave_Regrid_hist_plt[M_i,:,:] - 273.15
    Var_plot_unit='Unit = °C'
    # create figure
    fig = plt.figure()
    # create an Axes at an arbitrary location, which makes a list of [left, bottom, width, height] values in 0-1 relative figure coordinates:
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    # create Basemap instance.
    m = Basemap(projection='cyl', lat_0=0, lon_0=0)
    m.drawcoastlines(linewidth=1.25)
    m.fillcontinents(color='0.95')
    #m.drawmapboundary(fill_color='0.9')
    # cacluate colorbar ranges
    #bounds_max=float("{0:.0f}".format(np.nanpercentile(Plot_Var, 99.99))) # Upper bound of plotted values to be used for colorbar, which is 99.99th percentile of data, with 0 decimals
    bounds_max=33
    bounds = np.arange(0, bounds_max, bounds_max/33)
    norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    # create a pseudo-color plot over the map
    im1 = m.pcolormesh(Lon_img, Lat_img, Plot_Var, norm=norm, shading='flat', cmap=plt.cm.jet, latlon=True) # Choose colormap: https://matplotlib.org/users/colormaps.html
    m.drawparallels(np.arange(-90.,90.001,30.),labels=[True,False,False,False], linewidth=0.01) # labels = [left,right,top,bottom]
    m.drawmeridians(np.arange(0.,360.,30.),labels=[False,False,False,True], linewidth=0.01) # labels = [left,right,top,bottom]
    # add colorbar
    cbar = m.colorbar(im1,"right", size="3%", pad="2%", extend='max') # extend='both' will extend the colorbar in both sides (upper side and down side)
    cbar.set_label(Var_plot_unit)
    #set title
    ax.set_title(Var_plot_name+', at surface - hist - average of '+start_date_cal_hist+'-'+end_date_cal_hist + ' - ' +str(GCM))
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized() # Maximizes the plot window to save figures in full
    #plt.show()
    fig.savefig(dir_figs+Var_name+'_'+ str(GCM)+'_surface_hist-' +start_date_cal_hist+'-'+end_date_cal_hist + '.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    
    print('Model '+str(GCM)+' - hist - processed successfully')
    
#################################################
#### Ploting for all models (with pcolormesh) ###
n_r=4 # Number of rows for subplot
n_c=4 # Number of columns for subplot
n_range=list(range(len(GCM_Names)))
bounds_max=33
bounds = np.arange(0, bounds_max, bounds_max/33)
norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
Var_plot_unit='Unit = °C'

fig=plt.figure()

for ii in n_range:
    ax = fig.add_subplot(n_r,n_c,ii+1)
    Plot_Var=Multimodel_Variable_Surface_Ave_Regrid_hist_plt[ii,:,:] - 273.15
    m = Basemap(projection='cyl', lat_0=0, lon_0=0)
    m.drawcoastlines(linewidth=1.25)
    m.fillcontinents(color='0.95')
    #m.drawmapboundary(fill_color='0.9')
    m.drawparallels(np.arange(-90.,90.001,30.),labels=[True,False,False,False], linewidth=0.01) # labels = [left,right,top,bottom]
    if ii+1 >= n_range[-n_c+1]: # Adds longitude ranges only to the last subplots that appear at the bottom of plot
        m.drawmeridians(np.arange(0.,360.,60.),labels=[False,False,False,True], linewidth=0.01) # labels = [left,right,top,bottom]
    im1 = m.pcolormesh(Lon_img, Lat_img, Plot_Var, norm=norm, shading='flat', cmap=plt.cm.jet, latlon=True) # Choose colormap: https://matplotlib.org/users/colormaps.html
    plt.title(GCM_Names[ii])
plt.suptitle( ( Var_plot_name+', at surface - hist - average of '+start_date_cal_hist+'-'+end_date_cal_hist ), fontsize=18)
cbar = plt.colorbar(cax=plt.axes([0.93, 0.1, 0.015, 0.8]), extend='max') # cax = [left position, bottom postion, width, height] 
cbar.set_label(Var_plot_unit)
#plt.show()
mng = plt.get_current_fig_manager()
mng.window.showMaximized() # Maximizes the plot window to save figures in full
fig.savefig(dir_figs+Var_name+'_AllGCMs_surface_hist-' +start_date_cal_hist+'-'+end_date_cal_hist + '.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
plt.close()


###########################
### RCP8.5 Calculations ###
###########################
for M_i in range(len(GCM_Names)): # M_i=0
    
    GCM=GCM_Names[M_i]
    
    dir_data_in2=(dir_data_in1+ GCM + '/rcp85/mo/')
    Input_File_Names = [xx for xx in sorted(os.listdir(dir_data_in2)) if xx.startswith(Var_name) and xx.endswith(".nc")] # List all the files in the directory that are .nc and end with year number (to avoid concated files)
    if GCM=='MIROC-ESM-CHEM': # Some models have only concated data which the file name ends with concat.nc, so the year characters in their file name is different
        Input_File_Names = [xx for xx in Input_File_Names if ( int(xx[-23:-19])>=int(start_date_cal_rcp85) and int(xx[-23:-19])<=int(end_date_cal_rcp85) ) or ( int(xx[-16:-12])>=int(start_date_cal_rcp85) and int(xx[-16:-12])<=int(end_date_cal_rcp85) ) or ( int(xx[-23:-19])<=int(start_date_cal_rcp85) and int(xx[-16:-12])>=int(end_date_cal_rcp85) )] # Keep only the files that the time range is in the specified time interval
    else:
        Input_File_Names = [xx for xx in Input_File_Names if not xx.endswith("concat.nc") ] # Some models have both decadal files and a concated file, which may result in duplication
        Input_File_Names = [xx for xx in Input_File_Names if ( int(xx[-16:-12])>=int(start_date_cal_rcp85) and int(xx[-16:-12])<=int(end_date_cal_rcp85) ) or ( int(xx[-9:-5])>=int(start_date_cal_rcp85) and int(xx[-9:-5])<=int(end_date_cal_rcp85) ) or ( int(xx[-16:-12])<=int(start_date_cal_rcp85) and int(xx[-9:-5])>=int(end_date_cal_rcp85) )] # Keep only the files that the time range is in the specified time interval

    dir_data_in_file=(dir_data_in2 +Input_File_Names[0])
    dset = Dataset(dir_data_in_file)
    ## dset.variables  # Shows the variables in the .nc file
    
    if GCM=='MIROC-ESM-CHEM' :
        lat_char='LAT'
        lon_char='LON'
        time_char='TIME'
        Var_char='THETAO'
    else:
        lat_char='lat'
        lon_char='lon'
        time_char='time'
        Var_char='thetao'
    
    # Reading lat and lon values from the first file
    Lat=np.asarray(dset.variables[lat_char][:])
    Lon=np.asarray(dset.variables[lon_char][:])
    if len(Lat.shape)==2:
        # 1 for curvelinear
        curvilinear=1
    else:
        # 0 for regular
        curvilinear=0 
    
    Variable_Surface_AllYears=[]
    Time_AllYears=[]   

    for F_i in  range(len(Input_File_Names[:])): # F_i=0
        
        try:#open netcdf file #MFDataset function can read multiple netcdf files, note * - wildcard which is used to do it                
            dir_data_in_file=(dir_data_in2 +Input_File_Names[F_i])
            dset = Dataset(dir_data_in_file)
            ## dset.variables  # Shows the variables in the .nc file
        except:
            print ('There is no such'+ GCM)
            continue
     
        # append time to variable
        Time_dset=dset.variables[time_char]
        # append thetao data to variable
        Var_dset = dset.variables[Var_char]
        # next lines of code are looking for specified dates
        start_date_i=0 # Conter of start_date (to be calculated below)
        end_date_i=0 # Conter of end_date (to be calculated below)
        text = Time_dset.units.split(' ') # Splits the Time_dset.UNITS wherever there's a ' ' separator
        Data_start_date=text[2][:4] # From TEXT, the (2+1)th row and the first 4 chars show the begenning year of data 
        # the loop goes through all the time indeces and checks if the time equals desired value
        ## This loop finds which row in the Time_dset variable is in the desired start_date_cal and end_date_cal range
        for ii in range(len(Time_dset[:])): 
            # converting time to date 
            date=str(num2date(Time_dset[ii],units=Time_dset.units,calendar=Time_dset.calendar))
            date_sp = date.split('-') # Splits the TIMES.UNITS wherever there's a '-' separator
            # if date_sp[0] which is year equals specified start year, append index of the year to start_date variable
            if (int(date_sp[0])>=int(start_date_cal_rcp85) and int(date_sp[0])<=int(end_date_cal_rcp85)): # Splits the DATE wherever there's a '-' separator and saves it as date_yr . Now the first row of WORDS would show the year
                Variable_Surface_AllYears.append(np.asarray(Var_dset[ii,0,:,:])) # Variable(time, lev, rlat, rlon) - Variable of all the desired years, only on the surface
                Time_AllYears.append(date)
    #dset.close() # closing netcdf file    
    Variable_Surface_AllYears = np.asarray(Variable_Surface_AllYears) # Converts the appended list to an array
    
    Variable_Surface_Ave=nanmean(Variable_Surface_AllYears,0) # Averages over the index 0 (over time)
    Variable_Surface_Ave_mask=ma.masked_where(Variable_Surface_Ave >= 1e19, Variable_Surface_Ave) # Masking grids with nodata(presented by value=1E20 by GCM)
    
    Data_regrid=Variable_Surface_Ave # Dataset (matrix) to be regrided
    data_vec=[] # Data to be regrided, converted into a vector, excluding nan/missing arrays
    lon_vec=[] # Lon converted to a vector
    lat_vec=[]
    for ii in range(Data_regrid.shape[0]): # Loop over rows
        for jj in range(Data_regrid.shape[1]): # Loop over columns
            # creating variables, used to calculate averages for each lat lon grid point
            if Data_regrid[ii,jj] < 1e19: # Values equal to 1e20 are empty arrays # Only selecting grids with available data
                data_vec.append(Data_regrid[ii,jj])
                # appending lat lons, which will be used in regriding the data
                # check for curvelinear coordinates
                if curvilinear==1:
                    # check if the GCM is GFDL, as GFDL lons start at -270 and go to 90
                    if GCM[:4]=='GFDL':
                        if Lon[ii,jj]<=0:
                            # converting -270 - 90 lon range to 0 - 360 lon range
                            lon_vec.append(Lon[ii,jj]+360)
                        else:  
                            lon_vec.append(Lon[ii,jj])
                    else:
                        lon_vec.append(Lon[ii,jj])
                    lat_vec.append(Lat[ii,jj])
                else:
                    lon_vec.append(Lon[jj])
                    lat_vec.append(Lat[ii]) 
                    
    # converting to numpy arrays, as regriding does not work with lists
    lon_vec = np.asarray(lon_vec)
    lat_vec = np.asarray(lat_vec)
    data_vec = np.asarray(data_vec)
    lon_lat_vec=zeros((lon_vec.shape[0],2)) # Combining lat and lon vectors into one vector
    lon_lat_vec[:,0]=lon_vec
    lon_lat_vec[:,1]=lat_vec
    
    #Data_regrid = ml.griddata(lon_vec, lat_vec, data_vec, Lon_regrid, Lat_regrid, interp='linear')
    Data_regrid = griddata(lon_lat_vec, data_vec, (Lon_img, Lat_img), method='nearest') # Options: method='linear' , method='cubic'

    Variable_Surface_Ave_Regrid=np.asarray(Data_regrid) # Converts to numpy array - fills empty (masked) gridcells with nan
    Multimodel_Variable_Surface_Ave_Regrid_rcp85_plt[M_i,:,:]=Variable_Surface_Ave_Regrid
    
    Variable_Surface_Ave_Regrid [ land_mask == 1] = nan # masking over land, so grid cells that fall on land area will be deleted
    Multimodel_Variable_Surface_Ave_Regrid_rcp85[M_i,:,:]=Variable_Surface_Ave_Regrid
    #Multimodel_Time_AllYears[M_i,:]=Time_AllYears
    Multimodel_Time_AllYears_rcp85.append(Time_AllYears)
    Multimodel_InputFileNames_rcp85.append(Input_File_Names)

    ################################################
    ### Ploting for each model (with pcolormesh) ###
    Plot_Var=Multimodel_Variable_Surface_Ave_Regrid_rcp85_plt[M_i,:,:] - 273.15
    Var_plot_unit='Unit = °C'
    # create figure
    fig = plt.figure()
    # create an Axes at an arbitrary location, which makes a list of [left, bottom, width, height] values in 0-1 relative figure coordinates:
    ax = fig.add_axes([0.1,0.1,0.8,0.8])
    # create Basemap instance.
    m = Basemap(projection='cyl', lat_0=0, lon_0=0)
    m.drawcoastlines(linewidth=1.25)
    m.fillcontinents(color='0.95')
    #m.drawmapboundary(fill_color='0.9')
    # cacluate colorbar ranges
    #bounds_max=float("{0:.0f}".format(np.nanpercentile(Plot_Var, 99.99))) # Upper bound of plotted values to be used for colorbar, which is 99.99th percentile of data, with 0 decimals
    bounds_max=33
    bounds = np.arange(0, bounds_max, bounds_max/33)
    norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    # create a pseudo-color plot over the map
    im1 = m.pcolormesh(Lon_img, Lat_img, Plot_Var, norm=norm, shading='flat', cmap=plt.cm.jet, latlon=True) # Choose colormap: https://matplotlib.org/users/colormaps.html
    m.drawparallels(np.arange(-90.,90.001,30.),labels=[True,False,False,False], linewidth=0.01) # labels = [left,right,top,bottom]
    m.drawmeridians(np.arange(0.,360.,30.),labels=[False,False,False,True], linewidth=0.01) # labels = [left,right,top,bottom]
    # add colorbar
    cbar = m.colorbar(im1,"right", size="3%", pad="2%", extend='max') # extend='both' will extend the colorbar in both sides (upper side and down side)
    cbar.set_label(Var_plot_unit)
    #set title
    ax.set_title(Var_plot_name+', at surface - rcp8.5 - average of '+start_date_cal_rcp85+'-'+end_date_cal_rcp85 + ' - ' +str(GCM))
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized() # Maximizes the plot window to save figures in full
    #plt.show()
    fig.savefig(dir_figs+Var_name+'_'+ str(GCM)+'_surface_rcp85-' +start_date_cal_rcp85+'-'+end_date_cal_rcp85 + '.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
    plt.close()
    
    print('Model '+str(GCM)+' - rcp85 - processed successfully')
    
#################################################
#### Ploting for all models (with pcolormesh) ###
n_r=4 # Number of rows for subplot
n_c=4 # Number of columns for subplot
n_range=list(range(len(GCM_Names)))
bounds_max=33
bounds = np.arange(0, bounds_max, bounds_max/33)
norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
Var_plot_unit='Unit = °C'

fig=plt.figure()

for ii in n_range:
    ax = fig.add_subplot(n_r,n_c,ii+1)
    Plot_Var=Multimodel_Variable_Surface_Ave_Regrid_rcp85_plt[ii,:,:] - 273.15
    m = Basemap(projection='cyl', lat_0=0, lon_0=0)
    m.drawcoastlines(linewidth=1.25)
    m.fillcontinents(color='0.95')
    #m.drawmapboundary(fill_color='0.9')
    m.drawparallels(np.arange(-90.,90.001,30.),labels=[True,False,False,False], linewidth=0.01) # labels = [left,right,top,bottom]
    if ii+1 >= n_range[-n_c+1]: # Adds longitude ranges only to the last subplots that appear at the bottom of plot
        m.drawmeridians(np.arange(0.,360.,60.),labels=[False,False,False,True], linewidth=0.01) # labels = [left,right,top,bottom]
    im1 = m.pcolormesh(Lon_img, Lat_img, Plot_Var, norm=norm, shading='flat', cmap=plt.cm.jet, latlon=True) # Choose colormap: https://matplotlib.org/users/colormaps.html
    plt.title(GCM_Names[ii])
plt.suptitle( ( Var_plot_name+', at surface - rcp8.5 - average of '+start_date_cal_rcp85+'-'+end_date_cal_rcp85 ), fontsize=18)
cbar = plt.colorbar(cax=plt.axes([0.93, 0.1, 0.015, 0.8]), extend='max') # cax = [left position, bottom postion, width, height] 
cbar.set_label(Var_plot_unit)
#plt.show()
mng = plt.get_current_fig_manager()
mng.window.showMaximized() # Maximizes the plot window to save figures in full
fig.savefig(dir_figs+Var_name+'_AllGCMs_surface_rcp85-' +start_date_cal_rcp85+'-'+end_date_cal_rcp85 + '.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
plt.close()


### Change in SST - rcp85 compared to hist ###
n_r=4 # Number of rows for subplot
n_c=4 # Number of columns for subplot
n_range=list(range(len(GCM_Names)))
bounds_max=6
bound_ranges=bounds_max/20
bounds = np.arange(-1*bounds_max, bounds_max+bound_ranges, bound_ranges)
norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
Var_plot_unit='Unit = °C'

fig=plt.figure()

for ii in n_range:
    ax = fig.add_subplot(n_r,n_c,ii+1)
    Plot_Var=Multimodel_Variable_Surface_Ave_Regrid_rcp85_plt[ii,:,:] - Multimodel_Variable_Surface_Ave_Regrid_hist_plt[ii,:,:]
    m = Basemap(projection='cyl', lat_0=0, lon_0=0)
    m.drawcoastlines(linewidth=1.25)
    m.fillcontinents(color='0.95')
    #m.drawmapboundary(fill_color='0.9')
    m.drawparallels(np.arange(-90.,90.001,30.),labels=[True,False,False,False], linewidth=0.01) # labels = [left,right,top,bottom]
    if ii+1 >= n_range[-n_c+1]: # Adds longitude ranges only to the last subplots that appear at the bottom of plot
        m.drawmeridians(np.arange(0.,360.,60.),labels=[False,False,False,True], linewidth=0.01) # labels = [left,right,top,bottom]
    im1 = m.pcolormesh(Lon_img, Lat_img, Plot_Var, norm=norm, shading='flat', cmap=plt.cm.RdBu_r, latlon=True) # Choose colormap: https://matplotlib.org/users/colormaps.html
    
    plt.title(GCM_Names[ii])
plt.suptitle( ('Change in sea surface temperature - rcp8.5 compared to hist - '+start_date_cal_rcp85+'-'+end_date_cal_rcp85+' minus '+ start_date_cal_hist+'-'+end_date_cal_hist), fontsize=18)
cbar = plt.colorbar(cax=plt.axes([0.93, 0.1, 0.015, 0.8]), extend='both') # cax = [left position, bottom postion, width, height] 
cbar.set_label(Var_plot_unit)
#plt.show()
mng = plt.get_current_fig_manager()
mng.window.showMaximized() # Maximizes the plot window to save figures in full
fig.savefig(dir_figs+Var_name+'_AllGCMs_detaSST_rcp85_minus_hist.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
plt.close()

### Change in Southern Ocean SST (south of 50S) ###

### Calculating grid cell area based on Lat and Lon
## Area = R^2 * ( Lon2 - Lon1 ) * ( Sin(Lat2) - Sin(Lat1) )
earth_R = 6378 # Earth Radius - Unit is kilometer (km)
GridCell_Areas = empty((lat_n_regrid, lon_n_regrid )) *nan
for ii in range(lat_n_regrid):
    for jj in range(lon_n_regrid):
        GridCell_Areas [ii,jj] = math.fabs( (earth_R**2) * (math.pi/180) * (Lon_bound_regrid[jj,1] - Lon_bound_regrid[jj,0])  * ( math.sin(math.radians(Lat_bound_regrid[ii,1])) - math.sin(math.radians(Lat_bound_regrid[ii,0]))) )
GridCell_Areas=GridCell_Areas / 1e6 # to convert the area to million km2
GridCell_Areas [ land_mask == 1] = nan # masking over land, so grid cells that fall on land area will be deleted

SST_South50S_hist=zeros((len(GCM_Names),1)) # Historical average of ocean SST south of South of 50°S
for ii in range(len(GCM_Names)):
    cell_var = Multimodel_Variable_Surface_Ave_Regrid_hist[ii,0:40,:]
    cell_area=GridCell_Areas [0:40,:]    
    SST_South50S_hist[ii,0]= np.nansum( np.multiply(cell_var, cell_area) ) / np.nansum(cell_area)

SST_South50S_rcp85=zeros((len(GCM_Names),1)) # RCP8.5 average of ocean SST south of South of 50°S
for ii in range(len(GCM_Names)):
    cell_var = Multimodel_Variable_Surface_Ave_Regrid_rcp85[ii,0:40,:]
    cell_area=GridCell_Areas [0:40,:]    
    SST_South50S_rcp85[ii,0]= np.nansum( np.multiply(cell_var, cell_area) ) / np.nansum(cell_area)

Delta_SST_South50S = SST_South50S_rcp85 - SST_South50S_hist # Average change in ocean SST south of South of 50°S

xx=SST_South50S_hist-273.15
yy=Delta_SST_South50S


fig, ax = plt.subplots()
ax.scatter(xx, yy, s=200, marker='d', c='r')
ax.scatter(xx, yy, s=20, marker='d', c='b')
for ii, txt in enumerate(GCM_Names):
    ax.annotate(txt, (xx[ii],yy[ii]), fontsize=14)
plt.xlabel('Southern Ocean SST (South of 50°S-hist ave) [°C]', fontsize=18)
plt.xlim(0, 6)
plt.xticks( fontsize = 18)
plt.ylabel('Δ SST (rcp8.5 minus hist) [°C]', fontsize=18)
plt.ylim(0, 2.5)
plt.yticks( fontsize = 18)
plt.title( ('Southern Ocean (South of 50°S) - Change in SST (rcp8.5 '+start_date_cal_rcp85+'-'+end_date_cal_rcp85+' minus hist '+ start_date_cal_hist+'-'+end_date_cal_hist)+') Vs. SST hist ave', fontsize=18)
mng = plt.get_current_fig_manager()
mng.window.showMaximized() # Maximizes the plot window to save figures in full
fig.savefig(dir_figs+Var_name+'_AllGCMs_detaSST_S50S_scatter.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
plt.close()



SST_Global_hist=zeros((len(GCM_Names),1)) # Historical average of ocean SST south of South of 50°S
for ii in range(len(GCM_Names)):
    cell_var = Multimodel_Variable_Surface_Ave_Regrid_hist[ii,:,:]
    cell_area=GridCell_Areas    
    SST_Global_hist[ii,0]= np.nansum( np.multiply(cell_var, cell_area) ) / np.nansum(cell_area)

SST_Global_rcp85=zeros((len(GCM_Names),1)) # RCP8.5 average of ocean SST south of South of 50°S
for ii in range(len(GCM_Names)):
    cell_var = Multimodel_Variable_Surface_Ave_Regrid_rcp85[ii,:,:]
    cell_area=GridCell_Areas 
    SST_Global_rcp85[ii,0]= np.nansum( np.multiply(cell_var, cell_area) ) / np.nansum(cell_area)

Delta_SST_Global = SST_Global_rcp85 - SST_Global_hist # Average change in ocean SST south of South of 50°S

xx=SST_Global_hist-273.15
yy=Delta_SST_Global


fig, ax = plt.subplots()
ax.scatter(xx, yy, s=200, marker='d', c='r')
ax.scatter(xx, yy, s=20, marker='d', c='b')
for ii, txt in enumerate(GCM_Names):
    ax.annotate(txt, (xx[ii],yy[ii]), fontsize=14)
plt.xlabel('Global SST (hist ave) [°C]', fontsize=18)
plt.xlim(17, 20)
plt.xticks( fontsize = 18)
plt.ylabel('Δ SST (rcp8.5 minus hist) [°C]', fontsize=18)
plt.ylim(1.5, 4)
plt.yticks( fontsize = 18)
plt.title( ('Global - Change in SST (rcp8.5 '+start_date_cal_rcp85+'-'+end_date_cal_rcp85+' minus hist '+ start_date_cal_hist+'-'+end_date_cal_hist)+') Vs. SST hist ave', fontsize=18)
mng = plt.get_current_fig_manager()
mng.window.showMaximized() # Maximizes the plot window to save figures in full
fig.savefig(dir_figs+Var_name+'_AllGCMs_detaSST_global_scatter.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
plt.close()








