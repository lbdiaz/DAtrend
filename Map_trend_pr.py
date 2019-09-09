#!/usr/bin/env python3

#Script to plot trends in CMIP6 models for different 
#DAMIP simulations (historical, hist-nat, hist-GHG, hist-stratO3) 

#%%
#Import required libraries

import numpy as np
import xarray as xr
import os.path
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cartopy.crs as ccrs	
import pandas
import cartopy.feature 	
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

#%%
#Define path to save figure

FIG_PATH = '/home/ldiaz/datos/Archivos/Investigacion/Figuras/DAtrend/'

#%%
#Define parameters to load data

#Define variable name
VAR = 'pr'

#Factor of unit conversion (1 if not change)
#86400*90*10 for precipitation trend to pass from kg(m^2*s/year) to mm/day/summer/decade
#1/100*10 for sea level pressure trend to pass from Pa/year to hPa/decade
CONV_UNIT=86400*90*10

#Define sample frequency
FREQ = 'mon'

#List of experiments to be used
EXPERIMENT = ['historical', 'hist-nat', 'hist-GHG', 'hist-stratO3']

#Define models to be load, with their corresponding variant_label and grid_label
MODEL = ['IPSL-CM6A-LR', 'MIROC6','CanESM5']
VARIANT_LABEL = ['i1p1f1', 'i1p1f1', 'i1p1f1']
GRID_LABEL = ['gr', 'gn', 'gn']

#Period for each experimet to be used
PERIOD = ['185001-201412', '185001-202012', '185001-202012', '185001-202012', '185001-202012']

#%%
#Define period of time and region to be used for computing the trends
#Region to plot. Following Vera and Diaz (2015)

LATMIN = -60
LATMAX = 15

LONMIN = 270
LONMAX = 330

YEARMIN=1901
YEARMAX=2014

#As we are only interested in DJF season, months are chosen so we don't have an incomplete season

TIMEMIN = str(YEARMIN)+'-04-01'
TIMEMAX = str(YEARMAX)+'-08-31'

#Define region of the box to show in the map
LATMIN_SESA = -40
LATMAX_SESA = -26
LONMIN_SESA = 297
LONMAX_SESA = 305

#%%
#Create dictionaries with latitdes, longitudes, trends and p-values trends  for multi-member ensemble mean from each experiment (first elemnent of tuple) and each model (second elemnent of tuple)

lat_exp_mod = {}
lon_exp_mod = {}
tend_var_MM = {}
pv_tend_var_MM = {}

# Compute trends in region selected for each member
ds = []

for i in range(len(EXPERIMENT)):
    for j in range(len(MODEL)):
        #Define path to model data
        PATH = '/datos3/CMIP6/' + EXPERIMENT[i] + '/' + FREQ + '/' + VAR + '/'
        #Define file name
        ARCHIVO = VAR + '_A' + FREQ + '_' + MODEL[j] + '_' + EXPERIMENT[i] + '_'  + VARIANT_LABEL[j] + '_'  + GRID_LABEL[j] + '_' + PERIOD[i] + '.nc4'
        FILE = PATH + ARCHIVO
        # Check if file exists for given experiment and model. In case that not, put nans in that element of the dictionary.
        if (os.path.isfile(FILE)==False):
            lat_exp_mod[(i,j)]=np.nan
            lon_exp_mod[(i,j)]=np.nan
            tend_var_MM[(i,j)]=np.nan
            pv_tend_var_MM[(i,j)]=np.nan            
        else:
            #Open data with xarray
            ds = xr.open_dataset(FILE, decode_coords=False)
            #Select latitudinal, longitudinal and temporal range
            ds_reg = ds.sel(lat=slice(LATMIN, LATMAX), lon=slice(LONMIN,LONMAX), time=slice(TIMEMIN, TIMEMAX))
            #Compute seasonals (3-month) average (DJF season)
            ds_reg_3m = ds_reg.resample(time='QS-DEC').mean()          
            #Select summer average (DJF season)
            ds_reg_djf = ds_reg_3m.sel(time=ds_reg_3m['time.month']==12)                      
            #Compute multi-member means
            ds_MM_djf=(ds_reg_djf.mean(dim='ensemble')) 
            #Obtain latitude and longitude from dataset
            lat_exp_mod[(i,j)]=ds_MM_djf['lat']
            lon_exp_mod[(i,j)]=ds_MM_djf['lon']                   
            #Compute trends for each gridpoint
            ds_MM_djf_stack=ds_MM_djf.stack(points=['lat', 'lon'])
            trends = np.empty_like(ds_MM_djf_stack[VAR][0,:])
            pval = np.empty_like(ds_MM_djf_stack[VAR][0,:])
            for k in range(ds_MM_djf_stack[VAR].shape[1]):
                y = ds_MM_djf_stack[VAR][:, k]
                [trends[k], interc, r_va, pval[k], z] = stats.linregress(np.arange(len(ds_MM_djf_stack['time'])), y)                    
            # Save trends in the dictionary
            #Unit of trends are kg(m^2*s*year). Change to mm/day/summer/decade
            tend_var_MM[(i,j)] = np.reshape(trends*CONV_UNIT, (len(ds_MM_djf['lat']), len(ds_MM_djf['lon'])))
            pv_tend_var_MM[(i,j)] = np.reshape(pval, (len(ds_MM_djf['lat']), len(ds_MM_djf['lon'])))
         
#%%

#Plot figure for all experiments and models

#Define figure
fig, ax = plt.subplots(figsize=(2*len(EXPERIMENT),2.2*len(MODEL)))

#Define grid for subplots
gs = gridspec.GridSpec(len(MODEL),len(EXPERIMENT),width_ratios=2*np.ones(len(EXPERIMENT)),height_ratios=2.2*np.ones(len(MODEL)))     

#Define maxim and minim value to use in the plot 
vmin=-10.5
vmax=10.5

#Define levels and colors to be used
levels = np.linspace(vmin, vmax,  15)
cmap=plt.get_cmap('bwr_r')


# Make one plot for each experiment and model
for i in range(len(EXPERIMENT)):
    for j in range(len(MODEL)):
        #Condition to exclude those models with no data
        if np.sum(pandas.isnull(lat_exp_mod[(i,j)]))==0:
            #Subplot position and define projection to be used
            ax = plt.subplot(gs[i+j*len(EXPERIMENT)],projection=ccrs.PlateCarree(central_longitude=180))
            #Grid for latitudes/longitudes for that model
            lons, lats = np.meshgrid(lon_exp_mod[(i,j)], lat_exp_mod[(i,j)])
            #Define projected lat/lon
            crs_latlon = ccrs.PlateCarree()
            #Define lat/lon plot limits
            ax.set_extent([LONMIN, LONMAX, LATMIN, LATMAX], crs=crs_latlon)
            #Plot fill contour of trends
            im=ax.contourf(lons, lats, tend_var_MM[(i,j)], levels, cmap=cmap, transform=crs_latlon, extend='both')
            #Plot line contour of 5% significance p-level
            ax.contour(lons, lats, pv_tend_var_MM[(i,j)],levels=[0.05],colors='k',linewidths=1 , transform=crs_latlon) 
            #Add map fearures like coasts, border
            ax.add_feature(cartopy.feature.COASTLINE)
            ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
            #Add map fearures like coasts, border
            ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
            #Define x and y ticks format
            ax.set_xticks(np.linspace(LONMIN, LONMAX,  4), crs=crs_latlon)
            ax.set_yticks(np.linspace(LATMIN, LATMAX,  6), crs=crs_latlon)
            ax.tick_params(axis='both', which='major', labelsize=6)
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)
            #Add subplot title                       
            ax.set_title(MODEL[j] + ' ' + EXPERIMENT[i], fontsize="7") 
            # Plot rectangle for defined box
            ax.plot([LONMIN_SESA, LONMIN_SESA], [LATMIN_SESA, LATMAX_SESA],
                    color='#006837', linestyle='--',
                    transform=crs_latlon,
                    )

            ax.plot([LONMAX_SESA, LONMAX_SESA], [LATMIN_SESA, LATMAX_SESA],
                    color='#006837', linestyle='--',
                    transform=crs_latlon,
                    )

            ax.plot([LONMIN_SESA, LONMAX_SESA], [LATMIN_SESA, LATMIN_SESA],
                    color='#006837', linestyle='--',
                    transform=crs_latlon,
                    )

            ax.plot([LONMIN_SESA, LONMAX_SESA], [LATMAX_SESA, LATMAX_SESA],
                    color='#006837', linestyle='--',
                    transform=crs_latlon,
                    )           
            
# Define subplots arrangement
fig.subplots_adjust(left=0.03, bottom=0.03, right=0.91,top=0.93, wspace=0.3 ,hspace=0.2)   

#Plot colorbar
cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.5])
fig.colorbar(im, cax=cbar_ax,orientation='vertical')

#Add main title for the figure
plt.figtext(1.2/len(EXPERIMENT),.97, 'Trend ' + VAR + ' (mm/day/summer/decade) ' + str(YEARMIN+1) + '-' + str(YEARMAX), fontsize=12)

#Save figure
fig.savefig(FIG_PATH + 'TrendDJF_' + VAR + '_' + str(YEARMIN+1) + '_' + str(YEARMAX)+ '.png', dpi=500, bbox_inches='tight')