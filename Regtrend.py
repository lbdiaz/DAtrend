#!/usr/bin/env python3

#Script to compare precipitation trends in CMIP6 models for different 
#DAMIP simulations (historical, hist-nat, hist-GHG, hist-stratO3) 

#%%
#Import required libraries

#import sys
import numpy as np
import xarray as xr
import os.path
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#%%
#Define path to save figure

FIG_PATH = '/home/ldiaz/datos/Archivos/Investigacion/Figuras/DA_PR_SESA/'

#%%
#Define parameters to load data

#Define variable name
VAR = 'pr'

#Define sample frequency
FREQ = 'mon'

#List of experiments to be used
EXPERIMENT = ['historical', 'hist-nat', 'hist-GHG', 'hist-stratO3', 'hist-sol']

#Define models to be load, with their corresponding variant_label and grid_label
MODEL = ['IPSL-CM6A-LR', 'MIROC6','CanESM5']
VARIANT_LABEL = ['i1p1f1', 'i1p1f1', 'i1p1f1']
GRID_LABEL = ['gr', 'gn', 'gn']

#Period for each experimet to be used
PERIOD = ['185001-201412', '185001-202012', '185001-202012', '185001-202012', '185001-202012']

#%%
#Define period of time and region to be used for computing the trends
#Region is defined in a similar way than Vera and Diaz (2015)

REGION_NAME='SESA'

LATMIN = -40
LATMAX = -26

LONMIN = 295
LONMAX = 300

YEARMIN=1950
YEARMAX=2014

#As we are only interested in DJF season, months are chosen so we don't have an incomplete season

TIMEMIN = str(YEARMIN)+'-04-01'
TIMEMAX = str(YEARMAX)+'-08-31'

#%%
#Create dictionaries with trends and p-values for regressions for all members from each experiment (first elemnent of tuple) and each model (second elemnent of tuple)

tend_var_reg_ave = {}
pv_tend_var_reg_ave = {}

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
            tend_var_reg_ave[(i,j)]=np.nan
            pv_tend_var_reg_ave[(i,j)]=np.nan
        else:
            #Open data with xarray
            ds = xr.open_dataset(FILE, decode_coords=False)
            #Select latitudinal, longitudinal and temporal range
            ds_reg = ds.sel(lat=slice(LATMIN, LATMAX), lon=slice(LONMIN,LONMAX), time=slice(TIMEMIN, TIMEMAX))
            #Compute seasonals (3-month) average (DJF season)
            ds_reg_3m = ds_reg.resample(time='QS-DEC').mean()          
            #Select summer average (DJF season)
            ds_reg_djf = ds_reg_3m.sel(time=ds_reg_3m['time.month']==12)                      
            #Compute regional average
            ds_regave_djf=(ds_reg_djf.mean(dim='lat')).mean(dim='lon')    
            #Compute trends            
            trends = np.empty([len(ds_regave_djf['ensemble'])])  
            pval = np.empty([len(ds_regave_djf['ensemble'])])  
            for m in range(len(ds_regave_djf['ensemble'])):
                [trends[m], interc, r_va, pval[m], z] = stats.linregress(np.arange(np.size(ds_regave_djf[VAR],0)),ds_regave_djf[VAR][:,m])
            # Save trends in the dictionary
            #Unit of trends are kg(m^2*s*year). Change to mm/day/summer/decade
            tend_var_reg_ave[(i,j)] = trends*86400*90*10
            pv_tend_var_reg_ave[(i,j)] = pval
         
#%%

#Create figure
fig, ax = plt.subplots()

#Define colors to be used
colores = np.array( ['k', 'r', 'b'])

#Define vector to use in x-axes
T=np.arange(0,len(EXPERIMENT))

for i in range(len(EXPERIMENT)):
    for j in range(len(MODEL)):
        #Plot all members trend
        ax.plot(np.repeat(T[i]+(0.3)/len(MODEL)*(j-len(MODEL)/2),np.size(tend_var_reg_ave[(i,j)])),tend_var_reg_ave[(i,j)],color=colores[j],markersize=6,marker='o',alpha=0.5,lw=0)
        #Plot member-average trend
        ax.plot(T[i]+(0.3)/len(MODEL)*(j-len(MODEL)/2),np.nanmean(tend_var_reg_ave[(i,j)]),color=colores[j],marker='_', markersize=20,lw=0)
#Define ticks and labeles
ax.set_xticks(T)
ax.set_xticklabels(EXPERIMENT)
ax.set_xlabel('Experiment')
ax.set_ylabel('Trend (mm/day/summer/decade)')

#Add y=0 line
ax.axhline(y=0,linewidth=.5,linestyle='--',color='k')

#Define legend
lines = [Line2D([0], [0], color=c, marker='o',linestyle='None',) for c in colores]
plt.legend(lines,MODEL,loc='best')

#Set title
ax.set_title( REGION_NAME + ' ' + VAR + ' Trend ' + str(YEARMIN) + '-' + str(YEARMAX), fontsize=12)

#Improve space in figure
fig.tight_layout()
fig.subplots_adjust(bottom=0.03, right=0.96,top=0.97)

#Save figure
fig.savefig(FIG_PATH + '_trend_'+ REGION_NAME + '_' + VAR + '_' + str(YEARMIN) + '_' + str(YEARMAX)+ '.png', dpi=500, bbox_inches='tight',
             papertype='A4', orientation='landscape')