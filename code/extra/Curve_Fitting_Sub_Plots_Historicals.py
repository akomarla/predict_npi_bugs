# -*- coding: utf-8 -*-
#Created on Fri Aug  9 07:54:45 2019
#@author: akomarla

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd 

# Inputs - 

# Username - akomarla, mroten, etc.
# Data - Path of sightings data (ensure formatting is correct in excel file)
# Data_markers - alpha, beta and PRQ markers (check formatting)
# Num_Programs_Completed - Number of programs in the Data variable that are completed. This script should only run on 
# historicals or completed programs 


Username = 'mroten'



# Loading data and initializing dataframes 

Data = (pd.read_excel(open('C:/Users/{}/Documents/Sightings_Curve_Fitting/all sightings static.xlsx'.format(Username), 'rb'), sheet_name='cum data groomed by PRQ')).set_index('WW Bucket')
Data_markers = (pd.read_excel(open('C:/Users/{}/Documents/Sightings_Curve_Fitting/all sightings static.xlsx'.format(Username), 'rb'), sheet_name='alpha beta PRQ markers'))
Parameters = pd.DataFrame(index = list(range(0,25)), columns = ['Program', 'a', 'b', 'c', 'd', 'Avg. Residuals'])



# Curve fitting and generating graphs

def sigmoid_fit(x, a, b, c, d) :
    return (a / (1 + np.exp(-c * (x - d)))) + b

fig_cum_sightings = plt.figure(figsize=(25,45), facecolor='w', edgecolor='k')
fig_cum_sightings.suptitle('Sigmoid Curve Fitting - Cum. Sightings by WW', fontsize = 20)

i = 0

### this is the # of columns in the all sightings static file for programs that are "done"
### you'll add completed prorams to the right side and those will become models

Num_Programs_Completed = 18

###


for i in range(0, Num_Programs_Completed) :
    
    y_data = np.array((Data.iloc[:,i][~Data.iloc[:,i].isna()]).reset_index(drop = True), dtype = 'float64')
    x_data = np.array(range(0,len(y_data)))

    popt, pcov = opt.curve_fit(sigmoid_fit, x_data, y_data, p0 = [max(y_data), min(y_data), 0.1, np.median(x_data)], method = 'dogbox', maxfev = 5000)
    perr = np.sqrt(np.diag(pcov))
    nstd = 2.0 
    popt_up = popt + nstd * perr
    popt_dw = popt - nstd * perr
    
    y_fit_m = sigmoid_fit(x_data, *popt)
    y_fit_up = sigmoid_fit(x_data, *popt_up)
    y_fit_dw = sigmoid_fit(x_data, *popt_dw)
    
    Parameters.iloc[i,0], Parameters.iloc[i,1], Parameters.iloc[i,2], Parameters.iloc[i,3], Parameters.iloc[i,4] = Data.iloc[:,i].name, *popt    
    
    alpha_WW = int(Data_markers.loc[1,Data.iloc[:,i].name])
    alpha_sightings = y_data[alpha_WW] 
    beta_WW = int(Data_markers.loc[2,Data.iloc[:,i].name])
    beta_sightings = y_data[beta_WW] 
    prq_WW = int(Data_markers.loc[3,Data.iloc[:,i].name])
    prq_sightings = y_data[prq_WW] 
    
    ax = fig_cum_sightings.add_subplot(7, 3, i + 1)
    points_org = ax.plot(x_data, y_data, '.', markersize = 3, markerfacecolor = 'blue', label = 'Original Data')
    line_fit = ax.plot(x_data, y_fit_m, 'r-', color = 'orange', label = 'Fitted Data')
    line_fit_up = ax.plot(x_data, y_fit_up, 'g--', color='gray', alpha = 0.4, label = '95% Confidence Interval')
    line_fit_dw = ax.plot(x_data, y_fit_dw, 'g--', color='gray', alpha = 0.4)
    ax.fill_between(x_data, y_fit_up, y_fit_dw, facecolor = 'gray', alpha = 0.15)
    points_markers_alpha = ax.plot(alpha_WW, alpha_sightings, '.', markersize = 15, markerfacecolor = 'green', markeredgecolor = 'none', label = 'Alpha')
    points_markers_beta = ax.plot(beta_WW, beta_sightings, '.', markersize = 15, markerfacecolor = 'purple',  markeredgecolor = 'none', label = 'Beta')
    points_markers_prq = ax.plot(prq_WW, prq_sightings, '.', markersize = 15, markerfacecolor = 'red',  markeredgecolor = 'none', label = 'PRQ')
    ax.set(title = Data.iloc[:,i].name, xlabel = 'WW', ylabel = 'Cum. Sightings')
    ax.legend(loc = 'best')
    ax.grid(linestyle=':')
    fig_cum_sightings.subplots_adjust(top = 0.95)
     
    residuals = y_data - y_fit_m
    Parameters.iloc[i,5] = np.mean(np.abs(residuals))
    
i = i + 1

Parameters = (Parameters.dropna(how = 'all')).set_index('Program')



# Saving excel data and figures 

fig_cum_sightings.savefig('C:/Users/{}/Documents/Sightings_Curve_Fitting/Historical_Analysis/historical_programs_sightings_curve_fit_plot.jpg'.format(Username))
with pd.ExcelWriter('C:/Users/{}/Documents/Sightings_Curve_Fitting/Historical_Analysis/Historical_Fit_Parameters.xlsx'.format(Username)) as writer:
    Parameters.to_excel(writer)