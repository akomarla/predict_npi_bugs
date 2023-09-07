# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:23:51 2019
@author: akomarla
"""
import sympy as sym
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Inputs - 

# Username - akomarla, mroten, etc.
# Data - Path of sightings data (check formatting)
# Parameters - Comes from previous script (no separate variable is loaded)

Username = 'mroten'



# Defining derivatives 

def sigmoid_fit(x, a, b, c, d) :
    return (a / (1 + sym.exp(-c * (x - d)))) + b

x, a, b, c, d = sym.symbols('x a b c d')
First_Derivative = sym.diff(sigmoid_fit(x, a, b, c, d), x)
Second_Derivative = sym.diff(First_Derivative, x)



# Find derivative for a specific program and WW (uses lookup method)
    
Program_Name = 'CD'
WW_Lookup = 163.373
Program_Lookup = Parameters.loc[Program_Name]

First_Derivative_Eval = First_Derivative.evalf(subs = ({a:Program_Lookup['a'], b:Program_Lookup['b'], c:Program_Lookup['c'], d:Program_Lookup['d'], x:WW_Lookup}))
Second_Derivative_Eval = Second_Derivative.evalf(subs = ({a:Program_Lookup['a'], b:Program_Lookup['b'], c:Program_Lookup['c'], d:Program_Lookup['d'], x:WW_Lookup}))



# Find first & second derivatives for all programs and all WWs

fig_sec_derv = plt.figure(figsize=(25,45), facecolor='w', edgecolor='k')
fig_fir_derv = plt.figure(figsize=(25,45), facecolor='w', edgecolor='k')


writer = pd.ExcelWriter('C:/Users/{}/Documents/Sightings_Curve_Fitting/Historical_Analysis/Historical_Derv_Analysis.xlsx'.format(Username))


j = 0 
k = 0 
p = 0 
Derivative_Analysis_Dict = {}

Num_Programs_Completed = 18
for j in range (0, Num_Programs_Completed ) :
    Program_Lookup = Parameters.iloc[j,:]
    y_data = np.array((Data.iloc[:,j][~Data.iloc[:,j].isna()]).reset_index(drop = True), dtype = 'float64')
    x_data = np.array(range(0,len(y_data)))
    Derivative_Analysis_df = pd.DataFrame(index = range(0,len(y_data)), columns = ['WW', 'First Derivative Eval', 'Second Derivative Eval', 'Linear Phase Length', 'Max Cum. Sightings', 'WW Linear Phase Beginning', 'WW Linear Phase Ending'])
    
    for k in range(0,len(y_data)) :
        Derivative_Analysis_df.iloc[k,0] = k + 1 
        Derivative_Analysis_df.iloc[k,1] = (First_Derivative.evalf(subs = ({a:Program_Lookup['a'], b:Program_Lookup['b'], c:Program_Lookup['c'], d:Program_Lookup['d'], x:k})))
        Derivative_Analysis_df.iloc[k,2] = (Second_Derivative.evalf(subs = ({a:Program_Lookup['a'], b:Program_Lookup['b'], c:Program_Lookup['c'], d:Program_Lookup['d'], x:k})))
        k = k + 1 
    
    Derivative_Analysis_df.iloc[0,3] = (Derivative_Analysis_df[Derivative_Analysis_df.iloc[:,2] == min(Derivative_Analysis_df.iloc[:,2])].iloc[0,0]) - (Derivative_Analysis_df[Derivative_Analysis_df.iloc[:,2] == max(Derivative_Analysis_df.iloc[:,2])].iloc[0,0])
    Derivative_Analysis_df.iloc[0,4] = max(y_data)
    Derivative_Analysis_df.iloc[0,5] = (Derivative_Analysis_df[Derivative_Analysis_df.iloc[:,2] == max(Derivative_Analysis_df.iloc[:,2])].iloc[0,0])
    Derivative_Analysis_df.iloc[0,6] = (Derivative_Analysis_df[Derivative_Analysis_df.iloc[:,2] == min(Derivative_Analysis_df.iloc[:,2])].iloc[0,0])
    
    Derivative_Analysis_Dict['Derivative_Analysis_{}'.format(Program_Lookup.name)] = Derivative_Analysis_df  
    Derivative_Analysis_df.to_excel(writer, sheet_name = 'Derv Analysis - {}'.format(Program_Lookup.name))
    
    
    alpha_WW = int(Data_markers.loc[1,Data.iloc[:,j].name])
    alpha_fd, alpha_sd = Derivative_Analysis_df.iloc[:,1][alpha_WW], Derivative_Analysis_df.iloc[:,2][alpha_WW] 
    beta_WW = int(Data_markers.loc[2,Data.iloc[:,j].name])
    beta_fd, beta_sd = Derivative_Analysis_df.iloc[:,1][beta_WW], Derivative_Analysis_df.iloc[:,2][beta_WW]  
    prq_WW = int(Data_markers.loc[3,Data.iloc[:,j].name])
    prq_fd, prq_sd = Derivative_Analysis_df.iloc[:,1][prq_WW], Derivative_Analysis_df.iloc[:,2][prq_WW]
    
    
    ax_sd = fig_sec_derv.add_subplot(7, 3 , j + 1)
    ax_sd.plot(x_data, Derivative_Analysis_df.iloc[:,2], '.', markersize = 5)
    ax_sd.plot(alpha_WW, alpha_sd, '.', markersize = 17, markerfacecolor = 'green', markeredgecolor = 'none', label = 'Alpha')
    ax_sd.plot(beta_WW, beta_sd, '.', markersize = 17, markerfacecolor = 'purple', markeredgecolor = 'none', label = 'Beta')
    ax_sd.plot(prq_WW, prq_sd, '.', markersize = 17, markerfacecolor = 'red', markeredgecolor = 'none', label = 'PRQ')
    ax_sd.set(title = Program_Lookup.name, xlabel = 'WW', ylabel = 'Second Derivative')
    ax_sd.legend(loc = 'best')
    ax_sd.grid(linestyle=':')
    fig_sec_derv.suptitle('Sigmoid Curve Fitting & Second Derivative - Cum. sightings by WW', fontsize = 20)
    fig_sec_derv.subplots_adjust(top = 0.95)
    
    ax_fd = fig_fir_derv.add_subplot(7, 3 , j + 1)
    ax_fd.plot(x_data, Derivative_Analysis_df.iloc[:,1], '.', markersize = 5)
    ax_fd.plot(alpha_WW, alpha_fd, '.', markersize = 17, markerfacecolor = 'green', markeredgecolor = 'none', label = 'Alpha')
    ax_fd.plot(beta_WW, beta_fd, '.', markersize = 17, markerfacecolor = 'purple', markeredgecolor = 'none', label = 'Beta')
    ax_fd.plot(prq_WW, prq_fd, '.', markersize = 17, markerfacecolor = 'red', markeredgecolor = 'none', label = 'PRQ')
    ax_fd.set(title = Program_Lookup.name, xlabel = 'WW', ylabel = 'First Derivative')
    ax_fd.legend(loc = 'best')
    ax_fd.grid(linestyle=':')
    fig_fir_derv.suptitle('Sigmoid Curve Fitting & First Derivative - Cum. sightings by WW', fontsize = 20)
    fig_fir_derv.subplots_adjust(top = 0.95)
 
writer.save()
j = j + 1



# Linear phase analysis using second derivative 

Linear_Phase_Lengths_df = pd.DataFrame(index = range(0,len(Derivative_Analysis_Dict.keys())), columns = ['Program Name', 'Linear Phase Length', 'Max Cum. Sightings', 'WW Linear Phase Beginning', 'WW Linear Phase Ending'])

for p in range(0, len(Derivative_Analysis_Dict.keys())) : 
    key = list(Derivative_Analysis_Dict.keys())[p]
    Linear_Phase_Lengths_df.iloc[p,0] = key
    Linear_Phase_Lengths_df.iloc[p,1] = ((Derivative_Analysis_Dict[key])['Linear Phase Length'][0])
    Linear_Phase_Lengths_df.iloc[p,2] = ((Derivative_Analysis_Dict[key])['Max Cum. Sightings'][0])
    Linear_Phase_Lengths_df.iloc[p,3] = ((Derivative_Analysis_Dict[key])['WW Linear Phase Beginning'][0])
    Linear_Phase_Lengths_df.iloc[p,4] = ((Derivative_Analysis_Dict[key])['WW Linear Phase Ending'][0])

fig_sec_derv.savefig('C:/Users/{}/Documents/Sightings_Curve_Fitting/Historical_Analysis/historical_second_derivative_plot.jpg'.format(Username))
fig_fir_derv.savefig('C:/Users/{}/Documents/Sightings_Curve_Fitting/Historical_Analysis/historical_first_derivative_plot.jpg'.format(Username))   
with pd.ExcelWriter('C:/Users/{}/Documents/Sightings_Curve_Fitting/Historical_Analysis/Historical_Linear_Phase_Lengths.xlsx'.format(Username)) as writer:
    Linear_Phase_Lengths_df.to_excel(writer) 