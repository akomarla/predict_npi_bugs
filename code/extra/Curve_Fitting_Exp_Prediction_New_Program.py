# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:47:17 2019
@author: akomarla
"""

import numpy as np
import pandas as pd 
import scipy.optimize as opt
import matplotlib.pyplot as plt

# Inputs - 

# Username - akomarla, mroten, etc.
# Data - Path of sightings data (check formatting)
# Program_Location - Location of program that is of interest
# Program Size - Uncomment the size of the new program 

Username = 'mroten'



# Extracting program data

Data = (pd.read_excel(open('C:/Users/{}/Documents/Sightings_Curve_Fitting/all sightings static.xlsx'.format(Username), 'rb'), sheet_name='cum data groomed by PRQ'))
Program_Location = 19
New_Program_Data = Data.iloc[:,Program_Location][~Data.iloc[:,Program_Location].isna()].reset_index(drop = True)



# Approximate measure of program size (in terms of final max. cum. sightings)

#Program_Size = 'Small'
#Program_Size = 'Medium'
#Program_Size = 'Large'
#Program_Size = 'Extra Large'
Program_Size = 'x Extra Large'



# Fitting the curve 

def exp_fit(x, a, b, c, d) : 
    return (a * np.exp(b * (x - c))) + d 


Num_Points_Avlb = len(New_Program_Data)
Num_Points_Fit = 50

fig_exp_fitted = plt.figure(figsize=(20,20), facecolor='w', edgecolor='k')
fig_exp_fitted.suptitle('Exponential Curve Fitting - Cum. sightings by WW', fontsize = 20)
fig_exp_extended = plt.figure(figsize=(20,20), facecolor='w', edgecolor='k')
fig_exp_extended.suptitle('Exponential Curve Extended - Cum. sightings by WW', fontsize = 20)

i = 0
WW_Beginning_LP = np.array([])

while Num_Points_Fit < Num_Points_Avlb : 
    
    y_data_fit = np.array(New_Program_Data.iloc[0:Num_Points_Fit], dtype = 'float64')
    x_data_fit = np.array(range(0,Num_Points_Fit))

    popt, pcov = opt.curve_fit(exp_fit, x_data_fit, y_data_fit, maxfev = 5000)
    y_fitted = exp_fit(x_data_fit, *popt)
    
    ax_f = fig_exp_fitted.add_subplot(5, 3, i + 1)
    ax_f.plot(x_data_fit, y_data_fit, '.', markersize = 3, label = 'True Data')
    ax_f.plot(x_data_fit, y_fitted, '-', label = 'Fitted Data')
    ax_f.grid(linestyle=':')
    ax_f.legend(loc = 'best')
    fig_exp_fitted.tight_layout()
    fig_exp_fitted.subplots_adjust(top = 0.95)
    
    residuals_f = y_fitted - y_data_fit


    Num_Points_Extnd = Num_Points_Fit + 10
    
    y_data_extnd = np.array(New_Program_Data.iloc[0:Num_Points_Extnd], dtype = 'float64')
    x_data_extnd = np.array(range(0,Num_Points_Extnd))
    y_fitted_extnd = exp_fit(x_data_extnd, *popt)
    
    ax_e = fig_exp_extended.add_subplot(5, 3, i + 1)
    ax_e.plot(x_data_extnd, y_data_extnd, '.', markersize = 3, label = 'True Data')
    ax_e.plot(x_data_extnd, y_fitted_extnd, '-', label = 'Fitted Data')
    ax_e.grid(linestyle=':')
    ax_e.legend(loc = 'best')
    fig_exp_extended.tight_layout()
    fig_exp_extended.subplots_adjust(top = 0.95)

    residuals_e = y_fitted_extnd - y_data_extnd
    
    Residuals_Over_50 = abs(np.int_(residuals_e))[(abs(np.int_(residuals_e)) > 50)]
    Residuals_Over_100 = abs(np.int_(residuals_e))[(abs(np.int_(residuals_e)) > 100)]
    
    if ((Program_Size == 'Small') or (Program_Size == 'Medium')) & (len(Residuals_Over_50) > 3) :
        WW_Beginning_LP = np.where((abs(np.int_(residuals_e)) <= 100) & (abs(np.int_(residuals_e)) >= 50))[0][0]
        print('Linear phase reached at WW', WW_Beginning_LP)
        break    
    if ((Program_Size == 'Large') or (Program_Size == 'Extra Large') or (Program_Size == 'x Extra Large')) & (len(Residuals_Over_100) > 3) :
        WW_Beginning_LP = np.where((abs(np.int_(residuals_e)) <= 200) & (abs(np.int_(residuals_e)) >= 100))[0][0]
        print('Linear phase reached at WW', WW_Beginning_LP)
        break
    
    i = i + 1 
    
    Num_Points_Fit = Num_Points_Extnd
    
fig_exp_fitted.savefig('C:/Users/{}/Documents/Sightings_Curve_Fitting/New_Program_Predictions/new_program_exp_fitted_plot.jpg'.format(Username))
fig_exp_extended.savefig('C:/Users/{}/Documents/Sightings_Curve_Fitting/New_Program_Predictions/new_program_exp_extended_plot.jpg'.format(Username))

if WW_Beginning_LP.size == 0 :
    print('Linear phase not yet reached, data is still in exponential phase','Try again in 10+ weeks', 'Program will exit')
    exit()



# Curve Inversion & Residual Calculation

def invert_data(y) :
    new_y = max(y) - y
    return new_y

def gen_ip(WW_Beginning_LP, LPL) :
    Inflection_Point = np.int((WW_Beginning_LP) + np.int(np.median(range(0,LPL))))
    return Inflection_Point

def gen_prediction_data(Inflection_Point_WW) : 
    y_data_for_prediction = np.array(New_Program_Data.iloc[0:Inflection_Point_WW+1], dtype = 'float64')
    Predicted_Data = np.flipud(invert_data(y_data_for_prediction)) + y_data_for_prediction[Inflection_Point_WW]
    Compl_Prediction = np.concatenate([np.int_(y_data_for_prediction), Predicted_Data[1:len(Predicted_Data)]])
    return Compl_Prediction 

def ww_ending_lp(WW_Beginning_LP, LPL) : 
    WW_Ending_LP = LPL + WW_Beginning_LP 
    return WW_Ending_LP 

LPL_Approx = ()

if Program_Size == 'Small' : 
    LPL_Approx = 30   
if Program_Size == 'Medium' : 
    LPL_Approx = 40   
if Program_Size == 'Large' : 
    LPL_Approx = 50   
if Program_Size == 'Extra Large' : 
    LPL_Approx = 60 
elif Program_Size == 'x Extra Large' : 
    LPL_Approx = 70

LPL_33_Up, LPL_33_Dw, LPL_16_Up, LPL_16_Dw  = np.int(LPL_Approx + 0.33*LPL_Approx), np.int(LPL_Approx - 0.33*LPL_Approx), np.int(LPL_Approx + 0.165*LPL_Approx), np.int(LPL_Approx - 0.165*LPL_Approx)

Inflection_Point_WW_M, Inflection_Point_WW_Up_33, Inflection_Point_WW_Dw_33, Inflection_Point_WW_Up_16, Inflection_Point_WW_Dw_16  = gen_ip(WW_Beginning_LP,LPL_Approx), gen_ip(WW_Beginning_LP,LPL_33_Up), gen_ip(WW_Beginning_LP,LPL_33_Dw), gen_ip(WW_Beginning_LP,LPL_16_Up), gen_ip(WW_Beginning_LP,LPL_16_Dw)

WW_Ending_LP_M, WW_Ending_LP_33_Up, WW_Ending_LP_33_Dw, WW_Ending_LP_16_Up, WW_Ending_LP_16_Dw = ww_ending_lp(WW_Beginning_LP, LPL_Approx), ww_ending_lp(WW_Beginning_LP, LPL_33_Up), ww_ending_lp(WW_Beginning_LP, LPL_33_Dw), ww_ending_lp(WW_Beginning_LP, LPL_16_Up), ww_ending_lp(WW_Beginning_LP, LPL_16_Dw) 


if Inflection_Point_WW_Up_33 <= Num_Points_Avlb : 
    Compl_Prediction_M, Compl_Prediction_Up_33, Compl_Prediction_Dw_33, Compl_Prediction_Up_16, Compl_Prediction_Dw_16 = gen_prediction_data(Inflection_Point_WW_M), gen_prediction_data(Inflection_Point_WW_Up_33), gen_prediction_data(Inflection_Point_WW_Dw_33), gen_prediction_data(Inflection_Point_WW_Up_16), gen_prediction_data(Inflection_Point_WW_Dw_16) 
elif Inflection_Point_WW_Up_33 > Num_Points_Avlb : 
    print('Data does not contain median of linear phase')
    print('Wait for', (Inflection_Point_WW_Up_33 - Num_Points_Avlb), 'WWs to reach median of linear phase')     
    exit()
    


# Adding values to dictionary & finding data set with least residual value 

Prediction_M, Prediction_33_Up, Prediction_33_Dw, Prediction_16_Up, Prediction_16_Dw = {}, {}, {}, {}, {} 
Prediction_M = {'Name': 'Prediction with M LPL', 'WW Beginning LP': WW_Beginning_LP, 'Linear Phase Length': LPL_Approx, 'Inflection Point WW': Inflection_Point_WW_M, 'WW Ending LP': WW_Ending_LP_M, 'Predicted Values': Compl_Prediction_M}
Prediction_33_Up = {'Name': 'Prediction with +33% LPL', 'WW Beginning LP': WW_Beginning_LP, 'Linear Phase Length': LPL_33_Up, 'Inflection Point WW': Inflection_Point_WW_Up_33, 'WW Ending LP': WW_Ending_LP_33_Up, 'Predicted Values': Compl_Prediction_Up_33}
Prediction_33_Dw = {'Name': 'Prediction with -33% LPL', 'WW Beginning LP': WW_Beginning_LP, 'Linear Phase Length': LPL_33_Dw, 'Inflection Point WW': Inflection_Point_WW_Dw_33, 'WW Ending LP': WW_Ending_LP_33_Dw, 'Predicted Values': Compl_Prediction_Dw_33}
Prediction_16_Up = {'Name': 'Prediction with +16% LPL', 'WW Beginning LP': WW_Beginning_LP, 'Linear Phase Length': LPL_16_Up, 'Inflection Point WW': Inflection_Point_WW_Up_16, 'WW Ending LP': WW_Ending_LP_16_Up,'Predicted Values': Compl_Prediction_Up_16}
Prediction_16_Dw = {'Name': 'Prediction with -16% LPL', 'WW Beginning LP': WW_Beginning_LP, 'Linear Phase Length': LPL_16_Dw, 'Inflection Point WW': Inflection_Point_WW_Dw_16, 'WW Ending LP': WW_Ending_LP_16_Dw, 'Predicted Values': Compl_Prediction_Dw_16}
Predictions_List = [Prediction_M, Prediction_33_Up, Prediction_33_Dw, Prediction_16_Up, Prediction_16_Dw]

writer = pd.ExcelWriter('C:/Users/{}/Documents/Sightings_Curve_Fitting/New_Program_Predictions/New_Program_Prediction_{}.xlsx'.format(Username, New_Program_Data.name))

u = 0 
for u in range(0,len(Predictions_List)) : 
    Selected_Prediction_Data = Predictions_List[u]
    pd.DataFrame(Selected_Prediction_Data['Predicted Values'], columns = ['Predicted Sightings Data']).to_excel(writer, sheet_name = '{}'.format(Selected_Prediction_Data['Name']))
    pd.DataFrame([Selected_Prediction_Data['Name'], Selected_Prediction_Data['WW Beginning LP'], Selected_Prediction_Data['Linear Phase Length'], Selected_Prediction_Data['Inflection Point WW'], Selected_Prediction_Data['WW Ending LP']],\
                 index = ['Name', 'WW Beginning LP', 'Linear Phase Length', 'Inflection Point WW', 'WW Ending LP'], columns = ['Details']).to_excel(writer, sheet_name = 'Info - {}'.format(Selected_Prediction_Data['Name']))
writer.save()
u = u + 1 



# Plotting data 
    
fig_predictions = plt.figure(figsize=(10,7), facecolor='w', edgecolor='k')
plt.plot(range(0,len(Compl_Prediction_M)), Compl_Prediction_M, label = 'LPL = {} WW'.format(LPL_Approx), color = 'salmon')
plt.plot(range(0,len(Compl_Prediction_Up_33)), Compl_Prediction_Up_33, label = 'LPL UB (+33%)', color = 'gold')
plt.plot(range(0,len(Compl_Prediction_Dw_33)), Compl_Prediction_Dw_33, label = 'LPL LB (-33%)', color = 'goldenrod')
plt.plot(range(0,len(Compl_Prediction_Up_16)), Compl_Prediction_Up_16, label = 'LPL UB (+16.5%)', color = 'turquoise')
plt.plot(range(0,len(Compl_Prediction_Dw_16)), Compl_Prediction_Dw_16, label = 'LPL LB (-16.5%)', color = 'teal')
plt.plot(range(0,len(np.array(New_Program_Data, dtype = 'float64'))), np.array(New_Program_Data, dtype = 'float64'), label = 'True Data', color = 'orangered', linewidth = 2.5, alpha = 0.8)
plt.xlabel('WW')
plt.ylabel('Cum. Num. of Sightings')
plt.title(New_Program_Data.name)
plt.legend(loc='upper left')
plt.grid(linestyle = ':')
fig_predictions.savefig('C:/Users/{}/Documents/Sightings_Curve_Fitting/New_Program_Predictions/new_program_predictions_plot.jpg'.format(Username))
