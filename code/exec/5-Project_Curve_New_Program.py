# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:53:17 2019
@author: akomarla
"""

# This script is to generate "projection" curves for forced LPL values. These curves are NOT predictions.
# Assume that you want to see what the curve would look like for beta to reach week 161 (from start of program)
# LPL_New = 161 - WW_Beginning_LP 
# LPL_New = 161 - 86 = 75 weeks 
# Note - ensure that this script is run after 'Prediction_New_Program.py' without clearning the console

# Inputs - 
# LPL_New_a/b = desired linear phase length
# Desired_WW_Tip_a/b = desired tip-off (slow-down in upper half) for program 
# For example, week 161 is WW 44 2019 for ADP, which is the Desired_WW_Tip_a


###############################################################################


# Change values to desired LPL and WW Tip

LPL_New_a = 75
LPL_New_b = 85

Desired_WW_Tip_a = "WW44'19"
Desired_WW_Tip_b = "WW02'20"


###############################################################################


def gen_projection_data(Inflection_Point_WW) : 
    y_data_for_projection = np.array(New_Program_Data.iloc[0:Inflection_Point_WW+1], dtype = 'float64')
    Projected_Data = np.flipud(invert_data(y_data_for_projection)) + y_data_for_projection[Inflection_Point_WW]
    Compl_Projection = np.concatenate([np.int_(y_data_for_projection), Projected_Data[1:len(Projected_Data)]])
    return Compl_Projection


###############################################################################
    

# Generating inflection point and projection data 

IP_New_a = gen_ip(WW_Beginning_LP,LPL_New_a)
IP_New_b = gen_ip(WW_Beginning_LP,LPL_New_b)

Compl_Projection_New_a = gen_projection_data(IP_New_a)
Compl_Projection_New_b = gen_projection_data(IP_New_b)


###############################################################################


# Plotting predicted data (5 curves) and projected data 

fig_projections = plt.figure(figsize=(10,7), facecolor='w', edgecolor='k')
plt.plot(range(0,len(Compl_Prediction_M)), Compl_Prediction_M, label = 'LPL = {} WW (Prediction)'.format(LPL_Approx), color = 'salmon')
plt.plot(range(0,len(Compl_Prediction_Up_33)), Compl_Prediction_Up_33, label = 'LPL UB (+33%) (Prediction)', color = 'gold')
plt.plot(range(0,len(Compl_Prediction_Dw_33)), Compl_Prediction_Dw_33, label = 'LPL LB (-33%) (Prediction)', color = 'goldenrod')
plt.plot(range(0,len(Compl_Prediction_Up_16)), Compl_Prediction_Up_16, label = 'LPL UB (+16.5%) (Prediction)', color = 'turquoise')
plt.plot(range(0,len(Compl_Prediction_Dw_16)), Compl_Prediction_Dw_16, label = 'LPL LB (-16.5%) (Prediction)', color = 'teal')
plt.plot(range(0,len(Compl_Projection_New_a)), Compl_Projection_New_a, linestyle='dashed', label = 'LPL for {} Tip (Projection)'.format(Desired_WW_Tip_a), color = 'black', alpha = 0.5)
plt.plot(range(0,len(Compl_Projection_New_b )), Compl_Projection_New_b , linestyle='dashed', label = 'LPL for {} Tip (Projection)'.format(Desired_WW_Tip_b), color = 'black', alpha = 0.5)
plt.plot(range(0,len(np.array(New_Program_Data, dtype = 'float64'))), np.array(New_Program_Data, dtype = 'float64'), label = 'True Data', color = 'orangered', linewidth = 2.5, alpha = 0.8)
plt.xlabel('WW')
plt.ylabel('Cum. Num. of Sightings')
plt.title(New_Program_Data.name)
plt.legend(loc='upper left')
plt.grid(linestyle = ':')
fig_projections.savefig('C:/Users/{}/Desktop/python stuff/Sightings_Prediction_Analysis/New_Program_Predictions/new_program_projections_plot_{}.jpg'.format(Username, New_Program_Data.name))
