# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 16:08:14 2019

@author: akomarla
"""
import pandas as pd 
import numpy as np
import xlsxwriter

path_s = "C:/Users/akomarla/Documents/Tickets_Analysis_Jira/Sightings_NCT_Linkage/CSDP_CDRDP_sightings_sanitized.xlsx"
sightingsdata = pd.read_excel(path_s, index = False)
sightingsdata.rename(columns = {'Status':'NSGSE Status'}, inplace=True)
sightingsdata['Security'] = np.nan
sightingsdata['Status'] = np.nan
sightingsdata['Job Number'] = np.nan

path_k = "C:/Users/akomarla/Documents/Tickets_Analysis_Jira/Sightings_NCT_Linkage/CSDP_CDRDP_NCTs_sanitized.xlsx"
keydataorg = pd.read_excel(path_k, index = False)
keydata = keydataorg

final_data = pd.DataFrame(columns = sightingsdata.columns.values)
stored_data = pd.DataFrame(columns = sightingsdata.columns.values)

i = 0 
j = 0

for i in range(0,len(sightingsdata)) :
    
    sightingsdata['Linked Issues'] = sightingsdata['Linked Issues'].astype(str)
    Linkage = pd.Series((sightingsdata.iloc[i,11]).split(', '))
    NCT_Linkage_TF = Linkage.str.contains('NCT', regex = False)
    Index_NCT_True = pd.array(NCT_Linkage_TF[NCT_Linkage_TF].index)
    
    sightingsdata['Assign Team'] = sightingsdata['Assign Team'].astype(str)
    Assign_Team = pd.Series((sightingsdata.iloc[i,9]))
    Assign_Team_TF = Assign_Team.str.contains('Intel - FW Development', regex = False)
    
    sightingsdata['Labels'] = sightingsdata['Labels'].astype(str)
    Labels = pd.Series((sightingsdata.iloc[i,10]))
    Labels_TF = Labels.str.contains('SecureStorage', regex = False)
    
    Security = ((Assign_Team_TF == True) & (Labels_TF == True)).bool()
    
    if len(Index_NCT_True) != 0 :
        
        for j in range (0, len(Index_NCT_True)) :
            sightingsdata.iloc[i,11] = Linkage[Index_NCT_True[j]]
            keydata.set_index(keydata.columns[0], drop = False, inplace = True)
            Customer = keydata.loc[Linkage[Index_NCT_True[j]], 'Customer']
            Status = keydata.loc[Linkage[Index_NCT_True[j]], 'Status']
            keydata = keydataorg            
            sightingsdata.iloc[i,8] = Customer 
            sightingsdata.iloc[i,13] = Status
            
            Customer = pd.Series(Customer)
            
            if Security == True :
                sightingsdata.iloc[i,12] = 'Yes'                
            if Security == False :
                sightingsdata.iloc[i,12] = 'No'            
            if (Customer.str.contains('Dell EMC', regex = False)).bool() : 
                sightingsdata.iloc[i,14] = 1               
            if (Customer.str.contains('Hitachi|HPE 3Par|IBM', regex = True)).bool() : 
                sightingsdata.iloc[i,14] = 2            
            if (~(Customer.str.contains('Dell EMC|Hitachi|HPE 3Par|IBM', regex = True))).bool() : 
                sightingsdata.iloc[i,14] = 3
                
            final_data = final_data.append(sightingsdata.iloc[i,:], ignore_index = True)
            
            j = j + 1
   
    else :
        if Security == True :
            sightingsdata.iloc[i,12] = 'Yes'                
        if Security == False :
            sightingsdata.iloc[i,12] = 'No' 
        sightingsdata.iloc[i,14] = 4
        stored_data = stored_data.append(sightingsdata.iloc[i,:], ignore_index = True) 
        
i = i + 1 

final_data = final_data.astype(object).where(final_data.notnull(), None)
stored_data = stored_data.astype(object).where(stored_data.notnull(), None)

workbook = xlsxwriter.Workbook('C:/Users/akomarla/Documents/Tickets_Analysis_Jira/Sightings_NCT_Linkage/NSGSEs_MBEs_Sightings_Data_Table.xlsx')
NSGSEs_Linked_NCT = workbook.add_worksheet('NSGSEs_MBEs_Linked_NCT')
NSGSEs_Unlinked_NCT = workbook.add_worksheet('NSGSEs_MBEs_Unlinked_NCT')

formatdict = {'num_format' : 'yyyy-mm-dd hh:mm:ss'}
fmt = workbook.add_format(formatdict)

header_f = [{'header': h} for h in final_data.columns.tolist()]
header_s = [{'header': h} for h in stored_data.columns.tolist()]

NSGSEs_Linked_NCT.set_column('F:I', 15, fmt)
NSGSEs_Linked_NCT.add_table('A1:O232', {'data': final_data.values.tolist(), 'header_row': True, 'columns': header_f})

NSGSEs_Unlinked_NCT.set_column('F:I', 15, fmt)
NSGSEs_Unlinked_NCT.add_table('A1:O1527', {'data': stored_data.values.tolist(), 'header_row': True, 'columns': header_s})
workbook.close()

#with ExcelWriter('C:/Users/akomarla/Documents/Tickets_Analysis_Jira/Sightings_NCT_Linkage/Test_Sightings_NCT_Customer_Status.xlsx') as writer:
     #final_data.to_excel(writer)

#with ExcelWriter('C:/Users/akomarla/Documents/Tickets_Analysis_Jira/Sightings_NCT_Linkage/Test_Sightings_Stored_Data.xlsx') as writer:
     #stored_data.to_excel(writer)