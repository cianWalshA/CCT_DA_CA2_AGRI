# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 13:27:50 2023

@author: cianw
"""


"""
*********************************************************************************************************************************************************************
SECTION 0: Declare Libraries
*********************************************************************************************************************************************************************
"""

import glob
import pandas as pd
import numpy as np
from pathlib import Path
pd.options.display.max_rows = 300
pd.options.display.max_columns = 100



def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
            " columns that have missing values.")
    return mis_val_table_ren_columns

"""
*********************************************************************************************************************************************************************
SECTION 1: Dictionary Codes for use if needs be and PATHWAYS
*********************************************************************************************************************************************************************
"""
cropsCodes = pd.read_csv("C:/Users/cianw/Documents/dataAnalytics/CA2/Data/Eurostat/Code Dictionary/crops.dic", sep='\t',names=['crops', 'crop_name'], header = None)
strucproCodes = pd.read_csv("C:/Users/cianw/Documents/dataAnalytics/CA2/Data/Eurostat/Code Dictionary/strucpro.dic",sep='\t',names=['code', 'units'], header = None)
unitCodes = pd.read_csv("C:/Users/cianw/Documents/dataAnalytics/CA2/Data/Eurostat/Code Dictionary/unit.dic",sep='\t',names=['code', 'units'], header = None)
fordCodes = pd.read_csv("C:/Users/cianw/Documents/dataAnalytics/CA2/Data/Eurostat/Code Dictionary/ford.dic",sep='\t',names=['ford', 'units'], header = None)

#Organic Production Data
stdProduction_path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Agricultural Production\Crops\apro_cpnh1_linear.csv")
orgTonne_path= Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Agricultural Production\Crops\org_croppro_linear.csv")
orgArea_path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Agricultural Production\Crops\org_cropar_linear.csv")

#Harmonised Risk Index 1 Data
hriPath = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Agricultural Production\Crops\Pesiticide Use Risk Indicator\aei_hri_linear.csv")
#Organic Processors Data
orgProcessors_path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Agricultural Production\Crops\Organic Processors\org_cpreact_linear.csv")
#Organic Area Utilisation Data
orgAreaUtil_path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Agricultural Production\Crops\Organic Area\sdg_02_40_linear.csv")
#N and P Fertilizer Data
fertUse_path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Agricultural Production\Crops\Fertlizer Use\aei_fm_usefert_linear.csv")
#Waste Generation Data
wasteGeneration_path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Resource Usage\cei_pc034_linear.csv")
#National Productivity Data
productivityIndex_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Resource Usage\cei_pc030_linear.csv")
#Country Gini Data
countryGini_path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Quality of Life\tessi190_linear.csv")
#Employment Rate Data
employmentRate_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Economics\tesem010_linear.csv")
#Median/Mean Income Data
income_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Economics\ilc_di03_linear.csv")
#Biodiversity of Birds Data
birdBiodiversity_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Biodiversity Index\env_bio2_linear.csv")
#Pesticide Sales
pestSales_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Agricultural Production\Crops\Pesticide Sales\aei_fm_salpest09_linear.csv")
#Pesticide Use
pestUse_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Agricultural Production\Crops\aei_pestuse_linear.csv")
#Farm Structure
farmStructure_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Farm Structure\ef_lac_main_linear.csv");
#Higher Education Spending
higherEdu_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Economics\rd_e_gerdtot_linear.csv");
#R&D Spending
research_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Economics\rd_e_gerdsc_linear.csv");


"""
*********************************************************************************************************************************************************************
SECTION 2: Basic Data import and formatting - TARGET
*********************************************************************************************************************************************************************
"""
#Crop Production
stdProduction = pd.read_csv(stdProduction_path)
stdProduction = stdProduction[stdProduction['crops'].str.contains("0000")]
stdProduction = stdProduction[~stdProduction['crops'].str.contains("_S0000")]
stdProduction = stdProduction[stdProduction['strucpro'].isin(['PR', 'AR']) ]
stdProduction  = stdProduction[~stdProduction['OBS_FLAG'].isin(['c','n'])]
stdProduction = stdProduction.drop(stdProduction[stdProduction['OBS_VALUE'] == 0].index)
stdProduction = stdProduction.pivot(index=['crops', 'geo', 'TIME_PERIOD'], columns='strucpro', values='OBS_VALUE').reset_index() 
stdProduction = stdProduction.dropna(subset = ['PR', 'AR'])
stdProduction['AR'] = stdProduction['AR']*1000
stdProduction['PR'] = stdProduction['PR']*1000

#Organic Crop Production Import
orgArea_all = pd.read_csv(orgArea_path)
orgArea_total = orgArea_all[(orgArea_all['agprdmet'] == 'TOTAL') & (orgArea_all['unit']=='HA' ) & orgArea_all['crops'].str.contains("0000")]
orgArea_total = orgArea_total.rename(columns={"OBS_VALUE":"area_HA"})
orgArea_total = orgArea_total[~orgArea_total['OBS_FLAG'].isin(['c','n'])]
orgArea_total = orgArea_total.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'unit', 'agprdmet', 'OBS_FLAG'])
orgArea_total = orgArea_total.dropna(subset=['area_HA']) #Removes last NAN value in SET
orgArea_total = orgArea_total[(orgArea_total['area_HA'] != 0)] #Removes 0 area values in SET which cause inf
missing_values_table(orgArea_total)

orgTonne = pd.read_csv(orgTonne_path)
orgTonne = orgTonne[orgTonne['crops'].str.contains("0000")]
orgTonne = orgTonne.rename(columns={"OBS_VALUE":"tonnes"})
orgTonne.info()
orgTonne = orgTonne[~orgTonne['OBS_FLAG'].isin(['c','n'])]
orgTonne = orgTonne.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'unit', 'OBS_FLAG'])
missing_values_table(orgTonne)

orgProduction = pd.merge(orgArea_total, orgTonne, on=['crops', 'geo', 'TIME_PERIOD'], how='inner', suffixes=('_A','_T'))

cropProd = pd.merge( stdProduction, orgProduction, on=['crops', 'geo', 'TIME_PERIOD'], how='inner', suffixes=('_std','_org'))
cropProd = pd.merge( cropProd, cropsCodes, on=['crops'], how='left')
cropProd['geo'] = cropProd['geo'].astype('str') 

cropProd['yieldOrg'] = cropProd['tonnes']/cropProd['area_HA']
cropProd['yieldStd'] = cropProd['PR']/cropProd['AR']
cropProd['yieldRatio'] = cropProd['yieldOrg']/cropProd['yieldStd']


cropProd = cropProd[cropProd['yieldRatio']<= 5] #Three rows were observed to have yield ratios of 20+, which is very out of sync with the rest, thus were removed.


"""
*********************************************************************************************************************************************************************
SECTION 3: Basic Data import and formatting - FEATURES
*********************************************************************************************************************************************************************
"""




"""
*********************************************************************************************************************************************************************
SECTION 4: Creation of Reference Dataset
*********************************************************************************************************************************************************************
"""


	PR_HU_EU
29	4817870.0
	AR
29	811790.0


"""
*********************************************************************************************************************************************************************
SECTION 5: Post Join processing - Analysis for nulls, spurious data etc.
*********************************************************************************************************************************************************************
"""




"""
*********************************************************************************************************************************************************************
SECTION 3: Basic Data import and formatting - FEATURES
*********************************************************************************************************************************************************************
"""







