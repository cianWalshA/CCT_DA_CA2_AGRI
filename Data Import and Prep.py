#!/usr/bin/env python
# coding: utf-8

# # Section 0 Defining modules/libraries/functions



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



cropsCodes = pd.read_csv("C:/Users/cianw/Documents/dataAnalytics/CA2/Data/Eurostat/Code Dictionary/crops.dic", sep='\t',names=['crops', 'crop_name'], header = None)
strucproCodes = pd.read_csv("C:/Users/cianw/Documents/dataAnalytics/CA2/Data/Eurostat/Code Dictionary/strucpro.dic",sep='\t',names=['code', 'units'], header = None)
unitCodes = pd.read_csv("C:/Users/cianw/Documents/dataAnalytics/CA2/Data/Eurostat/Code Dictionary/unit.dic",sep='\t',names=['code', 'units'], header = None)
fordCodes = pd.read_csv("C:/Users/cianw/Documents/dataAnalytics/CA2/Data/Eurostat/Code Dictionary/ford.dic",sep='\t',names=['ford', 'units'], header = None)


"""
Setting Paths
"""

#Organic Production Data
stdProduction_path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Agricultural Production\Crops\apro_cpsh1_linear.csv")
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
Inputting and formatting of datasets
"""

#Harmonised Risk Index 1 Data
hriPesticide = pd.read_csv(hriPath)
hriPesticide = hriPesticide[hriPesticide['subst_cat'].isin(['HRI1'])]
hriPesticide = hriPesticide[~hriPesticide['geo'].isin(['EU', 'EU27_2020', 'EU28', 'UK'])]
hriPesticide = hriPesticide.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'unit', 'OBS_FLAG'])
hriPesticide = hriPesticide.rename(columns={"OBS_VALUE":"harmRiskInd"})
#print(hriPesticide.groupby('TIME_PERIOD').geo.nunique())
#hriPesticide.shape
#hriPesticide.info()
hriPesticide.describe()
#hriPesticide.head(25)


#Organic Processors
orgProcessors = pd.read_csv(orgProcessors_path)
orgProcessors  = orgProcessors[orgProcessors['nace_r2'].isin(['C103','C101', 'C102', 'C104', 'C105', 'C109' 'C106'])]
orgProcessors  = orgProcessors[orgProcessors['unit'].isin(['NR'])]
orgProcessors = orgProcessors.drop(columns = ['DATAFLOW', 'LAST UPDATE','freq', 'unit', 'OBS_FLAG'])
orgProcessors = orgProcessors.pivot(index=['geo', 'TIME_PERIOD'], columns='nace_r2', values='OBS_VALUE').reset_index() 
orgProcessors = orgProcessors.rename(columns={"OBS_VALUE":"numOrganicProcessors"})
#print(orgProcessors.groupby('TIME_PERIOD').geo.size())
#print(orgProcessors.groupby('TIME_PERIOD').geo.nunique())
#print(orgProcessors.groupby(['TIME_PERIOD', 'geo']).geo.size())
orgProcessors.shape
#orgProcessors.info()
orgProcessors.head()


#Organic Area Utilisation Data
orgAreaUtil = pd.read_csv(orgAreaUtil_path)
orgAreaUtil = orgAreaUtil.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'crops','freq', 'agprdmet', 'unit', 'OBS_FLAG'])
orgAreaUtil = orgAreaUtil.rename(columns={"OBS_VALUE":"areaUsedForOrganic_PCT"})
#orgAreaUtil.shape
#orgAreaUtil.info()
#orgAreaUtil.head(10)

# Can apply an interesting step here by using left joins to exclude unwanted data later.
# 
# The dataset imported and pivoted/transposed below contains more than just the Countries required, it contains subdivisions indicated by numbers beside the country name. Instead of manually writing code that excludes these cases, this set will be left joined later, which will exclude the data automatically.

#N and P Fertilizer Data
fertUse = pd.read_csv(fertUse_path)
fertUse = fertUse.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'unit', 'OBS_FLAG'])
fertUse = fertUse.pivot(index=['geo', 'TIME_PERIOD'], columns='nutrient', values='OBS_VALUE').reset_index() 
fertUse = fertUse.rename(columns={"N":"N_use_tonne",
                                 "P":"P_use_tonne"})
#print(fertUseGroup3.groupby('TIME_PERIOD').geo.nunique())

# Below I am imputing the missing odd years by averaging the two around it. By treating the numbers as a trend.

#Waste Generation Data
wasteGeneration = pd.read_csv(wasteGeneration_path)
wasteGeneration = wasteGeneration.drop(columns = ['DATAFLOW', 'LAST UPDATE','nace_r2', 'unit','freq', 'OBS_FLAG', 'hazard', 'waste'])
wasteGeneration["waste_unit"] = "KG per Capita"
wasteGeneration = wasteGeneration.rename(columns={"OBS_VALUE":"waste"})

wasteGeneration_lag = wasteGeneration
wasteGeneration_lag['TIME_LAG'] = wasteGeneration_lag.groupby(['geo'])['TIME_PERIOD'].shift(1)
wasteGeneration_lag['waste_lag'] = wasteGeneration_lag.groupby(['geo'])['waste'].shift(1)

wasteGeneration_lag['waste_temp'] = ((wasteGeneration_lag['waste'] + wasteGeneration_lag['waste_lag'])/2)
wasteGeneration_lag['TIME_PERIOD_temp'] = ((wasteGeneration_lag['TIME_PERIOD'] + wasteGeneration_lag['TIME_LAG'])/2)
wasteGeneration_lag = wasteGeneration_lag.drop(columns = ['TIME_PERIOD', 'TIME_LAG','waste', 'waste_lag'])
wasteGeneration_lag = wasteGeneration_lag.dropna()
wasteGeneration_lag = wasteGeneration_lag.rename(columns={"waste_temp":"waste",
                                                           "TIME_PERIOD_temp":"TIME_PERIOD"})
wasteGeneration_lag['TIME_PERIOD'] = wasteGeneration_lag['TIME_PERIOD'].astype('int')
wasteGeneration = pd.concat([wasteGeneration, wasteGeneration_lag], join='inner', ignore_index=True)


#National Productivity Data
productivityIndex = pd.read_csv(productivityIndex_Path)
productivityIndex  = productivityIndex[productivityIndex['unit'].isin(['PPS_KG'])]
productivityIndex = productivityIndex.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'unit','freq', 'OBS_FLAG'])
productivityIndex["productivity_unit"] = "Purchase Power Standard Per KG"
productivityIndex = productivityIndex.rename(columns={"OBS_VALUE":"productivity"})
#productivityIndex.shape
#productivityIndex.info()



#GINI index data
countryGini = pd.read_csv(countryGini_path)

countryGini = countryGini.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'indic_il', 'OBS_FLAG'])
countryGini = countryGini.rename(columns={"OBS_VALUE":"gini"})


#Employment Rate Data
emplyomentRate = pd.read_csv(employmentRate_Path)
emplyomentRate = emplyomentRate.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'OBS_FLAG', 'age', 'unit', 'indic_em'])
emplyomentRate = emplyomentRate.pivot(index=['geo', 'TIME_PERIOD'], columns='sex', values='OBS_VALUE').reset_index() 
emplyomentRate = emplyomentRate.rename(columns={"T":"emplyomentRate_T",
                                               "M":"emplyomentRate_M",
                                               "F":"emplyomentRate_F"})
#emplyomentRate.describe()

#Median/Mean Income Data
income = pd.read_csv(income_Path)
income  = income[income['unit'].isin(['EUR'])]
income  = income[income['age'].isin(['TOTAL'])]
income = income.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'OBS_FLAG', 'age', 'unit'])
income = income.pivot(index=['geo', 'TIME_PERIOD'], columns=['indic_il', 'sex'], values='OBS_VALUE').reset_index() 
income.columns = [''.join(col) for col in income.columns.values]
#income.info()




#Bird Biodiversity
birdBiodiversity = pd.read_csv(birdBiodiversity_Path)
birdBiodiversity = birdBiodiversity.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'OBS_FLAG', 'unit'])
birdBiodiversity = birdBiodiversity.rename(columns={"OBS_VALUE":"birdBiodiversityIndex"})
#birdBiodiversity.head()

# Pest Use
pestUse= pd.read_csv(pestUse_Path)
#pestUse = pestUse.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'OBS_FLAG'])
#pestSales = pestSales[(pestSales['TIME_PERIOD'].isin([2016,2018]))]

pestUse['TIME_PERIOD'] = pestUse['TIME_PERIOD'].apply(lambda x: str(x)+ '_y')
pestUse = pestUse[(pestUse['unit'].isin(['KG']))]
pestUse = pestUse[(pestUse['pesticid'].str.contains("_"))]
pestUse  = pestUse[~pestUse['OBS_FLAG'].isin(['c'])]
pestUse['OBS_VALUE'] = np.where(pestUse.OBS_FLAG == 'n', 0, pestUse.OBS_VALUE)
#pestSales = pestSales[(pestSales['pesticid'].str.isdigit()==True)]


pestUse = pestUse.pivot(index=['pesticid','crops'], columns =['geo', 'TIME_PERIOD'], values='OBS_VALUE').reset_index() 
pestUse.columns = ['_'.join(col) for col in pestUse.columns.values]

farmStructure = pd.read_csv(farmStructure_Path)
farmStructure = farmStructure[(farmStructure['arable'].isin(['TOTAL']))]
farmStructure = farmStructure[~(farmStructure['crops'].isin(['ARA']))]
farmStructure = farmStructure[(farmStructure['so_eur'].isin(['TOTAL']))]
#pestUse = pestUse[(pestUse['unit'].isin(['KG']))]
#pestUse = pestUse[(pestUse['unit'].isin(['KG']))]
farmStructure = farmStructure.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'OBS_FLAG'])
farmStructure = farmStructure.rename(columns={"OBS_VALUE":"gini"})

farmStructure.head()


# Pest Sales for Stats Analysis
pestSales= pd.read_csv(pestSales_Path)
pestSales['mainFert'] = pestSales['pesticid'].str.contains(r'[0-9]')
pestSales = pestSales.dropna(subset=['OBS_VALUE'])
pestSales = pestSales.drop(pestSales[pestSales['mainFert'] == True].index)
pestSales = pestSales.drop(pestSales[pestSales['pesticid'] == 'TOTAL'].index)
pestSales = pestSales.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'OBS_FLAG', 'unit', 'mainFert'])

pestSales = pestSales.sort_values('OBS_VALUE').drop_duplicates(['geo', 'TIME_PERIOD'], keep='last')
pestSales = pestSales.rename(columns={"OBS_VALUE":"pest_KG",
                                     "pesticid" : "mostFrequentPest"})
pestSales.shape
#pestSales.head(300)


#pestSales = pestSales.pivot(index=['pesticid'], columns =['TIME_PERIOD', 'geo'], values='OBS_VALUE').reset_index() 
#pestSales.columns = ['_'.join(col) for col in pestSales.columns.values]

#pestSales.head(30)




orgTonne = pd.read_csv(orgTonne_path)
orgTonne = orgTonne[orgTonne['crops'].str.contains("0000")]
orgTonne = orgTonne.drop(orgTonne[orgTonne['OBS_VALUE'] == 0].index)
orgTonne = orgTonne.rename(columns={"OBS_VALUE":"orgTonnes",
                                   "crops":"mostGrownOrganic"})
#orgTonne.info()
orgTonne = orgTonne[~orgTonne['OBS_FLAG'].isin(['c','n'])]
orgTonne = orgTonne.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'unit', 'OBS_FLAG'])
orgTonne = orgTonne.sort_values('orgTonnes').drop_duplicates(['geo', 'TIME_PERIOD'], keep='last')

#print(orgTonne.head(10))
#missing_values_table(orgTonne)



stdProduction_lin = pd.read_csv(stdProduction_path)
stdProduction_lin = stdProduction_lin[stdProduction_lin['crops'].str.contains("0000")]
stdProduction_lin = stdProduction_lin[stdProduction_lin['strucpro'].isin(['PR_HU_EU']) ]
stdProduction_lin  = stdProduction_lin[~stdProduction_lin['OBS_FLAG'].isin(['c','n'])]
stdProduction_lin = stdProduction_lin.drop(stdProduction_lin[stdProduction_lin['OBS_VALUE'] == 0].index)
stdProduction_lin = stdProduction_lin.dropna(subset = ['OBS_VALUE'])
stdProduction_lin['OBS_VALUE'] = stdProduction_lin['OBS_VALUE']*1000
stdProduction_lin = stdProduction_lin.rename(columns={"OBS_VALUE":"stdTonnes",
                                                      "crops":"mostGrownStd"})
#orgTonne.info()
stdProduction_lin = stdProduction_lin[~stdProduction_lin['OBS_FLAG'].isin(['c','n'])]
stdProduction_lin = stdProduction_lin.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'OBS_FLAG', 'strucpro'])
stdProduction_lin = stdProduction_lin.sort_values('stdTonnes').drop_duplicates(['geo', 'TIME_PERIOD'], keep='last')

#print(stdProduction_lin.head(300))
#missing_values_table(orgTonne)




higherEdu = pd.read_csv(higherEdu_Path)

higherEdu = higherEdu[higherEdu['sectperf'].isin(['HES']) ]
higherEdu = higherEdu[higherEdu['unit'].isin(['EUR_HAB']) ]
higherEdu = higherEdu.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'OBS_FLAG', 'unit', 'sectperf'])
higherEdu = higherEdu.rename(columns={"OBS_VALUE":"eduSpend_eur_hab"})

#print(higherEdu.groupby('TIME_PERIOD').geo.nunique())




research = pd.read_csv(research_Path)
research = pd.merge(research, fordCodes, on=['ford'], how='inner')

research = research[research['sectperf'].isin(['GOV']) ]
research = research[research['ford'].isin(['FORD1','FORD2','FORD3','FORD4', 'FORD5', 'FORD6', 'FORD401', 'FORD402', 'FORD403', 'FORD404', 'FORD405', 'FORD504', 'FORD303', 'FORD208']) ]
research = research[research['unit'].isin(['EUR_HAB']) ]
research = research.pivot(index=['geo', 'TIME_PERIOD'], columns='units', values='OBS_VALUE').reset_index() 

print(missing_values_table(research))

#Unfortunately, the data on specific agricultural details was not readily avialable and had too many nulls, will delete columns with greater than 10% nulls as they are effectively useless.
researchColumns = list(research.drop(research.filter(regex=r'geo|TIME_PERIOD').columns, axis=1))
research_null = research[researchColumns].isnull().sum() / len(research)
missing_features_research = list(research_null[research_null > 0.1].index)
research = research.drop(missing_features_research, axis=1)
research = research.fillna(0)

print(missing_values_table(research))
print(research.shape)





#Standard Crop Production Import
stdProduction_lin = pd.read_csv(stdProduction_path)
stdProduction_lin = stdProduction_lin[stdProduction_lin['crops'].str.contains("0000")]
stdProduction_lin = stdProduction_lin[stdProduction_lin['strucpro'].isin(['AR', 'PR_HU_EU']) ]
stdProduction_lin  = stdProduction_lin[~stdProduction_lin['OBS_FLAG'].isin(['c','n'])]
stdProduction_lin_yield = stdProduction_lin[stdProduction_lin['strucpro'].isin(['YI_HU_EU']) ]
stdProduction = stdProduction_lin.pivot(index=['crops', 'geo', 'TIME_PERIOD'], columns='strucpro', values='OBS_VALUE').reset_index() 
stdProduction = stdProduction.dropna(subset=['AR', 'PR_HU_EU']) #Removes last NAN value in SET
#stdProduction = stdProduction[(stdProduction['AR'] != 0)] #Removes last NAN value in SET
stdProduction['area_HA'] = stdProduction['AR']*1000
stdProduction['tonnes'] = stdProduction['PR_HU_EU']*1000




missing_values_table(stdProduction)
print('Unique Geo:' + str(stdProduction.geo.nunique()))
print(stdProduction.groupby('TIME_PERIOD').geo.nunique())
stdProduction.describe()
stdProduction.info()



#Organic Crop Production Import
orgArea_all = pd.read_csv(orgArea_path)
orgArea_total = orgArea_all[(orgArea_all['agprdmet'] == 'TOTAL') & (orgArea_all['unit']=='HA' ) & orgArea_all['crops'].str.contains("0000")]
orgArea_total = orgArea_total.rename(columns={"OBS_VALUE":"area_HA"})
orgArea_total.info()
orgArea_total = orgArea_total[~orgArea_total['OBS_FLAG'].isin(['c','n'])]
orgArea_total = orgArea_total.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'unit', 'agprdmet', 'OBS_FLAG'])
orgArea_total = orgArea_total.dropna(subset=['area_HA']) #Removes last NAN value in SET
orgArea_total = orgArea_total[(orgArea_total['area_HA'] != 0)] #Removes 0 area values in SET which cause inf
missing_values_table(orgArea_total)
orgArea_total.head(25)

#del orgArea_all

orgTonne = pd.read_csv(orgTonne_path)
orgTonne = orgTonne[orgTonne['crops'].str.contains("0000")]
orgTonne = orgTonne.rename(columns={"OBS_VALUE":"tonnes"})
orgTonne.info()
orgTonne = orgTonne[~orgTonne['OBS_FLAG'].isin(['c','n'])]
orgTonne = orgTonne.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'unit', 'OBS_FLAG'])
missing_values_table(orgTonne)

orgProduction = pd.merge(orgArea_total, orgTonne, on=['crops', 'geo', 'TIME_PERIOD'], how='inner', suffixes=('_A','_T'))
orgProduction.head(100)




cropProd = pd.merge( stdProduction, orgProduction, on=['crops', 'geo', 'TIME_PERIOD'], how='inner', suffixes=('_std','_org'))
cropProd = pd.merge( cropProd, cropsCodes, on=['crops'], how='inner')
cropProd['geo'] = cropProd['geo'].astype('str') 
print(cropProd.groupby('TIME_PERIOD').geo.nunique())
hriPesticide.shape
cropProd.describe()
cropProd.info()
cropProd.head()



cropProd['tonne_per_HA_org'] = cropProd['tonnes_org']/cropProd['area_HA_org']
cropProd['tonne_per_HA_std'] = cropProd['tonnes_std']/cropProd['area_HA_std']
cropProd['util_ratio'] = cropProd['tonne_per_HA_org']/cropProd['tonne_per_HA_std']

cropProdTotals = cropProd[cropProd['crops'].str.contains("0000")]

cropProdTotals_Geo_Y= cropProdTotals.groupby(['geo', 'TIME_PERIOD']).sum(numeric_only = True).reset_index()
#cropProdTotals_Geo= cropProdTotals.groupby(['geo']).sum(numeric_only = True).reset_index()
#cropProdTotals_crop_Y= cropProdTotals.groupby(['crops', 'crop_name', 'TIME_PERIOD']).sum(numeric_only = True).reset_index()
#cropProdTotals_crop= cropProdTotals.groupby(['crops', 'crop_name']).sum(numeric_only = True).reset_index()

cropProdTotals_Geo_Y.shape
cropProdTotals_Geo_Y.head(10)
#print(cropProdTotals.groupby(['TIME_PERIOD', 'geo']).size())


# # Variable Creation



#%whos DataFrame
#hriPesticide.head()
hriPesticide.shape




import functools as ft
#extraVars = [hriPesticide, orgProcessors, orgAreaUtil, countryGini, cropProdTotals_Geo_Y, birdBiodiversity, emplyomentRate, income, fertUse, productivityIndex, wasteGeneration]
extraVars = [hriPesticide, orgProcessors, orgAreaUtil, countryGini, birdBiodiversity, emplyomentRate, income, fertUse, productivityIndex, wasteGeneration, pestSales, orgTonne, stdProduction_lin, higherEdu, research]
rds = ft.reduce(lambda left, right: pd.merge(left,right, how='left', on=['geo', 'TIME_PERIOD']), extraVars)

#rds.shape
#rds.head(20)
#print(extraVars_df.groupby(['TIME_PERIOD', 'geo']).size())
#missing_values_table(rds)#Will likely keep NAN values and use as a category when clustered to create scorecard perhaps?




rds["mostGrownOrganic"] = rds["mostGrownOrganic"].fillna("NoOrganic")
rds["orgTonnes"] = rds["orgTonnes"].fillna(0)
rds["mostGrownStd"] = rds["mostGrownStd"].fillna("NoStd")
rds["stdTonnes"] = rds["stdTonnes"].fillna(0)
rds["mostFrequentPest"] = rds["mostFrequentPest"].fillna("NoPest")
rds["C101"] = rds["C101"].fillna(0)
rds["C102"] = rds["C102"].fillna(0)
rds["C103"] = rds["C103"].fillna(0)
rds["C104"] = rds["C104"].fillna(0)
rds["C105"] = rds["C105"].fillna(0)


organicGrown = pd.get_dummies(rds["mostGrownOrganic"], prefix='org_')
stdGrown = pd.get_dummies(rds["mostGrownStd"], prefix='std_')
pestMax = pd.get_dummies(rds["mostFrequentPest"])

rds=pd.merge(rds,organicGrown, left_index=True, right_index=True)
rds=pd.merge(rds,stdGrown, left_index=True, right_index=True)
rds=pd.merge(rds,pestMax, left_index=True, right_index=True)

#rds.head(20)




missing_values_table(rds)

rdsColumns = list(rds.drop(rds.filter(regex=r'geo|TIME_PERIOD').columns, axis=1))
rds_null = rds[rdsColumns].isnull().sum() / len(rds)
missing_features_rds = list(rds_null[rds_null > 0.1].index)
rds = rds.drop(missing_features_rds, axis=1)

rds['N_use_tonne'] = rds['N_use_tonne'].fillna(rds.groupby('geo')['N_use_tonne'].transform('mean'))
rds['P_use_tonne'] = rds['P_use_tonne'].fillna(rds.groupby('geo')['P_use_tonne'].transform('mean'))
rds['pest_KG'] = rds['pest_KG'].fillna(rds.groupby('geo')['pest_KG'].transform('mean'))
rds['waste'] = rds['waste'].fillna(rds.groupby('geo')['waste'].transform('mean'))
rds['areaUsedForOrganic_PCT'] = rds['areaUsedForOrganic_PCT'].fillna(rds.groupby('geo')['areaUsedForOrganic_PCT'].transform('mean'))

missing_values_table(rds)




#rds.to_csv(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Datasets\referenceDataSet.csv")




#cropProdTotals_Geo_Y.to_csv(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Datasets\cropProdTotals_Geo_Y.csv")
#pestSales.to_csv(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Datasets\pestSales.csv")
#pestUse.to_csv(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Datasets\pestUse.csv")






