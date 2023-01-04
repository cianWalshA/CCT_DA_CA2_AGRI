# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 16:29:19 2023

@author: cianw
"""
import glob
import pandas as pd
import numpy as np
from pathlib import Path
import pycountry
from countrygroups import EUROPEAN_UNION

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
nacer2 = pd.read_csv(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\dic\nace_r2.dic",sep='\t',names=['nace_r2', 'units'], header = None)

hicp_ROC_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\prc_hicp_manr_linear.csv")
hicp_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\prc_hicp_midx_linear.csv")
marketPrices_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\market-prices-all-products_en.csv")
prodInIndustry_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\sts_inpr_m_linear.csv")
foodPrice_Path= Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\prc_fsc_idx_linear.csv")
importPrice_Path= Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\sts_inpi_m_linear.csv")

stdProduction_path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Agricultural Production\Crops\apro_cpnh1_linear.csv")
orgTonne_path= Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Agricultural Production\Crops\org_croppro_linear.csv")
orgArea_path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Agricultural Production\Crops\org_cropar_linear.csv")
poultry_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\apro_ec_poulm_linear.csv")
meatProdTrade_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\apro_mt_pheadm_linear.csv")

country = [[country.alpha_2,country.alpha_3, country.name] for country in pycountry.countries]
countries = pd.DataFrame(country, columns=['alpha_2', 'alpha_3', 'name'])
eu = pd.DataFrame(EUROPEAN_UNION, columns=['alpha_3'])
eu = pd.merge(eu, countries, on='alpha_3', how='inner')

hicp_ROC = pd.read_csv(hicp_ROC_Path)
hicp_ROC = hicp_ROC[hicp_ROC['coicop'].isin(['CP00'])]
hicp_ROC = hicp_ROC[['geo', 'TIME_PERIOD', 'OBS_VALUE']]
hicp_ROC = hicp_ROC.rename(columns = {"OBS_VALUE":"hicp"})
hicp_ROC['date_roc'] = pd.to_datetime(hicp_ROC['TIME_PERIOD'])
hicp_ROC['date_roc_n12M'] = hicp_ROC['date_roc'] + pd.offsets.DateOffset(years=+1)
hicp_ROC['date_roc_l12M'] = hicp_ROC['date_roc'] + pd.offsets.DateOffset(years=-1)
print(hicp_ROC.head(100))
print(hicp_ROC.shape)
hicp_ROC['dupes']=hicp_ROC.duplicated(subset=['geo', 'date_roc_l12M'])


hicp = pd.read_csv(hicp_Path)
hicp = hicp[hicp['coicop'].isin(['CP00'])]
hicp = hicp[hicp['unit'].isin(['I15'])]
hicp = pd.merge(hicp, eu, left_on='geo', right_on='alpha_2', how='inner')
hicp.info()
print(hicp.shape)
hicp['date'] = pd.to_datetime(hicp['TIME_PERIOD'])
hicp['month_var'] = pd.DatetimeIndex(hicp['date']).month
hicp['year_var'] = pd.DatetimeIndex(hicp['date']).year
hicp['date_n12m'] = hicp['date'] + pd.offsets.DateOffset(years=+1)
hicp = hicp.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'unit', 'coicop', 'TIME_PERIOD', 'OBS_FLAG'])
hicp = hicp.rename(columns = {"OBS_VALUE":"hicp"})
hicp = pd.merge(hicp, hicp_ROC[['geo', 'date_roc_l12M','hicp']], how='left', left_on=['geo', 'date'], right_on=['geo', 'date_roc_l12M'],suffixes=('', '_ROC') )
hicp = pd.merge(hicp, hicp[['geo', 'date_n12m','hicp']], how='left', left_on=['geo', 'date'], right_on=['geo', 'date_n12m'],suffixes=('', '_n12m') )
hicp['dupes']=hicp.duplicated(subset=['geo', 'date'])
hicp = hicp[['geo', 'date', 'month_var', 'year_var', 'date_n12m', 'hicp_ROC', 'hicp', 'hicp_n12m']]
print(hicp.shape)
print(hicp.info())
print(hicp.head(10))


foodPrice = pd.read_csv(foodPrice_Path)
foodPrice['date'] = pd.to_datetime(foodPrice['TIME_PERIOD'])
foodPrice = foodPrice[~foodPrice['indx'].isin(['PPI'])]
foodPrice = foodPrice[foodPrice['coicop'].isin(['CP011','CP0111','CP0112','CP0113','CP0114','CP0115','CP0116','CP0117','CP0213'])]
foodPrice = foodPrice.pivot(index=['geo', 'date'], columns=['unit', 'coicop', 'indx'], values='OBS_VALUE').reset_index()
foodPrice.columns = [''.join(col) for col in foodPrice.columns.values]
print(missing_values_table(foodPrice))


marketPrices = pd.read_csv(marketPrices_Path)
marketPrices['MP Market Price']=marketPrices['MP Market Price'].replace(" ", "")
marketPrices['MP Market Price']=marketPrices['MP Market Price'].astype(float)
marketPrices['date'] = pd.to_datetime(marketPrices['Period'], format='%Y%m')
marketPrices = marketPrices.pivot(index=['Country', 'date'], columns='Product desc', values='MP Market Price').reset_index() 
marketPrices = marketPrices.rename(columns = {"Country":"geo"})




prodInIndustry = pd.read_csv(prodInIndustry_Path)
prodInIndustry = prodInIndustry[prodInIndustry['unit'].isin(['I15', 'PCH_SM','PCH_PRE'])]
prodInIndustry = prodInIndustry[prodInIndustry['s_adj'].isin(['CA', 'SCA'])]
prodInIndustry = prodInIndustry[prodInIndustry['nace_r2'] == "C10"]
prodInIndustry['date'] = pd.to_datetime(prodInIndustry['TIME_PERIOD'])
prodInIndustry = prodInIndustry.pivot(index=['geo', 'date'], columns=['unit','s_adj'], values='OBS_VALUE').reset_index()
prodInIndustry.columns = [''.join(col) for col in prodInIndustry.columns.values]

prodInIndustry = prodInIndustry.rename(columns = {"I15CA":"manu_2015_index_CA",
                                                  "I15SCA":"manu_2015_index_SCA",
                                                  "PCH_SMCS":"manu_pctChange_L12m",
                                                  "PCH_PRESCA":"manu_pctChange_L1m"})

"""
USELESS!!!
importPrice = pd.read_csv(importPrice_Path)
importPrice = importPrice[importPrice['unit'].isin(['I15', 'PCH_SM','PCH_PRE'])]
importPrice = importPrice[importPrice['indic_bt']=="IMPX"]
importPrice = importPrice[importPrice['cpa2_1'].isin(["CPA_C1013", "CPA_C1082"])]
importPrice['date'] = pd.to_datetime(importPrice['TIME_PERIOD'])
importPrice = importPrice.pivot(index=['geo', 'date'], columns=['cpa2_1', 'unit'], values='OBS_VALUE').reset_index()
importPrice.columns = [''.join(col) for col in importPrice.columns.values]
"""

poultry = pd.read_csv(poultry_Path)
poultry = poultry[poultry['animals'].str.contains("A5130O")]
poultry['date'] = pd.to_datetime(poultry['TIME_PERIOD'])
poultry = poultry.pivot(index=['geo', 'date'], columns=['animals', 'hatchitm'], values='OBS_VALUE').reset_index()
poultry.columns = [''.join(col) for col in poultry.columns.values]
#poultry = poultry.fillna(0)
print(missing_values_table(poultry))
print(poultry.info())

meatProdTrade = pd.read_csv(meatProdTrade_Path)
meatProdTrade['date'] = pd.to_datetime(meatProdTrade['TIME_PERIOD'])
meatProdTrade = meatProdTrade[meatProdTrade['meat'].isin(["B1000", "B1100", "B1110", "B1120", "B1200", "B1240", "B1210_B1220"])]
meatProdTrade = meatProdTrade.pivot(index=['geo', 'date'], columns=['meatitem', 'meat', 'unit'], values='OBS_VALUE').reset_index()
meatProdTrade.columns = [''.join(col) for col in meatProdTrade.columns.values]

import functools as ft
#extraVars = [hriPesticide, orgProcessors, orgAreaUtil, countryGini, cropProdTotals_Geo_Y, birdBiodiversity, emplyomentRate, income, fertUse, productivityIndex, wasteGeneration]
extraVars = [hicp, foodPrice, marketPrices, prodInIndustry, prodInIndustry, meatProdTrade]
rds = ft.reduce(lambda left, right: pd.merge(left,right, how='left', on=['geo', 'date']), extraVars)
print(rds.shape)

rds = rds.loc[(rds['date'] >= '2009-01-01') & (rds['date'] <='2020-12-31')]
print(missing_values_table(rds))


rdsColumns = list(rds.drop(rds.filter(regex=r'geo|TIME_PERIOD').columns, axis=1))
rds_null = rds[rdsColumns].isnull().sum() / len(rds)
missing_features_rds = list(rds_null[rds_null > 0.1].index)
rds = rds.drop(missing_features_rds, axis=1)

print(missing_values_table(rds))
rds = rds.fillna(rds.groupby(['geo','year_var']).transform('mean'))
print(missing_values_table(rds))
rds = rds.fillna(rds.groupby(['year_var']).transform('mean'))
print(missing_values_table(rds))


rds.set_index(["geo","date"], inplace=True)
rdsNo_n12m= rds.drop(columns=list(rds.filter(regex=r'n12m|year|month').columns))

#rds_mean_l3m = rds.groupby('geo').rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
rds_mean_l6m = rdsNo_n12m.groupby('geo').rolling(window=6, min_periods=1).mean().reset_index(level=0, drop=True)
#rds_max_l6m = rds.groupby('geo').rolling(window=6, min_periods=1).max().reset_index(level=0, drop=True)
rds_min_l6m = rdsNo_n12m.groupby('geo').rolling(window=6, min_periods=1).min().reset_index(level=0, drop=True)
rds_sum_l6m = rdsNo_n12m.groupby('geo').rolling(window=6, min_periods=1).sum().reset_index(level=0, drop=True)
rds_lag_diff_6m = rdsNo_n12m.groupby('geo').diff(periods=6).reset_index(level=0)
rds_lag_diff_3m = rdsNo_n12m.groupby('geo').diff(periods=3).reset_index(level=0)
rds_lag_diff_1m = rdsNo_n12m.groupby('geo').diff(periods=1).reset_index(level=0)


#rds = pd.merge(rds, rds_mean_l3m,on=['geo', 'date'], how='left', suffixes=('','mean_l3m'))
rds = pd.merge(rds, rds_mean_l6m, on=['geo', 'date'], how='left', suffixes=('','_mean_l6m'))
#rds = pd.merge(rds, rds_max_l6m, on=['geo', 'date'], how='left', suffixes=('','max_l6m'))
rds = pd.merge(rds, rds_min_l6m, on=['geo', 'date'], how='left', suffixes=('','_min_l6m'))
rds = pd.merge(rds, rds_lag_diff_6m, on=['geo', 'date'], how='left', suffixes=('','_diff_6m'))
rds = pd.merge(rds, rds_lag_diff_3m, on=['geo', 'date'], how='left', suffixes=('','_diff_3m'))
rds = pd.merge(rds, rds_lag_diff_1m, on=['geo', 'date'], how='left', suffixes=('','_diff_1m'))
rds = pd.merge(rds, rds_sum_l6m, on=['geo', 'date'], how='left', suffixes=('','_sum_l6m')).reset_index()


#rds.set_index(["geo","date"], inplace=True)

rds.to_csv(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Datasets\referenceDataSet.csv")












