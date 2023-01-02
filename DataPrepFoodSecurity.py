# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 16:29:19 2023

@author: cianw
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

cropsCodes = pd.read_csv("C:/Users/cianw/Documents/dataAnalytics/CA2/Data/Eurostat/Code Dictionary/crops.dic", sep='\t',names=['crops', 'crop_name'], header = None)
strucproCodes = pd.read_csv("C:/Users/cianw/Documents/dataAnalytics/CA2/Data/Eurostat/Code Dictionary/strucpro.dic",sep='\t',names=['code', 'units'], header = None)
unitCodes = pd.read_csv("C:/Users/cianw/Documents/dataAnalytics/CA2/Data/Eurostat/Code Dictionary/unit.dic",sep='\t',names=['code', 'units'], header = None)
fordCodes = pd.read_csv("C:/Users/cianw/Documents/dataAnalytics/CA2/Data/Eurostat/Code Dictionary/ford.dic",sep='\t',names=['ford', 'units'], header = None)
nacer2 = pd.read_csv(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\dic\nace_r2.dic",sep='\t',names=['nace_r2', 'units'], header = None)

HICP_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\prc_hicp_manr_linear.csv")
marketPrices_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\market-prices-all-products_en.csv")
prodInIndustry_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\sts_inpr_m_linear.csv")
foodPrice_Path= Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\prc_fsc_idx_linear.csv")
importPrice_Path= Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\sts_inpi_m_linear.csv")

stdProduction_path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Agricultural Production\Crops\apro_cpnh1_linear.csv")
orgTonne_path= Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Agricultural Production\Crops\org_croppro_linear.csv")
orgArea_path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Eurostat\Agricultural Production\Crops\org_cropar_linear.csv")
poultry_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\apro_ec_poulm_linear.csv")
meatProdTrade_Path = Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\apro_ec_poulm_linear.csv")


HICP = pd.read_csv(HICP_Path)
HICP = HICP[HICP['coicop'].isin(['CP00'])]
HICP.info()
HICP['date'] = pd.to_datetime(HICP['TIME_PERIOD'])
HICP['date_12M'] = HICP['date'] + pd.offsets.DateOffset(years=+1)
HICP['date_p12M'] = HICP['date'] + pd.offsets.DateOffset(months=-6)
HICP = HICP.drop(columns = ['DATAFLOW', 'LAST UPDATE', 'freq', 'unit', 'coicop', 'TIME_PERIOD', 'OBS_FLAG'])
HICP = HICP.rename(columns = {"OBS_VALUE":"HICP"})
print(HICP.shape)
print(HICP.info())
print(HICP.head(10))


foodPrice = pd.read_csv(foodPrice_Path)
foodPrice['date'] = pd.to_datetime(foodPrice['TIME_PERIOD'])
foodPrice = foodPrice[~foodPrice['indx'].isin(['PPI'])]
foodPrice = foodPrice[foodPrice['coicop'].isin(['CP011','CP0111','CP0112','CP0113','CP0114','CP0115','CP0116','CP0117','CP0213'])]
foodPrice = foodPrice.pivot(index=['geo', 'date'], columns=['unit', 'coicop', 'indx'], values='OBS_VALUE').reset_index()
foodPrice.columns = ['_'.join(col) for col in foodPrice.columns.values]
missing_values_table(foodPrice)


marketPrices = pd.read_csv(marketPrices_Path)
marketPrices['date'] = pd.to_datetime(marketPrices['Period'], format='%Y%m')
marketPrices = marketPrices.pivot(index=['Country', 'date'], columns='Product desc', values='MP Market Price').reset_index() 



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

foodPrice = pd.read_csv(foodPrice_Path)
foodPrice_Path= Path(r"C:\Users\cianw\Documents\dataAnalytics\CA2\Data\Food Security\prc_fsc_idx_linear.csv")

























