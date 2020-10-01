import numpy as np
import pandas as pd
import datetime as dt
#Import the relevant fixed income rates
_7d = pd.read_csv(r"Pdata\raw data\FIXED INCOME\7d A2P2.csv", index_col = 0).replace('.', np.NaN).fillna(method='ffill')
_15d = pd.read_csv(r"Pdata\raw data\FIXED INCOME\15d A2P2.csv", index_col = 0).replace('.', np.NaN).fillna(method='ffill')
_3m_aa = pd.read_csv(r"Pdata\raw data\FIXED INCOME\3_m_AA.csv", index_col = 0).replace('.', np.NaN).fillna(method='ffill')
_1w_aa = pd.read_csv(r"Pdata\raw data\FIXED INCOME\1_wk_AA.csv", index_col = 0).replace('.', np.NaN).fillna(method='ffill')
ffr = pd.read_csv(r"Pdata\raw data\FIXED INCOME\EFFR.csv", index_col = 0).replace('.', np.NaN).fillna(method='ffill')
t_sprd = pd.read_csv(r"Pdata\raw data\FIXED INCOME\TEDRATE.csv", index_col = 0).replace('.', np.NaN).fillna(method='ffill').astype(float)

#dataset split for 70% train, 30% test
start_date = dt.datetime(2010,1,4)
split_date = dt.datetime(2016,12,30)
end_date = dt.datetime(2020,6,25)


#Preparation of spread 1
spread1 = pd.concat([_7d, _15d, ffr], axis = 1).dropna()
spread1 = spread1.astype(float)
spread1['diff'] = spread1['15d_a2'] - spread1['7d_a2']
spread1['S_1'] = spread1['diff']/spread1['EFFR'] 
spread1 = spread1.iloc[:,4:]
spread1['spread1_ret'] = spread1 - spread1.shift(1)
spread1_ret = spread1.iloc[:,1]
spread1.to_csv("Pdata/Spread1.csv")

#Preparation of spread 2
spread2 = pd.concat([_3m_aa, _1w_aa], axis = 1).dropna().astype(float)
spread2['spread2'] = spread2.iloc[:,0] - spread2.iloc[:,1]
spread2['spread2_ret'] = spread2['spread2'] - spread2['spread2'].shift(1)
spread2.to_csv("Pdata/Spread2.csv")

#spread3
spread3 = pd.concat([spread2, t_sprd], axis=1).dropna()
spread3['spread3'] = spread3['spread2'] /spread3['TEDRATE']
spread3['spread3_ret'] = spread3['spread3'] - spread3['spread3'].shift(1) 
spread3.to_csv("Pdata/Spread3.csv")