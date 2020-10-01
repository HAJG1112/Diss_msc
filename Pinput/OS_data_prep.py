import pandas as pd
import numpy as np
# LETS GET OUR VIX DATA
vix = pd.read_csv(r"Pdata/vix.csv", index_col = 0)
vix = pd.DataFrame(vix['Close'])
vix = pd.DataFrame(vix.loc['2017-01-01': '2020-06-25'])
vix = vix.rename(columns = {'Close': 'vix'})

#LETS GET THE EXOGENOUS VARIABLES 
#SENSEX index data
bsesn = pd.read_csv(r"Pdata/BSESN.csv", index_col = 0)
bsesn_dr = pd.DataFrame(bsesn['return'])
bsesn_idr = pd.DataFrame(bsesn['idr_ret'])
bsesn_dr = pd.DataFrame(bsesn_dr.loc['2017-01-01': '2020-06-25'])
bsesn_idr = pd.DataFrame(bsesn_idr.loc['2017-01-01': '2020-06-25'])
bsesn_dr = bsesn_dr.rename(columns = {'return': 'bsesn_dr'})
bsesn_idr = bsesn_idr.rename(columns = {'idr_ret': 'bsesn_idr'})

#MOVE index
move = pd.read_csv(r"Pdata/MOVE.csv", index_col = 0)
move = pd.DataFrame(move.loc['2017-01-01': '2020-06-25'])
move = move.rename(columns = {'Close': 'MOVE'})
move_l1 = move.shift(1)
move_l1 = move_l1.rename(columns = {'MOVE': 'MOVE_l1'})

#S&P500 - split this into positive and negative, and take the 5 and 22 day MA as well like the L-HAR
snp = pd.read_csv(r"Pdata/snp.csv", index_col = 0)
snp = pd.DataFrame(snp.loc['2017-01-01': '2020-06-25'])

#USD index 
usd = pd.read_csv(r"Pdata/USD_index.csv", index_col = 0)
usd = pd.DataFrame(usd.loc['2017-01-01': '2020-06-25'])
usd = pd.DataFrame(usd['return'])
usd = usd.rename(columns = {'return': 'USD_lret'})
usd_l1 = usd.shift(1)
usd_l1 = usd_l1.rename(columns = {'USD_lret': 'USD_lret_l1'})

#WTI
wti = pd.read_csv(r"Pdata/WTIc1.csv", index_col = 0)
wti = pd.DataFrame(wti.loc['2017-01-01': '2020-06-25'])
wti = pd.DataFrame(wti['return'])
wti = wti.rename(columns = {'return': 'WTI_lret'})
wti_l1 = wti.shift(1)
wti_l1 = wti_l1.rename(columns = {'WTI_lret': 'WTI_lret_l1'})

#VVIX
vvix = pd.read_csv(r"Pdata/vvix.csv",  index_col = 0)
vvix = pd.DataFrame(vvix.loc['2017-01-01': '2020-06-25'])
vvix = pd.DataFrame(vvix['Close'])
vvix = vvix.rename(columns = {'Close': 'vvix'})
vvix_l1 = vvix.shift(1)
vvix_l1 = vvix_l1.rename(columns = {'vvix': 'vvix_l1'})

#SKEW
skew = pd.read_csv(r"Pdata/skew.csv",  index_col = 0)
skew = pd.DataFrame(skew.loc['2017-01-01': '2020-06-25'])
skew_l1 = skew.shift(1)
skew_l1 = skew_l1.rename(columns = {'skew': 'skew_l1'})
skew_l1 = skew_l1.rename(columns = {'skew': 'skew_l1'})
skew_change = pd.DataFrame(skew - skew.shift(1)).rename(columns = {'skew': 'Skew_diff'})

#Spread Conditional Volatiltiy Forecasts


#concat all the frames and make all fit the vix
data_vix = pd.concat([vix, snp, bsesn_dr, bsesn_idr, move, move_l1, usd, usd_l1, wti, wti_l1, vvix, vvix_l1, skew, skew_l1, skew_change], axis = 1)

#foward fill the missing columns in the exogenous variables
data_vix.iloc[:,1:] = data_vix.iloc[:,1:].ffill()

#drop values that arent in the VIX index
data_vix = data_vix.dropna(subset = ['vix'])

data_vix.to_csv("Pdata/vix_os.csv")
print(data_vix)
