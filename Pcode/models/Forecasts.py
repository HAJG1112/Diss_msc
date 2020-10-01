import pandas as pd
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import arch
import datetime as dt
from arch.univariate.mean import HARX
from arch.univariate.volatility import HARCH, ARCH, EGARCH, GARCH
from arch.univariate import Normal, StudentsT, SkewStudent, GeneralizedError
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from loss_function_and_test import  mse, directional_accuracy, test_forecast, binary_accuracy, ql

#call in our cleaned and split datasets containing the dependent and the exogenous for the VIX
vix_is = pd.read_csv(r"Pdata\vix_is.csv", index_col = 0)
vix_os = pd.read_csv(r"Pdata\vix_os.csv", index_col = 0)
vix_all = pd.concat([vix_is, vix_os])  #join them 
vix_all.index = pd.to_datetime(vix_all.index)  #set the index as a datetime64 object

#call in the Open prices
vix_open = pd.read_csv(r"Pdata\vix.csv", index_col = 0 )
vix_open.index = pd.to_datetime(vix_open.index)
vix_open = vix_open['Open']

#sort out out variable designation here, we need to lag the returns as we assume that it takes one day to compute them to be used for the following day.
snp = pd.DataFrame(vix_all.iloc[:,1]).shift(1) #snp return
snp_pos_ret = pd.DataFrame(vix_all.iloc[:,2]).shift(1) #snp pos_ret
snp_neg_ret = pd.DataFrame(vix_all.iloc[:,3]).shift(1) #snp neg_ret
snp_pos_ret_5 = pd.DataFrame(vix_all.iloc[:,4]).shift(1) #snp pos_ret_5
snp_pos_ret_22 = pd.DataFrame(vix_all.iloc[:,5]).shift(1) #snp pos_ret_22
snp_neg_ret_5 = pd.DataFrame(vix_all.iloc[:,6]).shift(1) #snp neg_ret_5
snp_neg_ret_22 = pd.DataFrame(vix_all.iloc[:,7]).shift(1) #snp neg_ret_22
bsesn_dr = pd.DataFrame(vix_all.iloc[:,8]) # bsesn
bsesn_idr = pd.DataFrame(vix_all.iloc[:,9])
move = pd.DataFrame(vix_all.iloc[:,10]) #MOVE
l1_move = pd.DataFrame(vix_all.iloc[:,11]).rename(columns = {'MOVE': 'MOVE_l1'}).fillna(0) #lag of MOVE
vvix = pd.DataFrame(vix_all.iloc[:,16]) #vvix
vvix_l1 = pd.DataFrame(vix_all.iloc[:,17]).rename(columns = {'VVIX': 'VVIX_l1'}).fillna(0)
skew = pd.DataFrame(vix_all.iloc[:,18]) 
skew_l1 = pd.DataFrame(vix_all.iloc[:,19]).rename(columns = {'skew': 'skew_l1'}).fillna(0)
skew_change = pd.DataFrame(vix_all.iloc[:,20]).rename(columns = {'skew_l1': 'skew_diff'}).fillna(0)


start_date = dt.datetime(2010,1,5)
split_date = dt.datetime(2017,1,6)
end_date = dt.datetime(2020,6,24)

vix_open = vix_open[split_date:end_date]
#strip our dependent variables
vix_val = vix_all['vix']
vix_all = vix_all[split_date:end_date]
vix_actual = vix_all['vix']

#all forecasts are matched according to date of forecast
def get_forecast(dependent, exogenous, lags ,distribution_):
    har = HARX(y = dependent, x = exogenous, lags = [1,5,22],distribution = distribution_,rescale=True)
    res = har.fit(options = {'maxiter': 10000}, cov_type='robust', first_obs=start_date, last_obs = split_date)
    forecast = res.forecast(horizon = 1, align = 'target', start = split_date)
    forecast = forecast.mean.dropna()
    return forecast

def get_forecast_egarch(dependent, exogenous, lags, distribution_):
    har = HARX(y = dependent, x = exogenous, lags = [1,5,22], volatility = EGARCH(1,1), distribution = distribution_)
    res = har.fit(options = {'maxiter': 10000}, cov_type='robust', first_obs=start_date, last_obs = split_date)
    forecast = res.forecast(horizon = 1, align = 'target',start = split_date)
    forecast = forecast.mean.dropna()
    return forecast

def get_forecast_3d(dependent, exogenous, lags, volatility, distribution_):
    har = HARX(y = dependent, x = exogenous, lags = [1,5,22], volatility = volatility ,distribution = distribution_)
    res = har.fit(options = {'maxiter': 10000, 'ftol':1e-10, 'eps':1e-12}, cov_type='robust',  last_obs = split_date)
    forecast = res.forecast(horizon = 3, align = 'target', method = 'bootstrap', start = split_date)
    forecast = forecast.mean[split_date:end_date]
    return forecast

def get_forecast_5d(dependent, exogenous, lags, volatility, distribution_):
    har = HARX(y = dependent, x = exogenous, lags = [1,5,22], volatility = volatility, distribution = distribution_)
    res = har.fit(options = {'maxiter': 10000, 'ftol':1e-12, 'eps':1e-14}, cov_type='robust',  last_obs = split_date)
    forecast = res.forecast(horizon = 5, align = 'target', method = 'bootstrap', start = split_date)
    forecast = forecast.mean.dropna()
    return forecast

def get_har_result(forecast, close, open1):
    forecast = forecast.rename(columns = {'h.1':'forecast'})
    forecast = forecast.iloc[:,0]
    compare = pd.concat([forecast, close], axis=1)
    #loss functions
    compare['close_ae'] = abs(compare.iloc[:,0] - compare.iloc[:,1])
    compare['close_mse'] = mse(compare.iloc[:,0], compare.iloc[:,1])
    compare['close_ql'] = ql(compare.iloc[:,0], compare.iloc[:,1])
    #daily return and directional accuracy
    compare['forecast_DR'] = np.log(compare.iloc[:,0]/compare.iloc[:,1].shift(1))
    compare['actual_DR'] = np.log(compare.iloc[:,1]/compare.iloc[:,1].shift(1))
    compare['DA_DR'] = np.where(np.sign(compare['forecast_DR'])==np.sign(compare['actual_DR']),1,0)
    #IDR forecast and directional accuracy, assumed we trade at open the next day
    compare['forecast_IDR'] = np.log(compare.iloc[:,0]/open1)
    compare['actual_IDR'] = np.log(compare.iloc[:,1]/open1)

    compare['DA_IDR'] = np.where(np.sign(compare['forecast_IDR'])==np.sign(compare['actual_IDR']),1,0) #if true broadcast 1, 0 oterwise
    compare['close_err'] = (compare.iloc[:,0] - compare.iloc[:,1])
    return compare.dropna()

def get_har_result_3d(forecast, close, open1):
    forecast = forecast.rename(columns = {'h.3':'forecast-3d'})
    forecast = forecast.iloc[:,2:]
    compare = pd.concat([forecast, close], axis=1)

    #loss functions
    compare['close_ae'] = abs(compare.iloc[:,0] - compare.iloc[:,1])
    compare['close_mse'] = mse(compare.iloc[:,0],compare.iloc[:,1])
    compare['close_ql'] = ql(compare.iloc[:,0], compare.iloc[:,1])

    #daily return and directional accuracy
    compare['forecast_3DR'] =  np.log(compare.iloc[:,0]/compare.iloc[:,1].shift(3))
    compare['actual_3DR'] = np.log(compare.iloc[:,1]/compare.iloc[:,1].shift(3))
    compare['DA_3DR'] = np.where(np.sign(compare['forecast_3DR'])==np.sign(compare['actual_3DR']),1,0)

    #IDR forecast and directional accuracy, assumed we trade at open the next day
    compare['forecast_I3DR'] = np.log(compare.iloc[:,0]/open1.shift(2))
    compare['actual_I3DR'] = np.log(compare.iloc[:,1]/open1.shift(2))
    compare['DA_I3DR'] = np.where(np.sign(compare['forecast_I3DR'])==np.sign(compare['actual_I3DR']),1,0) #if true broadcast 1, 0 oterwise
    compare['close_err'] = (compare.iloc[:,0] - compare.iloc[:,1])

    return compare.dropna()

def get_har_result_5d(forecast, close, open1):
    forecast = forecast.rename(columns = {'h.5':'forecast-5d'})
    forecast = forecast.iloc[:,4:]
    compare = pd.concat([forecast, close], axis=1)
    #loss functions
    compare['close_ae'] = abs(compare.iloc[:,0] - compare.iloc[:,1])
    compare['close_mse'] = mse(compare.iloc[:,0],compare.iloc[:,1])
    compare['close_ql'] = ql(compare.iloc[:,0], compare.iloc[:,1])

    #daily return and directional accuracy
    compare['forecast_5DR'] = np.log(compare.iloc[:,0]/compare.iloc[:,1].shift(4))
    compare['actual_5DR'] = np.log(compare.iloc[:,1]/compare.iloc[:,1].shift(4))
    compare['DA_5DR'] = np.where(np.sign(compare['forecast_5DR'])==np.sign(compare['actual_5DR']),1,0)

    #IDR forecast and directional accuracy, assumed we trade at open the next day
    compare['forecast_I5DR'] = np.log(compare.iloc[:,0]/open1.shift(4))
    compare['actual_I5DR'] = np.log(compare.iloc[:,1]/open1.shift(4))
    compare['DA_I5DR'] = np.where(np.sign(compare['forecast_I5DR'])==np.sign(compare['actual_I5DR']),1,0) #if true broadcast 1, 0 oterwise
    compare['close_err'] = (compare.iloc[:,0] - compare.iloc[:,1])
    return compare.dropna()

#these our the distributions for testing
norm = arch.univariate.Normal()  #set parameter values for nu and lambda
ss = arch.univariate.SkewStudent() #standardized Skewed student
ged = arch.univariate.GeneralizedError()  #generalized error distribution
st = arch.univariate.StudentsT()

#variable grouping
leverage = pd.concat([snp_neg_ret, snp_neg_ret_5, snp_neg_ret_22], axis = 1)
l1 = pd.concat([snp_neg_ret, snp_neg_ret_5], axis = 1)
vol = pd.concat([l1_move, vvix_l1, skew_l1], axis = 1)
vol_1 = pd.concat([vvix_l1, skew_l1], axis = 1)

#forecasts for the VIX under the 4 different distributions
y_hat_normal_vix = get_forecast(vix_val, None, [1,5,22], norm)
y_hat_skewstud_vix = get_forecast(vix_val, None,  [1,5,22], ss)
y_hat_GED_vix = get_forecast(vix_val, None, [1,5,22], ged)
y_hat_studT_vix = get_forecast(vix_val, None, [1,5,22], st)

y_hat_normal_vix_egarch = get_forecast_egarch(vix_val, None, [1,5,22], norm)
y_hat_skewstud_vix_egarch = get_forecast_egarch(vix_val, None, [1,5,22], ss)
y_hat_GED_vix_egarch = get_forecast_egarch(vix_val, None, [1,5,22], ged)
y_hat_studT_vix_egarch = get_forecast_egarch(vix_val, None, [1,5,22], st)

y_hat_normal_vix_3d = get_forecast_3d(vix_val, None, [1,5,22], None,norm)
y_hat_skewstud_vix_3d = get_forecast_3d(vix_val, None,  [1,5,22], None, ss)
y_hat_GED_vix_3d = get_forecast_3d(vix_val, None, [1,5,22], None,ged)
y_hat_studT_vix_3d = get_forecast_3d(vix_val, None, [1,5,22],None, st)

y_hat_normal_vix_egarch_3d = get_forecast_3d(vix_val, None,  [1,5,22], EGARCH(1,1), norm)
y_hat_skewstud_vix_egarch_3d = get_forecast_3d(vix_val, None, [1,5,22], EGARCH(1,1), ss)
y_hat_GED_vix_egarch_3d = get_forecast_3d(vix_val, None, [1,5,22], EGARCH(1,1), ged)
y_hat_studT_vix_egarch_3d = get_forecast_3d(vix_val, None, [1,5,22], EGARCH(1,1), st)

y_hat_normal_vix_5d = get_forecast_5d(vix_val, None, [1,5,22], None,norm)
y_hat_skewstud_vix_5d = get_forecast_5d(vix_val, None,  [1,5,22], None,ss)
y_hat_GED_vix_5d = get_forecast_5d(vix_val, None, [1,5,22], None,ged)
y_hat_studT_vix_5d = get_forecast_5d(vix_val, None, [1,5,22],None, st)

y_hat_normal_vix_egarch_5d = get_forecast_5d(vix_val, None, [1,5,22],EGARCH(1,1), norm)
y_hat_skewstud_vix_egarch_5d = get_forecast_5d(vix_val, None, [1,5,22],EGARCH(1,1), ss)
y_hat_GED_vix_egarch_5d = get_forecast_5d(vix_val, None,[1,5,22],EGARCH(1,1), ged)
y_hat_studT_vix_egarch_5d = get_forecast_5d(vix_val, None, [1,5,22], EGARCH(1,1), st)

#get the results for the base VIX models
normal_vix = get_har_result(y_hat_normal_vix, vix_actual, vix_open)
skewstud_vix = get_har_result(y_hat_skewstud_vix, vix_actual, vix_open)
GED_vix = get_har_result(y_hat_GED_vix, vix_actual, vix_open)
studT_vix = get_har_result(y_hat_studT_vix, vix_actual, vix_open)

normal_vix_egarch = get_har_result(y_hat_normal_vix_egarch, vix_actual, vix_open)
skewstud_vix_egarch = get_har_result(y_hat_skewstud_vix_egarch, vix_actual, vix_open)
GED_vix_egarch = get_har_result(y_hat_GED_vix_egarch, vix_actual, vix_open)
studT_vix_egarch = get_har_result(y_hat_studT_vix_egarch, vix_actual, vix_open)

normal_vix_3d = get_har_result_3d(y_hat_normal_vix_3d, vix_actual, vix_open)
skewstud_vix_3d = get_har_result_3d(y_hat_skewstud_vix_3d, vix_actual, vix_open)
GED_vix_3d = get_har_result_3d(y_hat_GED_vix_3d, vix_actual, vix_open)
studT_vix_3d = get_har_result_3d(y_hat_studT_vix_3d, vix_actual, vix_open)

normal_vix_egarch_3d = get_har_result_3d(y_hat_normal_vix_egarch_3d, vix_actual, vix_open)
skewstud_vix_egarch_3d = get_har_result_3d(y_hat_skewstud_vix_egarch_3d, vix_actual, vix_open)
GED_vix_egarch_3d = get_har_result_3d(y_hat_GED_vix_egarch_3d, vix_actual, vix_open)
studT_vix_egarch_3d = get_har_result_3d(y_hat_studT_vix_egarch_3d, vix_actual, vix_open)

normal_vix_5d = get_har_result_5d(y_hat_normal_vix_5d, vix_actual, vix_open)
skewstud_vix_5d = get_har_result_5d(y_hat_skewstud_vix_5d, vix_actual, vix_open)
GED_vix_5d = get_har_result_5d(y_hat_GED_vix_5d, vix_actual, vix_open)
studT_vix_5d = get_har_result_5d(y_hat_studT_vix_5d, vix_actual, vix_open)

normal_vix_egarch_5d = get_har_result_5d(y_hat_normal_vix_egarch_5d, vix_actual, vix_open)
skewstud_vix_egarch_5d = get_har_result_5d(y_hat_skewstud_vix_egarch_5d, vix_actual, vix_open)
GED_vix_egarch_5d = get_har_result_5d(y_hat_GED_vix_egarch_5d, vix_actual, vix_open)
studT_vix_egarch_5d = get_har_result_5d(y_hat_studT_vix_egarch_5d, vix_actual, vix_open)

#Forecasts for the VIX with exogenous variables (1-day)
y_hat_m1_n = get_forecast(vix_val, snp, [1,5,22], norm)
y_hat_m1_st = get_forecast(vix_val, snp, [1,5,22], st)
y_hat_m1_ged = get_forecast(vix_val, snp, [1,5,22], ged)
y_hat_m2_n = get_forecast(vix_val, l1, [1,5,22], norm)
y_hat_m2_st = get_forecast(vix_val, l1, [1,5,22], st)
y_hat_m2_ged = get_forecast(vix_val, l1, [1,5,22], ged)
y_hat_m2a_n = get_forecast(vix_val, leverage, [1,5,22], norm)
y_hat_m2a_st = get_forecast(vix_val, leverage, [1,5,22], st)
y_hat_m2a_ged = get_forecast(vix_val, leverage, [1,5,22], ged)
y_hat_m3_n = get_forecast(vix_val, bsesn_dr, [1,5,22], norm)
y_hat_m3_st = get_forecast(vix_val, bsesn_dr, [1,5,22], st)
y_hat_m3_ged = get_forecast(vix_val, bsesn_dr, [1,5,22], ged)
y_hat_m4_n = get_forecast(vix_val, bsesn_idr, [1,5,22], norm)
y_hat_m4_st = get_forecast(vix_val, bsesn_idr, [1,5,22], st)
y_hat_m4_ged = get_forecast(vix_val, bsesn_idr, [1,5,22], ged)
y_hat_m5_ged = get_forecast(vix_val, l1_move, [1,5,22], ged)
y_hat_m6_ged = get_forecast(vix_val, vvix_l1, [1,5,22], ged)
y_hat_m7_ged = get_forecast(vix_val, skew_l1, [1,5,22], ged)
y_hat_m8_n = get_forecast(vix_val, skew_change, [1,5,22], norm)
y_hat_m8_st = get_forecast(vix_val, skew_change, [1,5,22], st)
y_hat_m8_ged = get_forecast(vix_val, skew_change, [1,5,22], ged)
y_hat_m9_n = get_forecast(vix_val, vol_1, [1,5,22], norm)
y_hat_m9_st = get_forecast(vix_val, vol_1, [1,5,22], st)

#Forecasts for the VIX with exogenous variables (1-day) egarch
y_hat_m1_ne = get_forecast_egarch(vix_val, snp, [1,5,22], norm)
y_hat_m1_ste = get_forecast_egarch(vix_val, snp, [1,5,22], st)
y_hat_m1_gede = get_forecast_egarch(vix_val, snp, [1,5,22], ged)
y_hat_m2_ne = get_forecast_egarch(vix_val, l1, [1,5,22], norm)
y_hat_m2_ste = get_forecast_egarch(vix_val, l1, [1,5,22], st)
y_hat_m2_gede = get_forecast_egarch(vix_val, l1, [1,5,22], ged)
y_hat_m2a_ne = get_forecast_egarch(vix_val, leverage, [1,5,22], norm)
y_hat_m2a_ste = get_forecast_egarch(vix_val, leverage, [1,5,22], st)
y_hat_m2a_gede = get_forecast_egarch(vix_val, leverage, [1,5,22], ged)
y_hat_m3_ne = get_forecast_egarch(vix_val, bsesn_dr, [1,5,22], norm)
y_hat_m3_ste = get_forecast_egarch(vix_val, bsesn_dr, [1,5,22], st)
y_hat_m3_gede = get_forecast_egarch(vix_val, bsesn_dr, [1,5,22], ged)
y_hat_m4_ne = get_forecast_egarch(vix_val, bsesn_idr, [1,5,22], norm)
y_hat_m4_ste = get_forecast_egarch(vix_val, bsesn_idr, [1,5,22], st)
y_hat_m4_gede = get_forecast_egarch(vix_val, bsesn_idr, [1,5,22], ged)
y_hat_m5_gede= get_forecast_egarch(vix_val, l1_move, [1,5,22], ged)
y_hat_m6_gede = get_forecast_egarch(vix_val, vvix_l1, [1,5,22], ged)
y_hat_m7_gede = get_forecast_egarch(vix_val, skew_l1, [1,5,22], ged)
y_hat_m8_ne = get_forecast_egarch(vix_val, skew_change, [1,5,22], norm)
y_hat_m8_ste = get_forecast_egarch(vix_val, skew_change, [1,5,22], st)
y_hat_m8_gede = get_forecast_egarch(vix_val, skew_change, [1,5,22], ged)
y_hat_m9_ne = get_forecast_egarch(vix_val, vol_1, [1,5,22], norm)
y_hat_m9_ste = get_forecast_egarch(vix_val, vol_1, [1,5,22], st)

#get the results for the VIX exogenous models
M1_n = get_har_result(y_hat_m1_n, vix_actual, vix_open)
M1_st = get_har_result(y_hat_m1_st, vix_actual, vix_open)
M1_ged = get_har_result(y_hat_m1_ged, vix_actual, vix_open)
M2_n = get_har_result(y_hat_m2_n, vix_actual, vix_open)
M2_st = get_har_result(y_hat_m2_st, vix_actual, vix_open)
M2_ged = get_har_result(y_hat_m2_ged, vix_actual, vix_open)
M2a_n = get_har_result(y_hat_m2a_n, vix_actual, vix_open)
M2a_st = get_har_result(y_hat_m2a_st, vix_actual, vix_open)
M2a_ged = get_har_result(y_hat_m2a_ged, vix_actual, vix_open)
M3_n = get_har_result(y_hat_m3_n, vix_actual, vix_open)
M3_st = get_har_result(y_hat_m3_st, vix_actual, vix_open)
M3_ged = get_har_result(y_hat_m3_ged, vix_actual, vix_open)
M4_n = get_har_result(y_hat_m4_n, vix_actual, vix_open)
M4_st = get_har_result(y_hat_m4_st, vix_actual, vix_open)
M4_ged = get_har_result(y_hat_m4_ged, vix_actual, vix_open)
M5_ged = get_har_result(y_hat_m5_ged, vix_actual, vix_open)
M6_ged = get_har_result(y_hat_m6_ged, vix_actual, vix_open)
M7_ged = get_har_result(y_hat_m7_ged, vix_actual, vix_open)
M8_n = get_har_result(y_hat_m8_n, vix_actual, vix_open)
M8_st = get_har_result(y_hat_m8_st, vix_actual, vix_open)
M8_ged = get_har_result(y_hat_m8_ged, vix_actual, vix_open)
M9_n = get_har_result(y_hat_m9_n, vix_actual, vix_open)
M9_st = get_har_result(y_hat_m9_st, vix_actual, vix_open)

#get results for Exog-Egarch
#get the results for the VIX exogenous models
M1_ne = get_har_result(y_hat_m1_ne, vix_actual, vix_open)
M1_ste = get_har_result(y_hat_m1_ste, vix_actual, vix_open)
M1_gede = get_har_result(y_hat_m1_gede, vix_actual, vix_open)
M2_ne = get_har_result(y_hat_m2_ne, vix_actual, vix_open)
M2_ste = get_har_result(y_hat_m2_ste, vix_actual, vix_open)
M2_gede = get_har_result(y_hat_m2_gede, vix_actual, vix_open)
M2a_ne = get_har_result(y_hat_m2a_ne, vix_actual, vix_open)
M2a_ste = get_har_result(y_hat_m2a_ste, vix_actual, vix_open)
M2a_gede = get_har_result(y_hat_m2a_gede, vix_actual, vix_open)
M3_ne = get_har_result(y_hat_m3_ne, vix_actual, vix_open)
M3_ste = get_har_result(y_hat_m3_ste, vix_actual, vix_open)
M3_gede = get_har_result(y_hat_m3_gede, vix_actual, vix_open)
M4_ne = get_har_result(y_hat_m4_ne, vix_actual, vix_open)
M4_ste = get_har_result(y_hat_m4_ste, vix_actual, vix_open)
M4_gede = get_har_result(y_hat_m4_gede, vix_actual, vix_open)
M5_gede = get_har_result(y_hat_m5_gede, vix_actual, vix_open)
M6_gede = get_har_result(y_hat_m6_gede, vix_actual, vix_open)
M7_gede = get_har_result(y_hat_m7_gede, vix_actual, vix_open)
M8_ne = get_har_result(y_hat_m8_ne, vix_actual, vix_open)
M8_ste = get_har_result(y_hat_m8_ste, vix_actual, vix_open)
M8_gede = get_har_result(y_hat_m8_gede, vix_actual, vix_open)
M9_ne = get_har_result(y_hat_m9_ne, vix_actual, vix_open)
M9_ste = get_har_result(y_hat_m9_ste, vix_actual, vix_open)


#Send the chosen exogenous results to CSV's
M1_n.to_csv("Poutput/VIX_X_forecast_result/base/1d/M1_n.csv")
M1_st.to_csv("Poutput/VIX_X_forecast_result/base/1d/M1_st.csv")
M1_ged.to_csv("Poutput/VIX_X_forecast_result/base/1d/M1_ged.csv")
M2_n.to_csv("Poutput/VIX_X_forecast_result/base/1d/M2_n.csv")
M2_st.to_csv("Poutput/VIX_X_forecast_result/base/1d/M2_st.csv")
M2_ged.to_csv("Poutput/VIX_X_forecast_result/base/1d/M2_ged.csv")
M2a_n.to_csv("Poutput/VIX_X_forecast_result/base/1d/M2a_n.csv")
M2a_st.to_csv("Poutput/VIX_X_forecast_result/base/1d/M2a_st.csv")
M2a_ged.to_csv("Poutput/VIX_X_forecast_result/base/1d/M2a_ged.csv")
M3_n.to_csv("Poutput/VIX_X_forecast_result/base/1d/M3_n.csv")
M3_st.to_csv("Poutput/VIX_X_forecast_result/base/1d/M3_st.csv")
M3_ged.to_csv("Poutput/VIX_X_forecast_result/base/1d/M3_ged.csv")
M4_n.to_csv("Poutput/VIX_X_forecast_result/base/1d/M4_n.csv")
M4_st.to_csv("Poutput/VIX_X_forecast_result/base/1d/M4_st.csv")
M4_ged.to_csv("Poutput/VIX_X_forecast_result/base/1d/M4_ged.csv")
M5_ged.to_csv("Poutput/VIX_X_forecast_result/base/1d/M5_ged.csv")
M6_ged.to_csv("Poutput/VIX_X_forecast_result/base/1d/M6_ged.csv")
M7_ged.to_csv("Poutput/VIX_X_forecast_result/base/1d/M7_ged.csv")
M8_n.to_csv("Poutput/VIX_X_forecast_result/base/1d/M8_n.csv")
M8_st.to_csv("Poutput/VIX_X_forecast_result/base/1d/M8_st.csv")
M8_ged.to_csv("Poutput/VIX_X_forecast_result/base/1d/M8_ged.csv")
M9_n.to_csv("Poutput/VIX_X_forecast_result/base/1d/M9_n.csv")
M9_st.to_csv("Poutput/VIX_X_forecast_result/base/1d/M9_st.csv")

#Send the chosen exogenous results to CSV's
M1_ne.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M1_n.csv")
M1_ste.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M1_st.csv")
M1_gede.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M1_ged.csv")
M2_ne.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M2_n.csv")
M2_ste.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M2_st.csv")
M2_gede.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M2_ged.csv")
M2a_ne.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M2a_n.csv")
M2a_ste.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M2a_st.csv")
M2a_gede.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M2a_ged.csv")
M3_ne.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M3_n.csv")
M3_ste.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M3_st.csv")
M3_gede.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M3_ged.csv")
M4_ne.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M4_n.csv")
M4_ste.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M4_st.csv")
M4_gede.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M4_ged.csv")
M5_gede.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M5_ged.csv")
M6_gede.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M6_ged.csv")
M7_gede.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M7_ged.csv")
M8_ne.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M8_n.csv")
M8_ste.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M8_st.csv")
M8_gede.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M8_ged.csv")
M9_ne.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M9_n.csv")
M9_ste.to_csv("Poutput/VIX_X_forecast_result/egarch/1d/M9_st.csv")

#Send the base results to CSV's in Poutput for later analysis 
normal_vix.to_csv("Poutput/VIX_base_forecast_result/1d/Normal.csv")
skewstud_vix.to_csv("Poutput/VIX_base_forecast_result/1d/skewstud.csv")
GED_vix.to_csv("Poutput/VIX_base_forecast_result/1d/GED.csv")
studT_vix.to_csv("Poutput/VIX_base_forecast_result/1d/studT.csv")

normal_vix_egarch.to_csv("Poutput/VIX_base_forecast_result/1d/Normal_egarch.csv")
skewstud_vix_egarch.to_csv("Poutput/VIX_base_forecast_result/1d/skewstud_egarch.csv")
GED_vix_egarch.to_csv("Poutput/VIX_base_forecast_result/1d/GED_egarch.csv")
studT_vix_egarch.to_csv("Poutput/VIX_base_forecast_result/1d/studT_egarch.csv")

normal_vix_3d.to_csv("Poutput/VIX_base_forecast_result/3d/Normal.csv")
skewstud_vix_3d.to_csv("Poutput/VIX_base_forecast_result/3d/skewstud.csv")
GED_vix_3d.to_csv("Poutput/VIX_base_forecast_result/3d/GED.csv")
studT_vix_3d.to_csv("Poutput/VIX_base_forecast_result/3d/studT.csv")

normal_vix_egarch_3d.to_csv("Poutput/VIX_base_forecast_result/3d/Normal_egarch.csv")
skewstud_vix_egarch_3d.to_csv("Poutput/VIX_base_forecast_result/3d/skewstud_egarch.csv")
GED_vix_egarch_3d.to_csv("Poutput/VIX_base_forecast_result/3d/GED_egarch.csv")
studT_vix_egarch_3d.to_csv("Poutput/VIX_base_forecast_result/3d/studT_egarch.csv")

normal_vix_5d.to_csv("Poutput/VIX_base_forecast_result/5d/Normal.csv")
skewstud_vix_5d.to_csv("Poutput/VIX_base_forecast_result/5d/skewstud.csv")
GED_vix_5d.to_csv("Poutput/VIX_base_forecast_result/5d/GED.csv")
studT_vix_5d.to_csv("Poutput/VIX_base_forecast_result/5d/studT.csv")

normal_vix_egarch_5d.to_csv("Poutput/VIX_base_forecast_result/5d/Normal_egarch.csv")
skewstud_vix_egarch_5d.to_csv("Poutput/VIX_base_forecast_result/5d/skewstud_egarch.csv")
GED_vix_egarch_5d.to_csv("Poutput/VIX_base_forecast_result/5d/GED_egarch.csv")
studT_vix_egarch_5d.to_csv("Poutput/VIX_base_forecast_result/5d/studT_egarch.csv")
