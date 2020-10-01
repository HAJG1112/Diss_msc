import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import arch
from arch.univariate.mean import HARX
from arch.univariate.volatility import HARCH, ARCH, GARCH, EGARCH
from arch.univariate import Normal, StudentsT, SkewStudent, GeneralizedError
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from diagnostics import get_breusch_pagan, get_engle_arch, get_durbin_watson, get_ljung_box
from stat_testing import adf_test, get_phillips_perron, get_prelim_stats

def get_harx_model_r(dependent, exogenous, lags, dist):
    '''
    This function takes a dependent variable(series/dataframe), the exogenous (dataframe), lags (list) and a distribution (arch distribution object)
    and fits a model according to the HAR specification.

    It sets the covariance estimate as the Bollerslev-Woodbridge Heteroscedastic consistent estimator

    It returns the errors of the fitted model

    It prints the summary of the model, preliminary statistcs of the residuals, and returns misspecification tests under the stat_testing.py module 
    '''
    har = HARX(y = dependent, x = exogenous, lags = lags, distribution = dist, constant = False, use_rotated=True) #HARCH(lags = lags) #, volatility = EGARCH(1,1)
    res_train = har.fit(options = {'maxiter': 1000,'ftol':1e-10, 'eps':1e-12}, cov_type='robust', disp = 'off')  #fit the data

    print("-------HAR MODEL" + str(lags) + str(exogenous) + " ---------")

    har_summ = res_train.summary()   #print summary
    errors = res_train.resid  #call the residual metod
    errors = errors.dropna()  #residuals are nan in bregiining for leng equal to lag
    
    print("Model summary", har_summ)
    print("\n")
    print("Analytics of redisual analysis", get_prelim_stats(errors))
    print("Misspecification of the process tests")
    print('\n')
    print(get_engle_arch(errors, 22))  #get ARCH-LM
    print(get_durbin_watson(errors,0))  #get DW test statistic
    print(get_ljung_box(errors, lags))  #get LjungBox test statistic
    
    return (errors)    #getting the model returns the errors from the model.

def get_harx_model(dependent, exogenous, lags, dist):
    '''
    This function takes a dependent variable(series/dataframe), the exogenous (dataframe), lags (list) and a distribution (arch distribution object)
    and fits a model according to the HAR specification.

    It sets the covariance estimate as the Bollerslev-Woodbridge Heteroscedastic consistent estimator

    It returns the errors of the fitted model

    It prints the summary of the model, preliminary statistcs of the residuals, and returns misspecification tests under the stat_testing.py module 
    '''
    har = HARX(y = dependent, x = exogenous, lags = lags, volatility = None, distribution = dist) #HARCH(lags = lags) #, volatility = EGARCH(1,1)
    res_train = har.fit(options = {'maxiter': 1000,'ftol':1e-10, 'eps':1e-12}, cov_type='robust', disp = 'off')  #fit the data

    print("-------HAR MODEL" + str(lags) + str(exogenous) + " ---------")

    har_summ = res_train.summary()   #print summary
    errors = res_train.resid  #call the residual metod
    errors = errors.dropna()  #residuals are nan in bregiining for leng equal to lag
    
    print("Model summary", har_summ)
    print("\n")
    print("Analytics of redisual analysis", get_prelim_stats(errors))
    print("Misspecification of the process tests")
    print('\n')
    print(get_engle_arch(errors, 22))  #get ARCH-LM
    print(get_durbin_watson(errors,0))  #get DW test statistic
    print(get_ljung_box(errors, lags))  #get LjungBox test statistic
    
    return (errors)    #getting the model returns the errors from the model.

vix_is = pd.read_csv(r"Pdata/vix_is.csv", index_col = 0).dropna()
vix_is.index = pd.to_datetime(vix_is.index)
vix = vix_is.iloc[:,0] #VIX

#individual exogenous
snp = pd.DataFrame(vix_is.iloc[:,1]).shift(1) #snp return
snp_pos_ret = pd.DataFrame(vix_is.iloc[:,2]).shift(1) #snp pos_ret
snp_neg_ret = pd.DataFrame(vix_is.iloc[:,3]).shift(1) #snp neg_ret
snp_pos_ret_5 = pd.DataFrame(vix_is.iloc[:,4]).shift(1) #snp pos_ret_5
snp_pos_ret_22 = pd.DataFrame(vix_is.iloc[:,5]).shift(1) #snp pos_ret_22
snp_neg_ret_5 = pd.DataFrame(vix_is.iloc[:,6]).shift(1) #snp neg_ret_5
snp_neg_ret_22 = pd.DataFrame(vix_is.iloc[:,7]).shift(1) #snp neg_ret_22
bsesn_dr = pd.DataFrame(vix_is.iloc[:,8]) # bsesn
bsesn_idr = pd.DataFrame(vix_is.iloc[:,9])
move = pd.DataFrame(vix_is.iloc[:,10]) #MOVE
l1_move = pd.DataFrame(vix_is.iloc[:,11]).rename(columns = {'MOVE': 'MOVE_l1'}) #lag of MOVE
vvix = pd.DataFrame(vix_is.iloc[:,16]) #vvix
vvix_l1 = pd.DataFrame(vix_is.iloc[:,17]).rename(columns = {'VVIX': 'VVIX_l1'})
skew = pd.DataFrame(vix_is.iloc[:,18]) 
skew_l1 = pd.DataFrame(vix_is.iloc[:,19]).rename(columns = {'skew': 'skew_l1'})
skew_change = pd.DataFrame(vix_is.iloc[:,20]).rename(columns = {'skew_l1': 'skew_diff'})

#variable grouping
leverage = pd.concat([snp_neg_ret, snp_neg_ret_5, snp_neg_ret_22], axis = 1)
l1 = pd.concat([snp_neg_ret, snp_neg_ret_5], axis = 1)
#volatility group
vol = pd.concat([vvix, move, skew], axis =1)
vol_ = pd.concat([vvix_l1, skew_l1], axis =1)

dist1 = arch.univariate.Normal()  #set parameter values for nu and lambda
dist2 = arch.univariate.StudentsT()
dist3 = arch.univariate.GeneralizedError()
dist4 = arch.univariate.SkewStudent()


a1 = get_harx_model(vix, snp, [1,5,22], dist2)
a2 = get_harx_model(vix, leverage, [1,5,22], dist2)
a3 = get_harx_model(vix, l1, [1,5,22], dist2)
a4 = get_harx_model(vix, bsesn_dr, [1,5,22], dist2)
a5 = get_harx_model(vix, bsesn_idr, [1,5,22], dist2)
a6 = get_harx_model(vix, l1_move, [1,5,22], dist2)
a7 = get_harx_model(vix, vvix_l1, [1,5,22], dist2)
a8 = get_harx_model(vix, skew_l1, [1,5,22], dist2)
a9 = get_harx_model(vix, skew_change, [1,5,22], dist2)

#a1################
c
