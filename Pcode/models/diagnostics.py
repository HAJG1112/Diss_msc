import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import het_breuschpagan, het_arch, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson

def get_engle_arch(errors, nlags):

    '''
    An uncorrelated time series can still be serially dependent due to a dynamic conditional variance process. 
    A time series exhibiting conditional heteroscedasticity—or autocorrelation in the squared series—is 
    said to have autoregressive conditional heteroscedastic (ARCH) effects. 
    Engle’s ARCH test is a Lagrange multiplier test to assess the significance of ARCH effects Rober, Engle (1982).
    
    H0: process is not autocorrelated in the squared residuals
    H1: process is autocorrelated in the squared residuals

    Large critical value of the F-statistic proves rejection of the null
    '''
    arch_lm = het_arch(errors.dropna(), nlags=nlags)
    print("Engle's Test for Autoregressive conditional Heteroscedasticity")
    print("Number of Lags: {}".format(nlags))
    print('LM test-stat:{}'.format(arch_lm[0]))
    print('P-val for LM:{}'.format(arch_lm[1]))
    print('F-statistic:{}'.format(arch_lm[2]))
    print('P-value for F-stat:{}'.format(arch_lm[3]))

def get_breusch_pagan(errors, exog):
    bp = het_breuschpagan(errors.dropna(), exog.dropna())
    print("Breusch-Pagan LM test for Heteroscedasticity")
    print('LM test-stat:{}'.format(bp[0]))
    print('P-val for LM:{}'.format(bp[1]))
    print('F-statistic:{}'.format(bp[2]))
    print('P-value for F-stat:{}'.format(bp[3]))

def get_durbin_watson(errors, axis): #must feed 1-d array
    '''
    A number which determines whether there is autocorrelation in the residuals of a time series regression. 
    The statistic ranges from 0 to 4 with 0 indicating positive autocorrelation and 4 indicating negative correlation. 
    A value of 2 indicates no auto correlation in the sample. The formula is expressed as:
    d=(sum from t=2 to t=T of: (et-et-1)2/(sum from t=1 to t=T of: et2)

    where the series of et are the residuals from a regression.

    Read more: http://www.businessdictionary.com/definition/Durbin-Watson-Statistic.html
    '''
    db = durbin_watson(errors.dropna(), axis)
    print('Durbin-Watson test statistic:{}'.format(db))

def get_ljung_box(errors, lags):
    '''
    Alternative to ARCH test, checks for serial dependence
    H0: The residuals are independently distributed.
    H1: The residuals are not independently distributed; they exhibit serial correlation.
    '''
    print ('Results of Ljung-Box test for autocorrelation')
    lb = acorr_ljungbox(errors, lags = lags , return_df = True)
    print("Ljung-Box test for Autocorrelation in residuals")
    return lb
    