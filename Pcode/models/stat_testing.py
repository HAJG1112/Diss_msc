import pandas as pd
import warnings
warnings.simplefilter('ignore')
import arch
import matplotlib.pyplot as plt
import statsmodels 
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import statistics
import scipy.stats as stats
from scipy.stats import kurtosis, skew, jarque_bera
import seaborn as sns

def adf_test(timeseries, trend):
    from arch.unitroot import ADF
    adf = ADF(timeseries)
    adf.trend = str(trend)
    reg_res = adf.regression
    #print('ADF statistic: {0:0.4f}'.format(adf.stat))
    #print('ADF p-value: {0:0.4f}'.format(adf.pvalue))
    #print(reg_res.summary().as_text())
    return (adf.stat, adf.pvalue)

def get_phillips_perron(timeseries):
    from arch.unitroot import PhillipsPerron
    pp = PhillipsPerron(timeseries)
    print(pp.summary().as_text())

def get_prelim_stats(timeseries):
    '''
    this prints returns the standard dev, median, skewness, kurtosis, Jarque-bera, and the two unit root tests
    input must be a slice of a dataframe or a series
    '''
    print(timeseries.describe())
    print('std_dev:{}'.format(statistics.stdev(timeseries)))
    print('median:', statistics.median(timeseries))
    print('skewness:{}'.format(skew(timeseries)))
    print('kurtosis:{}'.format(kurtosis(timeseries)))
    print('JB test stat:{}'.format(jarque_bera(timeseries)))
    print('\n')
    print('Unit Root Tests on {}'.format(timeseries.name))
    adf_test(timeseries, 'n')
    get_phillips_perron(timeseries)



