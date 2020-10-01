import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import arch
import arch.univariate as au
from arch.univariate.mean import HARX
from arch.univariate.volatility import HARCH, ARCH
from arch.univariate import Normal, StudentsT, SkewStudent, GeneralizedError
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from diagnostics import get_breusch_pagan, get_engle_arch, get_durbin_watson, get_ljung_box
from stat_testing import adf_test, get_phillips_perron, get_prelim_stats

from test_dist import SkewStudent_Haisun

vxc = pd.read_csv(r"Pdata\vcx1.csv", parse_dates = True, index_col = 0)
vxc.index = pd.to_datetime(vxc.index, format = "%Y - %m - %d")

vix = pd.read_csv(r"Pdata\vix.csv", parse_dates = True, index_col = 0)
vix.index = pd.to_datetime(vix.index, format = "%Y - %m - %d")

vxc_IS = vxc.loc['2010-01-01':'2017-01-01']
vxc_OS = vxc.loc['2017-01-02':]

vix_IS = vix.loc['2010-01-01':'2017-01-01']
vix_OS = vix.loc['2017-01-02':]

vxc_c = vxc_IS['Close'].dropna()
vxc_DR = vxc_IS['Return'].dropna()

vix_c = vix_IS['VIX Close'].dropna()
vix_DR = vix_IS['VIX_DR'].dropna()


def get_har_model(dependent, lags):

    dist = SkewStudent_Haisun(random_state=None)  #set parameter values for nu and lambda

    har = HARX(y = dependent, lags = lags, distribution = dist)
    res_train = har.fit()  #fit the data

    print("-------HAR MODEL" + str(lags) + " Skewed Student [eta = 10, lam = 0.4] ---------")
    har_summ = res_train.summary()   #print summary
    errors = res_train.resid  #call the residual metod
    errors = errors.dropna()  #residuals are nan in bregiining for leng equal to lag
    
    print("Model summary",har_summ)
    print("\n")
    print("Misspecification of the process tests")
    print('\n')
    print("Analytics of redisual analysis", get_prelim_stats(errors))
    print(get_engle_arch(errors, 22))  #get ARCH-LM
    print(get_durbin_watson(errors,0))  #get DW test statistic
    print(get_ljung_box(errors, lags))  #get LjungBox test statistic
    
    
    return (errors)    #getting the model returns the errors from the model.

x = get_har_model(vix_c, [1,5,22])
print(x)


#4 plots needed, residual histogram, qq, ACF and Square resid ACF
sns.set_style('dark')
sns.distplot(x, label = 'Residuals',
                hist = True,
                kde=True,
                fit=stats.norm,  # A Student’s t continuous random variable.
                kde_kws = {'color': '#fc4f30', 'label': 'KDE'},
                hist_kws = {'alpha': 0.25, 'label': 'Returns'},
                fit_kws = {'color': '#e5ae38', 'label': 'Normal', 'alpha' : 0.75})
plt.legend()
plt.savefig('Pgraphs/HAR-SkewedStudent[10,0.4] hist.png')

plot_acf(x, title = 'ACF of the Close Residuals')
plt.savefig('Pgraphs/ACF-HAR-SkewedStudent[10,0.4] error.png')

plot_acf(x**2,  title = 'ACF of the Close Squared Residuals')
plt.savefig('Pgraphs/ACF-HAR-SkewedStudent[10,0.4].png')

sm.qqplot(x , line ='45') 
plt.savefig('Pgraphs/QQ-HAR-SkewedStudent[10,0.4].png')

