import pandas as pd
import numpy as np
from arch.univariate import HARX
import arch
from forecast_tests import rmse, mse, directional_accuracy, test_forecast, binary_accuracy, ql
import statsmodels.api as sm
import scipy.stats as stats
import seaborn as sns
from test_dist import SkewStudent_Haisun
import matplotlib.pyplot as plt

# rolling 1 step ahead forecasts from the final observation
def get_har_forecast(dependent, lags, dist):

    import sys
    har_test = HARX(y = dependent, lags = lags, distribution = dist, rescale=True)
    index = dependent.index
    end_loc = np.where(index >= '2017-01-03')[0].min() #returns where index is greater than a date.
    forecasts = {}

    for i in range(890):  #878 values between the start and end of the OS dataset.
        sys.stdout.write('.')
        sys.stdout.flush()
        res = har_test.fit(first_obs=i, last_obs = i + end_loc , disp='off', options = {'maxiter': 10000000, 'eps' : 0.01, 'ftol': 0.0001}) 
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html
        temp = res.forecast(horizon=1).mean
        fcast = temp.iloc[i + end_loc - 1]
        forecasts[fcast.name] = fcast
    har_forecast = (pd.DataFrame(forecasts).T)
    return(har_forecast) 


def get_har_results(dependent, lags, dist, _open):
    y_hat = get_har_forecast(dependent, lags, dist)
    y = dependent.loc['2017-01-03':]
    y_open = _open.loc['2017-01-03':]
    y = y.dropna()

    compare = pd.concat([y_hat, y, y_open], axis=1)
    compare = compare.rename(columns = {'h.1': 'Forecast'})

    # error formula's
    compare['close_error'] = compare['Forecast'] - compare['Close']
    compare['close_rmse'] = rmse(compare['Forecast'],compare['Close'])
    compare['close_mse'] = mse(compare['Forecast'],compare['Close'])
    compare['close_ql'] = ql(compare['Forecast'], compare['Close'])

    #daily return forecast and directional accuracy
    compare['forecast_DR'] = np.log(compare['Forecast']/compare['Close'].shift(1))
    compare['actual_DR'] = np.log(compare['Close']/compare['Close'].shift(1))
    compare['DA_DR'] = np.where(np.sign(compare['forecast_DR'])==np.sign(compare['actual_DR']),1,0)

    #IDR forecast and directional accuracy, assumed we trade at open the next day
    compare['forecast_IDR'] = np.log(compare['Forecast']/compare['Open'])
    compare['actual_IDR'] = np.log(compare['Close']/compare['Open'])
    compare['DA_IDR'] = np.where(np.sign(compare['forecast_IDR'])==np.sign(compare['actual_IDR']),1,0)

    return(compare.dropna())

vxc = pd.read_csv(r"Pdata\vxc.csv", parse_dates = True, index_col = 'Date')

#vix = pd.read_csv(r"Pdata\vix.csv", parse_dates = True, index_col = 'Date')

# Distribution
dist = arch.univariate.Normal(random_state=None)  #set parameter values for nu and lambda
dist1 = arch.univariate.StudentsT()
dist2 = arch.univariate.SkewStudent() #standardized Skewed student
dist3 = arch.univariate.GeneralizedError()  #generalized error distribution
dist4 = SkewStudent_Haisun(random_state=None)  #set parameter values for nu and lambda, has error in the final week, invalid log value?


## these are the 4 basic HAR models for the VIX identified in the estimation phase
skewstud = get_har_results(vxc['Close'], [1, 5, 22], dist4, vxc['Open'])

print('-------------Results-----------------')
print(skewstud)
print((skewstud.iloc[:,3:].mean()))
skewstud.to_csv(r"Poutput\VXF_base\HAR_f1_skewstudent[8,0.4].csv")
#interpolate missing open, high and low data

#redo the continuous VIX data!!!! its datetime+interpolation of missing values is shit!



#Forecast
#https://arch.readthedocs.io/en/latest/univariate/generated/generated/arch.univariate.HARX.forecast.html
# #arch.univariate.HARX.forecast
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html optimization algorithm sub-routine