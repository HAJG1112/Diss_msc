import pandas as pd
import numpy as np
from arch.univariate import HARX
import arch
from loss_function_and_test import mse, test_forecast, binary_accuracy, ql
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import seaborn as sns

vix = pd.read_csv(r"Pdata\vix_os.csv", index_col = 0)
vix.index = pd.to_datetime(vix.index)


# rolling 1 step ahead forecasts from the final observation
def get_har_forecast(dependent, lags):

    dist = arch.univariate.Normal(random_state=None)  #set parameter values for nu and lambda
    dist1 = arch.univariate.StudentsT()
    dist2 = arch.univariate.SkewStudent() #standardized Skewed student
    dist3  =arch.univariate.GeneralizedError()  #generalized error distribution

    har_test = HARX(y = dependent, lags = lags, distribution = dist)
    
    import sys
    index = dependent.index
    end_loc = np.where(index >= '2017-01-07')[0].min() #returns where index is greater than a date.
    forecasts = {}

    for i in range(500):  #900 takes us to 2019-11-27 #1044 is the perfect number!
        sys.stdout.write('.')
        sys.stdout.flush()
        res = har_test.fit(first_obs=i, last_obs=i + end_loc, disp='off')
        temp = res.forecast(horizon=1, align='target').mean
        fcast = temp.iloc[i + end_loc - 1]
        forecasts[fcast.name] = fcast

    print()
    har_forecast = (pd.DataFrame(forecasts).T)
    final = pd.concat([har_forecast, dependent], axis=1)
    final = final.dropna()  #dropna as concatenating dataframes of 2 different sizes
    final['FE:' + str(dependent.name)] = final.iloc[:,0] -  final.iloc[:,1]
    return(final) 

def get_forecast_tests(prediction, actual, error, lags):
    try:
        mse = mse(prediction, actual)
        ql = ql(prediction, actual)
        test_forecast(actual, prediction, error, lags)
        binary_accuracy(prediction, actual)
    except (AttributeError,TypeError)as e:
        print("Error has occured: ", e)
    return

def forecast():
    try:
        x = get_har_forecast(vix['VIX Close'].dropna(),[1, 5, 22])
        actual = vix['VIX Close'].dropna()
        forecast = pd.concat([x, actual], axis = 1)
        forecast = forecast.dropna()
        forecast['error'] = forecast.iloc[:,0] - forecast.iloc[:,1]
    except (AttributeError,TypeError)as e:
        print("Error has occured: ", e)
    return (forecast)  #returns data frame object of otcr forecast, actual and the error

#y = forecast()
#get_forecast_tests(y.iloc[:,0], y.iloc[:,1], y.iloc[:,2], 22)
print(get_har_forecast(vix, [1, 5, 22]))

#print(vix)
#fig1 = sns.lineplot(data = y)  #plots predicted, actual, and error
#plt.show()




#Forecast
#https://arch.readthedocs.io/en/latest/univariate/generated/generated/arch.univariate.HARX.forecast.html#arch.univariate.HARX.forecast