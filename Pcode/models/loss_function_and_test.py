import numpy as np
from diagnostics import get_breusch_pagan, get_engle_arch, get_durbin_watson, get_ljung_box

def mse(prediction, targets):
    mse = ((targets - prediction) ** 2)
    return(mse)

def ql(predictions, targets):
    ql = (predictions/targets) - np.log(predictions/targets) -1
    return (ql)

def test_forecast(actual, prediction, error, lags):
    print(get_engle_arch(error, 22))  #get ARCH-LM
    print(get_durbin_watson(error,0))  #get DW test statistic
    print(get_ljung_box(error, lags)) 
    return

def binary_accuracy(prediction, y_value):   #tests for the correct sign of the forecast
    a = sum(np.sign(prediction)==np.sign(y_value))
    b = len(prediction)
    accuracy = a/b
    print("sign accuracy: {}".format(accuracy))

def directional_accuracy(prediction, actual):
    a = np.where(np.sign(prediction)==np.sign(actual),1,0)
    return a
# https://arch.readthedocs.io/en/latest/multiple-comparison/generated/arch.bootstrap.SPA.html?highlight=spa