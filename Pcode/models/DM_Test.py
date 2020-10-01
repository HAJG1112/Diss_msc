
def dm_test(actual_lst, pred1_lst, pred2_lst, h, crit="MSE", power = 2):

    # Routine for checking errors
    def error_check():
        rt = 0
        msg = ""
        # Check if h is an integer
        if (not isinstance(h, int)):
            rt = -1
            msg = "The type of the number of steps ahead (h) is not an integer."
            return (rt,msg)
        # Check the range of h
        if (h < 1):
            rt = -1
            msg = "The number of steps ahead (h) is not large enough."
            return (rt,msg)
        len_act = len(actual_lst)
        len_p1  = len(pred1_lst)
        len_p2  = len(pred2_lst)
        # Check if lengths of actual values and predicted values are equal
        if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
            rt = -1
            msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
            return (rt,msg)
        # Check range of h
        if (h >= len_act):
            rt = -1
            msg = "The number of steps ahead is too large."
            return (rt,msg)
        # Check if criterion supported
        if (crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly"):
            rt = -1
            msg = "The criterion is not supported."
            return (rt,msg)  
        # Check if every value of the input lists are numerical values
        from re import compile as re_compile
        comp = re_compile("^\d+?\.\d+?$")  
        def compiled_regex(s):
            """ Returns True is string is a number. """
            if comp.match(s) is None:
                return s.isdigit()
            return True
        for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
            is_actual_ok = compiled_regex(str(abs(actual)))
            is_pred1_ok = compiled_regex(str(abs(pred1)))
            is_pred2_ok = compiled_regex(str(abs(pred2)))
            if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):  
                msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                rt = -1
                return (rt,msg)
        return (rt,msg)
    
    # Error check
    error_code = error_check()
    # Raise error if cannot pass error check
    if (error_code[0] == -1):
        raise SyntaxError(error_code[1])
        return
    # Import libraries
    from scipy.stats import t
    import collections
    import pandas as pd
    import numpy as np
    
    # Initialise lists
    e1_lst = []
    e2_lst = []
    d_lst  = []
    
    # convert every value of the lists into real values
    actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
    pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
    pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()
    
    # Length of lists (as real numbers)
    T = float(len(actual_lst))
    
    # construct d according to crit
    if (crit == "MSE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append((actual - p1)**2)
            e2_lst.append((actual - p2)**2)
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAD"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs(actual - p1))
            e2_lst.append(abs(actual - p2))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "MAPE"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(abs((actual - p1)/actual))
            e2_lst.append(abs((actual - p2)/actual))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)
    elif (crit == "poly"):
        for actual,p1,p2 in zip(actual_lst,pred1_lst,pred2_lst):
            e1_lst.append(((actual - p1))**(power))
            e2_lst.append(((actual - p2))**(power))
        for e1, e2 in zip(e1_lst, e2_lst):
            d_lst.append(e1 - e2)    
    
    # Mean of d        
    mean_d = pd.Series(d_lst).mean()
    
    # Find autocovariance and construct DM test statistics
    def autocovariance(Xi, N, k, Xs):
        autoCov = 0
        T = float(N)
        for i in np.arange(0, N-k):
              autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
        return (1/(T))*autoCov
    gamma = []
    for lag in range(0,h):
        gamma.append(autocovariance(d_lst,len(d_lst),lag,mean_d)) # 0, 1, 2
    V_d = (gamma[0] + 2*sum(gamma[1:]))/T
    DM_stat=V_d**(-0.5)*mean_d
    harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
    DM_stat = harvey_adj*DM_stat
    # Find p-value
    p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
    
    # Construct named tuple for return
    dm_return = collections.namedtuple('dm_return', 'DM p_value')
    
    rt = dm_return(DM = DM_stat, p_value = p_value)
    
    return rt

import pandas as pd
import numpy as np

#1d forecat base models
model1 = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\Normal.csv", index_col = 0).dropna()
model2 = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\GED.csv", index_col = 0).dropna()
model3 = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\skewstud.csv", index_col = 0).dropna()
model4 = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\studT.csv", index_col = 0).dropna()
model1e = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\Normal_egarch.csv", index_col = 0).dropna()
model2e = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\GED_egarch.csv", index_col = 0).dropna()
model3e = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\skewstud_egarch.csv", index_col = 0).dropna()
model4e = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\studT_egarch.csv", index_col = 0).dropna()

#Model with no volatility process estimated
M1_n = pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M1_n.csv", index_col=0).dropna()
M1_st = pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M1_st.csv", index_col=0).dropna()
M1_ged = pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M1_ged.csv", index_col=0).dropna()
M2_n= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M2_n.csv", index_col=0).dropna()
M2_st= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M2_st.csv", index_col=0).dropna()
M2_ged= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M2_ged.csv", index_col=0).dropna()
M2a_n= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M2a_n.csv", index_col=0).dropna()
M2a_st= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M2a_st.csv", index_col=0).dropna()
M2a_ged= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M2a_ged.csv", index_col=0).dropna()
M3_n= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M3_n.csv", index_col=0).dropna()
M3_st= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M3_st.csv", index_col=0).dropna()
M3_ged = pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M3_ged.csv", index_col=0).dropna()
M4_n= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M4_n.csv", index_col=0).dropna()
M4_st= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M4_st.csv", index_col=0).dropna()
M4_ged= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M4_ged.csv", index_col=0).dropna()
M5_ged= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M5_ged.csv", index_col=0).dropna()
M6_ged= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M6_ged.csv", index_col=0).dropna()
M7_ged= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M7_ged.csv", index_col=0).dropna()
M8_st= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M8_st.csv", index_col=0).dropna()
M8_n= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M8_n.csv", index_col=0).dropna()
M8_ged= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M8_ged.csv", index_col=0).dropna()
M9_st= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M9_st.csv", index_col=0).dropna()
M9_n= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M9_n.csv", index_col=0).dropna()
#Model with EGARCH(1,1)
M1_n_e = pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M1_n.csv", index_col=0).dropna()
M1_st_e = pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M1_st.csv", index_col=0).dropna()
M1_ged_e = pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M1_ged.csv", index_col=0).dropna()
M2_n_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M2_n.csv", index_col=0).dropna()
M2_st_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M2_st.csv", index_col=0).dropna()
M2_ged_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M2_ged.csv", index_col=0).dropna()
M2a_n_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M2a_n.csv", index_col=0).dropna()
M2a_st_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M2a_st.csv", index_col=0).dropna()
M2a_ged_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M2a_ged.csv", index_col=0).dropna()
M3_n_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M3_n.csv", index_col=0).dropna()
M3_st_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M3_st.csv", index_col=0).dropna()
M3_ged_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M3_ged.csv", index_col=0).dropna()
M4_n_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M4_n.csv", index_col=0).dropna()
M4_st_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M4_st.csv", index_col=0).dropna()
M4_ged_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M4_ged.csv", index_col=0).dropna()
M5_ged_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M5_ged.csv", index_col=0).dropna()
M6_ged_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M6_ged.csv", index_col=0).dropna()
M7_ged_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M7_ged.csv", index_col=0).dropna()
M8_st_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M8_st.csv", index_col=0).dropna()
M8_n_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M8_n.csv", index_col=0).dropna()
M8_ged_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M8_ged.csv", index_col=0).dropna()
M9_st_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M9_st.csv", index_col=0).dropna()
M9_n_e= pd.read_csv("Poutput/VIX_X_forecast_result/egarch/1d/M9_n.csv", index_col=0).dropna()


model1_3d = pd.read_csv(r"Poutput\VIX_base_forecast_result\3d\Normal.csv", index_col = 0).dropna()
model2_3d = pd.read_csv(r"Poutput\VIX_base_forecast_result\3d\GED.csv", index_col = 0).dropna()
model3_3d = pd.read_csv(r"Poutput\VIX_base_forecast_result\3d\skewstud.csv", index_col = 0).dropna()
model4_3d = pd.read_csv(r"Poutput\VIX_base_forecast_result\3d\studT.csv", index_col = 0).dropna()
model1_3de = pd.read_csv(r"Poutput\VIX_base_forecast_result\3d\Normal_egarch.csv", index_col = 0).dropna()
model2_3de= pd.read_csv(r"Poutput\VIX_base_forecast_result\3d\GED_egarch.csv", index_col = 0).dropna()
model3_3de = pd.read_csv(r"Poutput\VIX_base_forecast_result\3d\skewstud_egarch.csv", index_col = 0).dropna()
model4_3de = pd.read_csv(r"Poutput\VIX_base_forecast_result\3d\studT_egarch.csv", index_col = 0).dropna()

model1_5d = pd.read_csv(r"Poutput\VIX_base_forecast_result\5d\Normal.csv", index_col = 0).dropna()
model2_5d = pd.read_csv(r"Poutput\VIX_base_forecast_result\5d\GED.csv", index_col = 0).dropna()
model3_5d = pd.read_csv(r"Poutput\VIX_base_forecast_result\5d\skewstud.csv", index_col = 0).dropna()
model4_5d = pd.read_csv(r"Poutput\VIX_base_forecast_result\5d\studT.csv", index_col = 0).dropna()

model1_5de = pd.read_csv(r"Poutput\VIX_base_forecast_result\5d\Normal_egarch.csv", index_col = 0).dropna()
model2_5de = pd.read_csv(r"Poutput\VIX_base_forecast_result\5d\GED_egarch.csv", index_col = 0).dropna()
model3_5de = pd.read_csv(r"Poutput\VIX_base_forecast_result\5d\skewstud_egarch.csv", index_col = 0).dropna()
model4_5de = pd.read_csv(r"Poutput\VIX_base_forecast_result\5d\studT_egarch.csv", index_col = 0).dropna()


exog_norm= [M1_n, M2_n,  M3_n, M4_n, M8_n, M9_n]
exog_st = [M1_st, M2_st,  M3_st, M4_st, M8_st, M9_st]
exog_ged = [M1_ged, M2_ged, M3_ged, M4_ged ,M5_ged, M6_ged, M7_ged, M8_ged]

exog_norm_e= [M1_n_e, M2_n_e,  M3_n_e, M4_n_e, M8_n_e, M9_n_e]
exog_st_e = [M1_st_e, M2_st_e, M3_st_e, M8_st_e]
exog_ged_e = [M1_ged_e, M2_ged_e,  M3_ged_e, M4_ged_e ,M5_ged_e, M6_ged_e, M7_ged_e, M8_ged_e]

#M3_ged and M3_ged_e isnt working???? for th1-day, not sure why, its important that it does

all_1d = [model1, model2, model3, model4, model1e, model2e, model3e, model4e, M1_n, M2_n, \
    M3_n, M4_n, M8_n, M9_n, M1_n_e, M2_n_e,  M3_n_e, M4_n_e, M8_n_e, M9_n_e, \
        M1_st, M2_st,  M3_st, M4_st, M8_st, \
           M1_st_e, M2_st_e, M3_st_e, M8_st_e,M1_ged, M2_ged, M3_ged,  M4_ged, M5_ged, M6_ged, M7_ged, M8_ged,  \
               M1_ged_e, M2_ged_e, M3_ged_e, M4_ged_e ,M5_ged_e, M6_ged_e, M7_ged_e, M8_ged_e]

all_3d = [model1_3d, model2_3d, model3_3d, model4_3d, model1_3de, model2_3de]
all_5d = [model1_5d, model2_5d, model3_5d, model4_5d, model1_5de, model2_5de]


actual1d = model1.iloc[:,1]
actual3d = model1_3d.iloc[:,1]
actual5d = model1_5d.iloc[:,1]

def get_dm_test(x, actual):
    count = 0
    df = pd.DataFrame(np.zeros((len(x), len(x))))
    for i in range(len(x)):
        for j in range(len(x)):
            count +=1
            print(count)
            if i != j:
                df.iloc[i,j] = dm_test(actual, x[i].iloc[:,0], x[j].iloc[:,0], 1,'MAD')[0]
            else:
                df.iloc[i,j] = 1
    return df

def get_dm_test3(x, actual):
    df = pd.DataFrame(np.zeros((len(x), len(x))))
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                df.iloc[i,j] = dm_test(actual, x[i].iloc[:,0], x[j].iloc[:,0], 1,'MAD')[0]
            else:
                df.iloc[i,j] = 1
    return df

def get_dm_test5(x, actual):
    df = pd.DataFrame(np.zeros((len(x), len(x))))
    for i in range(len(x)):
        for j in range(len(x)):
            if i != j:
                df.iloc[i,j] = dm_test(actual, x[i].iloc[:,0], x[j].iloc[:,0], 1, 'MAD')[0]
            else:
                df.iloc[i,j] = 1
    return df


oned = get_dm_test(all_1d, actual1d)
threed = get_dm_test3(all_3d, actual3d)
fived = get_dm_test5(all_5d, actual5d)
oned.to_csv("Poutput/DM/1d/DM_1d_mad.csv")
threed.to_csv("Poutput/DM/3d/DM_3d_mad.csv")
fived.to_csv("Poutput/DM/5d/DM_5d_mad.csv")

