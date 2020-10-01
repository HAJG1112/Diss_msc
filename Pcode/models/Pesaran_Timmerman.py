import pandas as pd
import numpy as np

model1 = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\Normal.csv", index_col = 0).dropna()
model2 = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\GED.csv", index_col = 0).dropna()
model3 = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\skewstud.csv", index_col = 0).dropna()
model4 = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\studT.csv", index_col = 0).dropna()

model1e = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\Normal_egarch.csv", index_col = 0).dropna()
model2e = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\GED_egarch.csv", index_col = 0).dropna()
model3e = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\skewstud_egarch.csv", index_col = 0).dropna()
model4e = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\studT_egarch.csv", index_col = 0).dropna()

model1_3d = pd.read_csv(r"Poutput\VIX_base_forecast_result\3d\Normal.csv", index_col = 0).dropna()
model2_3d = pd.read_csv(r"Poutput\VIX_base_forecast_result\3d\GED.csv", index_col = 0).dropna()
model3_3d = pd.read_csv(r"Poutput\VIX_base_forecast_result\3d\skewstud.csv", index_col = 0).dropna()
model4_3d = pd.read_csv(r"Poutput\VIX_base_forecast_result\3d\studT.csv", index_col = 0).dropna()

model1_5d = pd.read_csv(r"Poutput\VIX_base_forecast_result\5d\Normal.csv", index_col = 0).dropna()
model2_5d = pd.read_csv(r"Poutput\VIX_base_forecast_result\5d\GED.csv", index_col = 0).dropna()
model3_5d = pd.read_csv(r"Poutput\VIX_base_forecast_result\5d\skewstud.csv", index_col = 0).dropna()
model4_5d = pd.read_csv(r"Poutput\VIX_base_forecast_result\5d\studT.csv", index_col = 0).dropna()

model1_3de = pd.read_csv(r"Poutput\VIX_base_forecast_result\3d\Normal_egarch.csv", index_col = 0).dropna()
model2_3de= pd.read_csv(r"Poutput\VIX_base_forecast_result\3d\GED_egarch.csv", index_col = 0).dropna()
model3_3de = pd.read_csv(r"Poutput\VIX_base_forecast_result\3d\skewstud_egarch.csv", index_col = 0).dropna()
model4_3de = pd.read_csv(r"Poutput\VIX_base_forecast_result\3d\studT_egarch.csv", index_col = 0).dropna()

model1_5de = pd.read_csv(r"Poutput\VIX_base_forecast_result\5d\Normal_egarch.csv", index_col = 0).dropna()
model2_5de = pd.read_csv(r"Poutput\VIX_base_forecast_result\5d\GED_egarch.csv", index_col = 0).dropna()
model3_5de = pd.read_csv(r"Poutput\VIX_base_forecast_result\5d\skewstud_egarch.csv", index_col = 0).dropna()
model4_5de = pd.read_csv(r"Poutput\VIX_base_forecast_result\5d\studT_egarch.csv", index_col = 0).dropna()

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
M3_ged= pd.read_csv("Poutput/VIX_X_forecast_result/base/1d/M3_ged.csv", index_col=0).dropna()
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

def PT_test(forecast_return, actual_return):
    Npp = np.where(np.sign(forecast_return) + np.sign(actual_return) == 2, 1,0).sum() #add signs
    Nnp = np.where(np.sign(forecast_return) < np.sign(actual_return), 1,0).sum()
    Npn = np.where(np.sign(forecast_return) > np.sign(actual_return), 1,0).sum()
    Nnn = np.where(np.sign(forecast_return) + np.sign(actual_return) == -2, 1,0).sum()
    T = len(forecast_return)
    x = (Npp + Nnp)/T #returns float
    y = (Npp + Npn)/T #returns float
    numer = (Npp/(Npp + Nnp)) - (Npn/(Npn + Nnn))
    denom = np.power((x*(1 - x)) / (y*(1-y)), 0.5)
    PT = ((np.sqrt(T)*numer)/denom)
    #print('Npp: {}, Nnp: {}, Npn: {}, Nnn {}'.format(Npp, Nnp, Npn, Nnn))
    #return(PT)
    tot = Npp+Npn+Nnp+Nnn
    return(Npp, Npn, Nnp, Nnn, tot, PT)

def get_PTS(x):
    fdr = x.iloc[:,5]
    adr = x.iloc[:,6]
    fidr = x.iloc[:,8]
    aidr = x.iloc[:,9]
    PTdr = PT_test(fdr, adr)
    PTidr = PT_test(fidr, aidr)
    return (PTdr, PTidr) #, PTidr

def get_PTS_models(x):
    for i in range(len(x)):
        y = get_PTS(x[i])
        print (y)

np.seterr(divide='ignore')

exog_normal= [M1_n, M2_n, M3_n, M4_n, M8_n, M9_n]
exog_st = [M1_st, M2_st, M3_st, M4_st, M8_st, M9_st]
exog_ged = [M1_ged, M2_ged,M3_ged, M4_ged ,M5_ged, M6_ged, M7_ged,M8_ged]
exog_normal_e= [M1_n_e, M2_n_e, M3_n_e, M4_n_e, M8_n_e, M9_n_e]
exog_st_e = [M1_st_e, M2_st_e, M3_st_e, M4_st_e, M8_st_e, M9_st_e]
exog_ged_e = [M1_ged_e, M2_ged_e, M3_ged_e, M4_ged_e ,M5_ged_e, M6_ged_e, M7_ged_e,M8_ged_e]

base_models_1d = [model1, model2, model3, model4]
base_models_3d = [model1_3d, model3_3d, model3_3d, model4_3d]
base_models_5d = [model1_5d, model3_5d, model3_5d, model4_5d]
base_models_1de = [model1e, model2e, model3e, model4e]
base_models_3de = [model1_3de, model2_3de]
base_models_5de = [model1_5de, model2_5de]

#Tuple on left is for the daily return, tuple on right is for the intra-day return'

print('base models 1d')
get_PTS_models(base_models_1d)
print('base models 1d egarch')
get_PTS_models(base_models_1de)
print('base models 3d')
get_PTS_models(base_models_3d)
print('base models 3d egarch')
get_PTS_models(base_models_3de)
print('base models 5d')
get_PTS_models(base_models_5d)
print('base models 5de')
get_PTS_models(base_models_5de)
print('exog under normal')
get_PTS_models(exog_normal)
print('exog under normal egarch')
get_PTS_models(exog_normal_e)
print('exog student')
get_PTS_models(exog_st)
print('exog student e')
get_PTS_models(exog_st_e)
print('exog ged')
get_PTS_models(exog_ged)
print('exog ged e')
get_PTS_models(exog_ged_e)

