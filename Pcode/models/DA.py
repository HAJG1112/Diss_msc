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

def get_mean_da(x):
    mean_da_dr = x.iloc[:,7].mean()
    mean_da_idr = x.iloc[:,10].mean()
    return mean_da_dr, mean_da_idr

def get_da_result(x):
    for i in range(len(x)):
        mean_da_dr = get_mean_da(x[i])[0]
        mean_da_idr = get_mean_da(x[i])[1]
        return(mean_da_dr, mean_da_idr)

def get_mean_losses(x):
    mean_ae = x.iloc[:,2].mean()
    mean_rmse = np.sqrt(x.iloc[:,3].mean())
    mean_mse = x.iloc[:,3].mean()
    sum_ql = x.iloc[:,4].sum()
    err = x.iloc[:,-1].mean()
    return(mean_ae, mean_rmse, mean_mse, sum_ql, err)

def get_result_losses(x):
    da_dr = []
    da_idr = []
    ae = []
    mse = []
    ql = []
    err = []
    for i in range(len(x)):
        da_DR = get_mean_da(x[i])[0]
        da_IDR = get_mean_da(x[i])[1]
        mean_ae = get_mean_losses(x[i])[0]
        mean_rmse = get_mean_losses(x[i])[1]
        mean_mse = get_mean_losses(x[i])[2]
        sum_ql = get_mean_losses(x[i])[3]
        errr = get_mean_losses(x[i])[4]
        da_dr.append(da_DR)
        da_idr.append(da_IDR)
        ae.append(mean_ae)
        mse.append(mean_mse)
        ql.append(sum_ql)
        err.append(errr)
        
    df = pd.DataFrame()
    df['DA_dr'] = da_dr
    df['DA_idr'] = da_idr
    df['mean_AE'] = ae
    df['rmse'] = mean_rmse
    df['MSE'] = mse
    df['QL'] = ql

    return(df)

def get_losses_for_tests(x):
    ae = x.iloc[:,2]
    se = x.iloc[:,3]
    ql = x.iloc[:,4]
    return(ae, se, ql)

exog_normal= [M1_n, M2_n, M3_n, M4_n, M8_n, M9_n]
exog_st = [M1_st, M2_st, M3_st, M4_st, M8_st, M9_st]
exog_ged = [M1_ged, M2_ged,M3_ged, M4_ged ,M5_ged, M6_ged, M7_ged,M8_ged]
exog_normal_e= [M1_n_e, M2_n_e, M3_n_e, M4_n_e, M8_n_e, M9_n_e]
exog_st_e = [M1_st_e, M2_st_e, M3_st_e, M4_st_e, M8_st_e, M9_st_e]
exog_ged_e = [M1_ged_e, M2_ged_e, M3_ged_e, M4_ged_e ,M5_ged_e, M6_ged_e, M7_ged_e,M8_ged_e]
base_models_1d = [model1, model2, model3, model4]
base_models_3d = [model1_3d, model2_3d, model3_3d, model4_3d]
base_models_5d = [model1_5d, model2_5d, model3_5d, model4_5d]
base_models_1de = [model1e, model2e, model3e, model4e]
base_models_3de = [model1_3de, model2_3de, model3_3de, model4_3de]
base_models_5de = [model1_5de, model2_5de, model3_5de, model4_5de]

a = (get_result_losses(base_models_1d))
b = (get_result_losses(base_models_1de))
c = (get_result_losses(base_models_3d))
d = (get_result_losses(base_models_3de))
e = (get_result_losses(base_models_5d))
f = (get_result_losses(base_models_5de))

g = get_result_losses(exog_normal)
h = get_result_losses(exog_normal_e)
i = get_result_losses(exog_ged)
j = get_result_losses(exog_ged_e)
k = get_result_losses(exog_st)
l = get_result_losses(exog_st_e)

df1 = pd.concat([g,h,i,j,k,l])
df = pd.concat([a,b,c,d,e,f])
df1.to_csv("Poutput/VIX_X_forecast_result/exog_loss.csv")
df.to_csv("Poutput/VIX_base_forecast_result/1d/base_loss.csv")
#print(df)
print(df1)
print(df)