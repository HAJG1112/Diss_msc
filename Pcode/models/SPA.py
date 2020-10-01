from arch.bootstrap import SPA
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
exog_norm= [M1_n, M2_n, M2a_n, M3_n, M4_n, M8_n, M9_n]
exog_st = [M1_st, M2_st, M2a_st, M3_st, M4_st, M8_st, M9_st]
exog_ged = [M1_ged, M2_ged, M2a_ged ,M3_ged, M4_ged ,M5_ged, M6_ged, M7_ged,M8_ged]
exog_norm_e= [M1_n_e, M2_n_e, M2a_n_e, M3_n_e, M4_n_e, M8_n_e, M9_n_e]
exog_st_e = [M1_st_e, M2_st_e, M3_st_e, M8_st_e]
exog_ged_e = [M1_ged_e, M2_ged_e, M2a_ged_e ,M3_ged_e, M4_ged_e ,M5_ged_e, M6_ged_e, M7_ged_e, M8_ged_e]
bm_1d = model1
bm_3d = model1_3d
bm_5d = model1_5d
base_models_1d = [model2, model3, model4]
base_models_3d = [model2_3d, model3_3d, model4_3d]
base_models_5d = [model2_5d, model3_5d,  model4_3d]
base_models_1de = [model1e, model2e, model3e, model4e]
base_models_3de = [model1_3de, model2_3de]
base_models_5de = [model1_5de, model2_5de]
exog = [M1_n, M2_n, M2a_n, M3_n, M4_n, M8_n, M9_n, M1_n_e, M2_n_e, M2a_n_e, M3_n_e, M4_n_e, M8_n_e, M9_n_e,\
    M1_st, M2_st, M2a_st, M3_st, M4_st, M8_st, M9_st, M1_st_e, M2_st_e, M3_st_e, M8_st_e, \
        M1_ged, M2_ged, M2a_ged ,M3_ged, M4_ged ,M5_ged, M6_ged, M7_ged,M8_ged, M1_ged_e, M2_ged_e, M2a_ged_e ,M3_ged_e, M4_ged_e ,M5_ged_e, M6_ged_e, M7_ged_e, M8_ged_e]



def get_SPA(bm_losses, model_losses):
    spa = SPA(bm_losses, model_losses, reps = 10000, block_size=44, bootstrap='mbb')
    spa.compute()
    return (spa.pvalues[1])

def get_ae(x):
    ae = x.iloc[:,2]
    return ae

def get_mse(x):
    mse = x.iloc[:,3]
    return mse

def get_ql(x):
    ql = x.iloc[:,4]
    return ql

def get_spa_models(bm_loss, models):
    AE = []
    MSE = []
    QL = []
    
    for i in range(len(models)):
        ae = get_SPA(get_ae(bm_loss), get_ae(models[i]))
        mse = get_SPA(get_mse(bm_loss), get_mse(models[i]))
        ql = get_SPA(get_ql(bm_loss), get_ql(models[i]))
        AE.append(ae)
        MSE.append(mse)
        QL.append(ql)
    
    df = pd.DataFrame()
    df['AE'] = AE
    df['MSE'] = MSE
    df['QL'] = QL
    s2 = pd.Series([np.nan, np.nan,np.nan])
    df = df.append(s2, ignore_index =True)
    #print(df)
    return df

def gon_get(x):
    spa = pd.DataFrame()
    for i in range(len(x)):
        all = get_spa_models(bm_1d, x)
        fin = pd.concat([spa, all])
        return fin

def gon_get3(x):
    spa = pd.DataFrame()
    for i in range(len(x)):
        all = get_spa_models(bm_3d, x)
        fin = pd.concat([spa, all])
    return spa
def gon_get5(x):
    spa = pd.DataFrame()
    for i in range(len(x)):
        all = get_spa_models(bm_5d, x)
        fin = pd.concat([spa, all])
    return spa

exog1 = [M3_n, M4_n, M8_n, M3_n_e, M4_n_e, M8_n_e,\
        M1_ged, M3_ged, M4_ged, M6_ged, M8_ged, \
        M1_ged_e, M3_ged_e, M4_ged_e, M6_ged_e, M8_ged_e,
        M1_st, M2_st, \
        M1_st_e,M2_st_e]

a1 = gon_get(base_models_1d)
a2 = gon_get(base_models_1de)
a3 = gon_get3(base_models_3d)
a4 = gon_get3(base_models_3de)
a5 = gon_get5(base_models_5d)
a6 = gon_get5(base_models_5de)
    

all_base = pd.concat([a1,a7,a2,a3,a4,a5,a6])
print(all_base)
all_base.to_csv(r"Poutput/SPA/SPA_values.csv")



