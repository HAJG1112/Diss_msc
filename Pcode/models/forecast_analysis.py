import pandas as pd
import stat_testing as st
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

base_models_1d = [model1, model2, model3, model4]
base_models_3d = [model1_3d, model2_3d, model3_3d, model4_3d]
base_models_5d = [model1_5d, model2_5d, model3_5d, model4_5d]
base_models_1de = [model1e, model2e, model3e, model4e]
base_models_3de = [model1_3de, model2_3de]
base_models_5de = [model1_5de, model2_5de]

exog_norm= [M1_n, M2_n,  M3_n, M4_n, M8_n, M9_n]
exog_st = [M1_st, M2_st,  M3_st, M4_st, M8_st, M9_st]
exog_ged = [M1_ged, M2_ged, M3_ged, M4_ged ,M5_ged, M6_ged, M7_ged, M8_ged]
exog_norm_e= [M1_n_e, M2_n_e,  M3_n_e, M4_n_e, M8_n_e, M9_n_e]
exog_st_e = [M1_st_e, M2_st_e, M3_st_e, M8_st_e]
exog_ged_e = [M1_ged_e, M2_ged_e,  M3_ged_e, M4_ged_e ,M5_ged_e, M6_ged_e, M7_ged_e, M8_ged_e]


def get_date_val(forecasts_group):
    from scipy.stats import kurtosis, skew, jarque_bera
    from stat_testing import adf_test
    import statistics
    date_min = []
    date_max = []
    max = []
    min = []
    skew = []
    kurtosis = []
    adf = []

    for i in range(len(forecasts_group)):
        df = forecasts_group[i]
        date = df[['close_err']].idxmax()[0]
        date1 = df[['close_err']].idxmin()[0]
        max1 = df[['close_err']].max()[0]
        min1 = df[['close_err']].min()[0]
        s = df[['close_err']].skew()[0]
        k = df[['close_err']].kurtosis()[0]
        adf_p = adf_test(df[['close_err']], 'ct')[1]
        date_max.append(date)
        date_min.append(date1)
        max.append(max1)
        min.append(min1)
        skew.append(s)
        kurtosis.append(k)
        adf.append(adf_p)

    d_min = pd.DataFrame(date_min)
    min_col = pd.DataFrame(min)
    d_max = pd.DataFrame(date_max)
    max_col = pd.DataFrame(max)
    skeww = pd.DataFrame(skew)
    kurtosiss = pd.DataFrame(kurtosis)
    adf_vals  = pd.DataFrame(adf)


    df = pd.concat([d_min, min_col, d_max, max_col, skeww, kurtosiss, adf_vals], axis=1)
    df.columns = ['Date_min', 'Min_val', 'Date_max', 'Max_val', 'Skew', 'Kurtosis', 'ADF']
    return(df)

a1 = get_date_val(base_models_1d)      
a2 = get_date_val(base_models_1de)       
a3 = get_date_val(base_models_3d)       
a4 = get_date_val(base_models_3de)       
a5 = get_date_val(base_models_5d)       
a6 = get_date_val(base_models_5de)       
a7 = get_date_val(exog_norm)
a8 = get_date_val(exog_ged)
a9 = get_date_val(exog_st)
a10 = get_date_val(exog_norm_e)
a11 = get_date_val(exog_ged_e)
a12 = get_date_val(exog_st_e)

all_res = pd.concat([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12])
print(all_res)
all_res.to_csv("Poutput/forecast_res_summ.csv")