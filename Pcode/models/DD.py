import pandas as pd
import datetime as dt
from stat_testing import adf_test, get_prelim_stats, get_phillips_perron
import matplotlib.pyplot as plt

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

base_models_1d = [model1, model2, model3, model4]
base_models_3d = [model1, model2_3d, model3_3d, model4_3d]
base_models_5d = [model1, model2_5d, model3_5d,  model4_3d]

base_models_1de = [model1e, model2e, model3e, model4e]
base_models_3de = [model1_3de, model2_3de]
base_models_5de = [model1_5de, model2_5de]
import numpy as np

def get_err(x):

    AE = []
    MSE = []
    RMSE = []
    QL = []
    for i in range(len(x)):
        ae = x[i].iloc[:,2].mean()
        se = x[i].iloc[:,3].mean()
        ql = x[i].iloc[:,4].sum()
        rmse = np.sqrt(se)
        AE.append(ae)
        MSE.append(se)
        RMSE.append(rmse)
        QL.append(ql)
    df = pd.concat([pd.Series(AE), pd.Series(RMSE),pd.Series(MSE), pd.Series(QL)], axis=1)
    return df


base_1d = get_err(base_models_1d)
base_1de = get_err(base_models_1de)
base_3d = get_err(base_models_3d)
base_3de = get_err(base_models_3de)
base_5d = get_err(base_models_5d)
base_5de = get_err(base_models_5de)

errors = pd.concat([base_1d, base_1de, base_3d, base_3de, base_5d, base_5de], axis=0)
errors.to_csv("Poutput\Lossfunctionvalue.csv")
print(errors)