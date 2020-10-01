import pandas as pd
import matplotlib.pyplot as plt
ged = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\GED.csv", index_col = 0)
ged_egarch = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\GED_egarch.csv", index_col = 0)
norm = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\Normal.csv", index_col = 0)
norm_egarch = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\Normal_egarch.csv", index_col = 0)
skew_stud = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\skewstud.csv", index_col = 0)
skewstud_egarch = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\skewstud_egarch.csv", index_col = 0)
studT = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\studT.csv", index_col = 0)
studT_egarch = pd.read_csv(r"Poutput\VIX_base_forecast_result\1d\studT_egarch.csv", index_col = 0)

base_forecasts = [ged, ged_egarch, norm, norm_egarch, skew_stud, skewstud_egarch, studT, studT_egarch]

plt.plot(ged.iloc[:,-1])
plt.show()