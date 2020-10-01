import pandas as pd

data = pd.read_csv(r"Pdata/vix_is.csv", index_col=0)

var = ['vix', 'return', 'neg_ret', 'neg_ret_5', 'neg_ret_22', 'bsesn_dr', 'bsesn_idr', 'MOVE','MOVE_l1', 'vvix','vvix_l1', 'skew','skew_l1', 'Skew_diff']
exog = data[var]
correlation_matrix = (exog.corr())
corr_matrix = pd.DataFrame(correlation_matrix)
print(corr_matrix)


corr_matrix.to_csv("Poutput/CorrelationMatrix.csv")
ret = data['return']
vix = data['vix']
sqr_ret = (data['return'])**2

df = pd.concat([ret, vix, sqr_ret], axis=1)
df.to_csv("Poutput/second_corr_matrix.csv")
print(df.corr())