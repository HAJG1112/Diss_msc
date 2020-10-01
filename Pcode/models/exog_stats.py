import pandas as pd
from stat_testing import get_prelim_stats,  adf_test, get_phillips_perron
import matplotlib.pyplot as plt
vix_is = pd.read_csv(r"Pdata\vix_is.csv", index_col = 0, parse_dates=True)

vix = vix_is['vix']
returns = vix_is['return']
bsesn_dr = vix_is['bsesn_dr']
bsesn_idr = vix_is['bsesn_idr']
move = vix_is['MOVE']
vvix = vix_is['vvix']
skew = vix_is['skew']
skew_diff = vix_is['Skew_diff'].dropna()

fig, (ax1, ax2,ax3, ax4, ax5, ax6, ax7, ax8) = plt.subplots(8, sharex=True, figsize=(10,14))
ax1.plot(vix, linewidth=0.5, color = 'black')
ax1.legend(['VIX'])

ax2.plot(returns, linewidth=0.5, color = 'blue')
ax2.legend(['S&P500 Returns'])

ax3.plot(bsesn_dr, linewidth=0.5, color = 'blue')
ax3.legend(['BSESN(DR)'])

ax4.plot(bsesn_idr,linewidth=0.5, color = 'purple')
ax4.legend(['BSESN(IDR)'])

ax5.plot(move, linewidth=0.5, color = 'orange')
ax5.legend(['MOVE'])

ax6.plot(vvix, linewidth=0.5, color = 'green')
ax6.legend(['VVIX'])

ax7.plot(skew, linewidth=0.5, color = 'red')
ax7.legend(['SKEW'])

ax8.plot(skew_diff, linewidth=0.5, color = 'black')
ax8.legend(['SKEW-diff'])

plt.savefig('Pgraphs/test/exog')
