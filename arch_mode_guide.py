import datetime as dt
import pandas as pd
import pandas_datareader as web
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn

seaborn.set_style("darkgrid")
plt.rc("figure", figsize=(16, 6))
plt.rc("savefig", dpi=90)
plt.rc("font", family="sans-serif")
plt.rc("font", size=14)

start = dt.datetime(2000, 1, 1)
end = dt.datetime(2020, 1, 1)

sp500 = web.DataReader('^GSPC', 'yahoo', start=start, end=end)

sp500
returns = 100 * sp500['Adj Close'].pct_change().dropna()

am = arch_model(returns)

res = am.fit()



ax = returns.plot()
xlim = ax.set_xlim(returns.index.min(), returns.index.max())
plt.show()

fig = res.plot(annualize="D")
plt.show()

sim_data = am.simulate(res.params, 100)
sim_data.head()

import numpy as np
data_store = np.zeros([10,11])

data_store[:,0] = 3500
data_store

for i in range(10):

    ret_sim = am.simulate(res.params, 10)["data"]
    
    for j in range(10):
    
        data_store[i,j+1] =  data_store[i,j]+ ret_sim[j]*data_store[i,j] 



Time_steps = np.arange(1, 10+1)

plt.plot(Time_steps, data_store)
plt.show()

data_store
for t in range(1,10):
    print(t)

