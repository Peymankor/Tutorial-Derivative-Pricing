import pandas as pd
# pip install xlrd

import matplotlib.pyplot as plt
import numpy as np
from arch import arch_model

#######################################

Brent_crude = pd.read_excel("https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls", 
                sheet_name="Data 1")

Brent_crude=Brent_crude.iloc[2:,]

Brent_crude.columns= ["Date", "Dollar"]

Brent_crude.dtypes

Brent_crude["Date"] = pd.to_datetime(Brent_crude["Date"])
Brent_crude["Dollar"] = pd.to_numeric(Brent_crude["Dollar"])


plt.plot(Brent_crude["Date"],Brent_crude["Dollar"])
plt.show()

###########################################

Brent_crude_ret = 100*Brent_crude["Dollar"].dropna().pct_change().dropna()
Brent_crude_ret

plt.plot(Brent_crude["Date"][1:],Brent_crude_ret)
#plt.xlim.set_xlim(Brent_crude_ret.index.min(), Brent_crude_ret.index.max())
#plt.x
plt.show()

###############################################
Brent_normal_model = arch_model(Brent_crude_ret)
Brent_normal_fit = arch_model(Brent_crude_ret).fit()

pd.DataFrame(Brent_normal.params)

#print(Brent_normal.summary())

Time_steps = 30
Number_of_paths = 1000

data_store = np.zeros([Number_of_paths,Time_steps +1])

data_store[:,0] = 115.13
data_store

for i in range(Number_of_paths):

    ret_sim = Brent_normal_model.simulate(Brent_normal_fit.params, Time_steps)["data"]
    
    for j in range(Time_steps):
    
        data_store[i,j+1] =  data_store[i,j]+ ret_sim[j]*data_store[i,j]/100 


Time_steps = np.arange(0, Time_steps+1)

plt.plot(Time_steps, data_store.T)
plt.xlabel("Days after Day 0")
plt.ylabel("Brent Crude Spot Price (Dollar)")
plt.title("Simulation of 1000 paths of Brent Crude Oil Price Follwing GARCH(1,1) model")
plt.show()