
# packages, rich is terminal output formatting
import numpy as np
import math
import rich
from rich import print, pretty
import matplotlib.pyplot as plt
pretty.install()

# GMB Model
# log(x_T) ~ N(log(x_S)+(mu-sigma^2/2)(T-S),sigma^2(T-S))

def SimulateGBM(S0, r, sd, T, paths, steps, reduce_variance = True):
    steps = int(steps)
    dt = T/steps
    Z = np.random.normal(0, 1, paths//2 * steps).reshape((paths//2, steps))
    # Z_inv = np.random.normal(0, 1, paths//2 * steps).reshape((paths//2, steps))
    if reduce_variance:
      Z_inv = -Z
    else:
      Z_inv = np.random.normal(0, 1, paths//2 * steps).reshape((paths//2, steps))
    dWt = math.sqrt(dt) * Z
    dWt_inv = math.sqrt(dt) * Z_inv
    dWt = np.concatenate((dWt, dWt_inv), axis=0)
    St = np.zeros((paths, steps + 1))
    St[:, 0] = S0
    for i in range (1, steps + 1):
        St[:, i] = St[:, i - 1]*np.exp((r - 1/2*np.power(sd, 2))*dt + sd*dWt[:, i-1])
    
    return St[:,1:]
    #return St

######## Test ##############################
S0_value = 36
r_value = 0.06
sd_value = 0.2
T_value = 1
paths_value = 100
steps_value = 50

Test_GBM = SimulateGBM(S0=S0_value, r=r_value, sd=sd_value, T=T_value, paths=paths_value,
steps=steps_value)


########## Plot the Data ################

Time_steps = np.arange(1, steps_value+1)

#plt.plot(Time_steps, Test_GBM.T)
#plt.show()

######################################################################

############### GARCH Mode######################################

from dataclasses import dataclass
import numpy as np
import scipy.optimize as spop
import scipy.stats as ss
import matplotlib.pyplot as plt
import pandas as pd

#############################################
Brent_crude_df = pd.read_excel("https://www.eia.gov/dnav/pet/hist_xls/RBRTEd.xls", 
                        sheet_name="Data 1")
###############################################


@dataclass(frozen=True)
class Brent_GARCH:

    brent_df: np.ndarray

    def data_process(self):

     Brent_crude=self.brent_df.iloc[2:,]

     Brent_crude.columns= ["Date", "Dollar"]

     Brent_crude["Date"] = pd.to_datetime(Brent_crude["Date"])
    
     Brent_crude["Dollar"] = pd.to_numeric(Brent_crude["Dollar"])

     returns = np.array(100*Brent_crude["Dollar"].dropna().pct_change().dropna())
     
     return Brent_crude, returns


    def garch_mle(self, params,returns):
    
        mu = params[0]
        omega = params[1]
        alpha = params[2]
        beta = params[3]
    
    #calculating long-run volatility
        long_run = (omega/(1 - alpha - beta))**(1/2)
    #calculating realised and conditional volatility
        resid = returns - mu
        realised = abs(resid)
        conditional = np.zeros(len(returns))
        conditional[0] =  long_run
        for t in range(1,len(returns)):
            conditional[t] = (omega + alpha*resid[t-1]**2 + beta*conditional[t-1]**2)**(1/2)
    #calculating log-likelihood
        likelihood = 1/((2*np.pi)**(1/2)*conditional)*np.exp(-realised**2/(2*conditional**2))
        log_likelihood = np.sum(np.log(likelihood))
        return -log_likelihood

    def mle_opt(self):

        _ , returns = self.data_process()
        

        mean = np.average(returns)
        var = np.std(returns)**2
        
        mle_param=spop.minimize(self.garch_mle, [mean, var, 0, 0], method='Nelder-Mead',
                args=returns)

        return mle_param.x, returns

    def compare_model(self):
        
        mle_param_v, returns = self.mle_opt()

        mu = mle_param_v[0]
        omega = mle_param_v[1]
        alpha = mle_param_v[2]
        beta = mle_param_v[3]

        long_run = (omega/(1 - alpha - beta))**(1/2)
        resid = returns - mu
        realised = abs(resid)
        conditional = np.zeros(len(returns))
        conditional[0] =  long_run
        for t in range(1,len(returns)):
                conditional[t] = (omega + alpha*resid[t-1]**2 + beta*conditional[t-1]**2)**(1/2)

        return realised, conditional

    def generate_paths(self,num_time_steps,num_path_numbers, initial_price):

        T_steps = num_time_steps
        path_numbers = num_path_numbers
        #resid = returns - mu

        mle_param_v, _ = self.mle_opt()

        mu = mle_param_v[0]
        omega = mle_param_v[1]
        alpha = mle_param_v[2]
        beta = mle_param_v[3]

        realised_v, conditional_v= self.compare_model()

        resid_sim = np.zeros((path_numbers, T_steps))
        resid_sim[:,0] = realised_v[-1]

        conditional_sim = np.zeros((path_numbers, T_steps))
        conditional_sim[:,0] = conditional_v[-1]

        S = np.zeros((path_numbers, T_steps))
        S[:,0] = initial_price


        for path in range(path_numbers):

            for t in range(1, T_steps):
    
                conditional_sim[path,t] = (omega + alpha*resid_sim[path,t-1]**2 + beta*conditional_sim[path,t-1]**2)**(1/2)

                r = ss.norm.rvs(loc=mu, scale=conditional_sim[path,t])

                resid_sim[path, t] = r - 0

                S[path, t] = S[path,t-1]*(1+(r/100))

        return S[:,1:]