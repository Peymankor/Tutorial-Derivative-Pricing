import numpy as np
import rich
from rich import print, pretty
pretty.install()

#############

from price_model import SimulateGBM
from basis_fun import laguerre_polynomials

##############


def priceOption(S0, K, r, paths, sd, T, steps, Stock_Matrix,k, reduce_variance = True):
  steps = int(steps)
  Stn = Stock_Matrix
  #Stn = Stock_Matrix
  dt = T/steps
  cashFlow = np.zeros((paths, steps))
  cashFlow[:,steps - 1] = np.maximum(K-Stn[:,steps - 1], 0)
      
  cont_value = cashFlow

  decision = np.zeros((paths, steps))
  decision[:, steps - 1] = 1

  discountFactor = np.tile(np.exp(-r*dt* np.arange(1, 
                                      steps + 1, 1)), paths).reshape((paths, steps))
  for i in reversed(range(steps - 1)):

          # Find in the money paths
          in_the_money_n = np.where(K-Stn[:, i] > 0)[0]
          out_of_money_n = np.asarray(list(set(np.arange(paths)) - set(in_the_money_n)))
          

          X = laguerre_polynomials(Stn[in_the_money_n, i], k)
          Y = cashFlow[in_the_money_n, i + 1]/np.exp(r*dt)

          A = np.dot(X.T, X)
          b = np.dot(X.T, Y)
          Beta = np.dot(np.linalg.pinv(A), b)

          cont_value[in_the_money_n,i] =  np.dot(X, Beta)
          try:
              cont_value[out_of_money_n,i] =  cont_value[out_of_money_n, i + 1]/np.exp(r*dt)
          except:
              pass

          decision[:, i] = np.where(np.maximum(K-Stn[:, i], 0)  - cont_value[:,i] >= 0, 1, 0)
          cashFlow[:, i] =  np.maximum(K-Stn[:, i], cont_value[:,i])
                  


  first_exercise = np.argmax(decision, axis = 1) 
  decision = np.zeros((len(first_exercise), steps))
  decision[np.arange(len(first_exercise)), first_exercise] = 1
  last = np.sum(decision*discountFactor*cashFlow, axis = 1)
  option_value = np.mean(last)
  var = np.sum((last-option_value)**2)/(last.shape[0]-1)
  return option_value
  #return option_value,var, cashFlow, decision


#######################################################

# Example of LSM Paper, First one

S0_value = 36
r_value = 0.06
sd_value = 0.2
T_value = 2
paths_value = 100000
steps_value = 50

K_value = 40
k_value = 4

Stock_Matrix_GBM = SimulateGBM(S0=S0_value, r=r_value, sd=sd_value, T=T_value, 
paths=paths_value,steps=steps_value)


price_reduced = priceOption(S0=S0_value, 
                                         K=K_value, r=r_value, paths=paths_value,
                                         sd=sd_value, T=T_value, steps=steps_value, 
                                         Stock_Matrix=Stock_Matrix_GBM,
                                         k=k_value,
                                         reduce_variance=True)

price_reduced


#########################################################
from scipy.stats import norm

def european_put_price(S0, K, r, sd, T) -> float:
  
  sigma_sqrt: float = sd * np.sqrt(T)
  d1: float = (np.log(S0 / K) +
                     (r + sd ** 2 / 2.) * T) \
            / sigma_sqrt
  d2: float = d1 - sigma_sqrt
  
  return K * np.exp(-r * T) * norm.cdf(-d2) \
            - S0 * norm.cdf(-d1)


#########################################################

S0_values_table1 = np.arange(36,44, 2)
sd_values_table1 = np.array([0.2, 0.4])
T_values_table1 = np.array([1, 2])



def Table1_func(S0_values,sd_values,T_values):
  print("%-10s %-10s %-10s %-20s %-20s %-20s" 
        %("S0","vol", "T", "Closed Form European", "Simulated American", "Early exercise value"))

  for S0_table1 in S0_values:
    for sd_table1 in sd_values:
      for T_table1 in T_values:

        euoption = european_put_price(S0=S0_table1, K=K_value, r=r_value,sd=sd_table1, T=T_table1)


        Stock_Matrix_GBM = SimulateGBM(S0=S0_table1, r=r_value, sd=sd_table1, T=T_table1, 
        paths=paths_value,steps=steps_value)
      
        Option_price = priceOption(S0=S0_table1, K=K_value, r=r_value, paths=paths_value,
        sd=sd_table1, T=T_table1, steps=steps_value, 
        Stock_Matrix=Stock_Matrix_GBM,
        k=k_value,reduce_variance=True)
      
        print("%d %10.2f %10d %20.3f %20.3f %20.3f" 
              %(S0_table1,sd_table1, T_table1, euoption, Option_price,Option_price-euoption))


Table1_func(S0_values=S0_values_table1, sd_values=sd_values_table1, T_values=T_values_table1)
