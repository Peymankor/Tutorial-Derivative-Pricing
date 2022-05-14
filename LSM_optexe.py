from audioop import reverse
from dataclasses import dataclass, replace
from typing import Callable, Sequence, Tuple, List
import numpy as np
from pandas import test
from scipy.stats import norm

from random import randrange
from numpy.polynomial.laguerre import lagval
from basis_fun import laguerre_polynomials
from price_model import SimulateGBM

from rich import print, pretty
pretty.install()

TrainingDataType = Tuple[int, float, float]

def lsm_policy(S0, K, r, paths, sd, T, steps, Stock_Matrix,k, reduce_variance = True):
  steps = int(steps)
  Stn = Stock_Matrix
  #Stn = Stock_Matrix
  dt = T/steps
  cashFlow = np.zeros((paths, steps))
  cashFlow[:,steps - 1] = np.maximum(K-Stn[:,steps - 1], 0)
      
  cont_value = cashFlow

  decision = np.zeros((paths, steps))
  decision[:, steps - 1] = 1
  Beta_list = {}
  #discountFactor = np.tile(np.exp(-r*dt* np.arange(1, 
  #                                    steps + 1, 1)), paths).reshape((paths, steps))
  for i in reversed(range(steps-1)):

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
          
          Beta_list.update({i: Beta}) 

                  
  return Beta_list

from price_model import SimulateGBM

S0_value = 36
r_value = 0.06
sd_value = 0.4
T_value = 1
paths_value = 1000000
steps_value = 50

K_value = 40
k_value = 4

Stock_Matrix_GBM = SimulateGBM(S0=S0_value, r=r_value, sd=sd_value, T=T_value, 
paths=paths_value,steps=steps_value)

Stock_Matrix_GBM
Beta_list = lsm_policy(S0=S0_value, 
                                         K=K_value, r=r_value, paths=paths_value,
                                         sd=sd_value, T=T_value, steps=steps_value, 
                                         Stock_Matrix=Stock_Matrix_GBM,
                                         k=k_value,
                                         reduce_variance=True)

#Beta_list[0]

def laguerre_polynomials_ind(S, k):
    
    #  the first k terms of Laguerre Polynomials (k<=4)
#    x1 = np.exp(-S/2)
#    x2 = np.exp(-S/2) * (1 - S)
#    x3 = np.exp(-S/2) * (1 - 2*S + S**2/2)
#    x4 = np.exp(-S/2) * (1 - 3*S + 3* S**2/2 - S**3/6)

    u0 = np.ones(S.shape)
    x1 = 1 - S
    x2 = 1 - 2*S + S**2/2
    x3 = 1 - 3*S + 3*S**2/2 - S**3/6
    x4 = 1 - 4*S + 3*S**2 - 2*S**3/3 + S**4/24

    X  = [np.stack([u0, x1, x2]),
          np.stack([u0, x1, x2, x3]),
          np.stack([u0, x1, x2, x3, x4])]
    
    return X[k-2]



def option_price_lsm(
        scoring_data: np.ndarray,
        num_steps,
        payoff,
        rate,
        expiry,
        Beta_list
        #func: FunctionApprox[Tuple[float, float]]
    ) -> float:
        num_paths: int = scoring_data.shape[0]
        prices: np.ndarray = np.zeros(num_paths)
        dt: float = expiry / num_steps

        #Beta_list.reverse()

        for i, path in enumerate(scoring_data):
            step: int = 0
            while step <= num_steps:
                t: float = (step+1) * dt
                exercise_price: float = payoff(t, path[step])
                
                if exercise_price>0:
                    XX=laguerre_polynomials_ind(path[step],4)
                    continue_price: float = np.dot(XX, Beta_list[step])  \
                        if step < num_steps else 0.

                #continue_price: float = func.evaluate([(t, path[step])])[0] \
                #    if step < self.num_steps else 0.
                    step += 1
                    if exercise_price >= continue_price:
                        prices[i] = np.exp(-rate * t) * exercise_price
                        step = num_steps + 1
                else:
                    step += 1


        return np.average(prices)


from rl_optexe import OptimalExerciseRL

strike = 40
def payoff_func(_: float, s: float) -> float:
        return max(strike - s, 0.)

data_object = OptimalExerciseRL(spot_price=K_value,payoff=payoff_func,expiry=1, 
rate=r_value, vol=sd_value, num_steps=steps_value)

number_scoring_paths = 10000
test_data = data_object.scoring_sim_data(num_paths=number_scoring_paths)

test_data_1 = test_data[:,1:]

#test_data_1
#Beta_list
steps_value_1 = steps_value-1
option_price_lsm(scoring_data=test_data_1, num_steps=steps_value_1,
payoff=payoff_func, rate=r_value, expiry=T_value, Beta_list=Beta_list)


stepss = 10
for i in reversed(range(10)):
    print(i)

testlist = [1,2,3]
reverse_list = reversed(testlist)
reverse_list
testlist

steps = 50
T =1
steps = int(steps)
r = 0.06
paths = 10  
  #Stn = Stock_Matrix
dt = T/steps
  #cashFlow = np.zeros((paths, steps))
  #cashFlow[:,steps - 1] = np.maximum(K-Stn[:,steps - 1], 0)
      
 # cont_value = cashFlow

# decision = np.zeros((paths, steps))
# decision[:, steps - 1] = 1

discountFactor = np.tile(np.exp(-r*dt* np.arange(1, 
                                      steps + 1, 1)), paths).reshape((paths, steps))
discountFactor


np.exp(-rate * t)