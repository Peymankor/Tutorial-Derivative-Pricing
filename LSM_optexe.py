from audioop import reverse
from dataclasses import dataclass, replace
from re import I
from typing import Callable, Dict, Sequence, Tuple, List
import numpy as np
from pandas import test
from scipy.stats import norm

from random import randrange
from numpy.polynomial.laguerre import lagval
from basis_fun import laguerre_polynomials
from price_model import SimulateGBM

from rich import print, pretty
pretty.install()

import matplotlib.pyplot as plt



from price_model import SimulateGBM


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

@dataclass(frozen=True)
class OptimalExerciseLSM:

    spot_price: float
    payoff: Callable[[float, float], float]
    expiry: float
    rate: float
    vol: float
    num_steps: int

    def GBMprice_training(self, num_paths_train) -> np.ndarray:
        
        return SimulateGBM(
            S0=self.spot_price,
            r=self.rate, sd=self.vol, T=self.expiry, 
            paths=num_paths_train, steps=self.num_steps)

    
    def train_LSM(self, num_paths_train: int, K:float, k:int, training_data: np.ndarray):
            
            steps = int(self.num_steps)
            Stn = training_data
            #Stn = Stock_Matrix
            dt = self.expiry/steps
            cashFlow = np.zeros((num_paths_train, steps))
            cashFlow[:,steps - 1] = np.maximum(K-Stn[:,steps - 1], 0)
      
            cont_value = cashFlow

            decision = np.zeros((num_paths_train, steps))
            decision[:, self.num_steps - 1] = 1
            Beta_list = {}
  
            for i in reversed(range(steps-1)):

          # Find in the money paths
                in_the_money_n = np.where(K-Stn[:, i] > 0)[0]
                out_of_money_n = np.asarray(list(set(np.arange(num_paths_train)) - set(in_the_money_n)))
          
                X = laguerre_polynomials(Stn[in_the_money_n, i], k)
                Y = cashFlow[in_the_money_n, i + 1]/np.exp(self.rate*dt)

                A = np.dot(X.T, X)
                b = np.dot(X.T, Y)
                Beta = np.dot(np.linalg.pinv(A), b)
                try:
                    cont_value[out_of_money_n,i] =  cont_value[out_of_money_n, i + 1]/np.exp(r*dt)
                except:
                    pass

                decision[:, i] = np.where(np.maximum(K-Stn[:, i], 0)  - cont_value[:,i] >= 0, 1, 0)
                cashFlow[:, i] =  np.maximum(K-Stn[:, i], cont_value[:,i])
                
                Beta_list.update({i: Beta}) 
            
            return Beta_list
    
    def scoring_sim_data(self, num_paths_test: int) -> np.ndarray:
            return SimulateGBM(
            S0=self.spot_price,
            r=self.rate, sd=self.vol, T=self.expiry, 
            paths=num_paths_test, steps=self.num_steps)

    def option_price(self,
        scoring_data: np.ndarray,
        Beta_list,
        k
        #func: FunctionApprox[Tuple[float, float]]
    ) -> float:
        num_paths: int = scoring_data.shape[0]
        prices: np.ndarray = np.zeros(num_paths)
        dt: float = self.expiry / self.num_steps

        #Beta_list.reverse()

        for i, path in enumerate(scoring_data):
            step: int = 0
            while step < self.num_steps:
                t: float = (step+1) * dt
                exercise_price: float = self.payoff(t, path[step])
                
                if exercise_price>0:
                    XX=laguerre_polynomials_ind(path[step],k)
                    continue_price: float = np.dot(XX, Beta_list[step])  \
                        if step < self.num_steps else 0.

                #continue_price: float = func.evaluate([(t, path[step])])[0] \
                #    if step < self.num_steps else 0.
                    step += 1
                    if exercise_price >= continue_price:
                        prices[i] = np.exp(-self.rate * t) * exercise_price
                        step = self.num_steps + 1
                else:
                    step += 1


        return np.average(prices)


K = 40
def payoff_func(_: float, s: float) -> float:
        return max(K - s, 0.)

LSMtest = OptimalExerciseLSM(spot_price=36,payoff=payoff_func,expiry=1, rate=0.06, 
vol=0.2, num_steps=50)

GBMdata = LSMtest.GBMprice_training(num_paths_train=10000)
trained_lsm = LSMtest.train_LSM(num_paths_train=10000,K=40,k=4, 
    training_data=GBMdata) 
trained_lsm

GBMtest = LSMtest.scoring_sim_data(num_paths_test=1000)
GBMtest 



price_of_option2 = LSMtest.option_price(scoring_data=GBMtest, Beta_list=trained_lsm,
                                        k=4)
price_of_option2

trained_lsm

LSMtest
GBMdata
GBMtest
trained_lsm

plt.figure(figsize=(15, 12))
plt.subplots_adjust(hspace=1)
plt.suptitle("Daily closing prices", fontsize=18, y=0.95)

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
          
          #ax=plt.subplot(5, 2, i+1)
          #plt.subplot(211)          
          #plt.plot(Stn[in_the_money_n, i], Y, "bs", Stn[in_the_money_n, i], cont_value[in_the_money_n,i],"rs")
          #plt.set_title('district_{}_change'.format(i))
          #plt.title('Time Step=' + str(i))
          #plt.xlabel("Price of the Underlying Asset (S)")
          #plt.ylabel("Continuation value , V(S)")
          #ax.set_ylim([0,40])
          #ax.set_xlim([10,40])
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
sd_value = 0.2
T_value = 1
paths_value = 10000
steps_value = 50

K_value = 40
k_value = 4

Stock_Matrix_GBM = SimulateGBM(S0=S0_value, r=r_value, sd=sd_value, T=T_value, 
paths=paths_value,steps=steps_value, reduce_variance=False)

#Stock_Matrix_GBM_train = Stock_Matrix_GBM

#from rl_optexe import OptimalExerciseRL

strike = 40
def payoff_func(_: float, s: float) -> float:
        return max(strike - s, 0.)


#data_object = OptimalExerciseRL(spot_price=S0_value,payoff=payoff_func,expiry=1, 
#rate=r_value, vol=sd_value, num_steps=steps_value)

#Stock_Matrix_GBM = data_object.scoring_sim_data(num_paths=paths_value)

#Stock_Matrix_GBM_train = Stock_Matrix_GBM[:,1:]
#Stock_Matrix_GBM_train


Beta_list1 = lsm_policy(S0=S0_value, 
                                         K=K_value, r=r_value, paths=paths_value,
                                         sd=sd_value, T=T_value, steps=steps_value, 
                                         Stock_Matrix=Stock_Matrix_GBM,
                                         k=k_value,
                                         reduce_variance=True)

#plt.show()
Beta_list1
price_of_option2 = LSMtest.option_price(scoring_data=GBMtest, Beta_list=Beta_list1,
                                        k=4)
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

# better regression equation of the 

def option_price_lsm(
        scoring_data: np.ndarray,
        num_steps,
        payoff,
        rate,
        expiry,
        Beta_list,
        k
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
                    XX=laguerre_polynomials_ind(path[step],k)
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


#from rl_optexe import OptimalExerciseRL

strike = 40
def payoff_func(_: float, s: float) -> float:
        return max(strike - s, 0.)

#data_object = OptimalExerciseRL(spot_price=K_value,payoff=payoff_func,expiry=1, 
#rate=r_value, vol=sd_value, num_steps=steps_value)

number_scoring_paths = 50000
#test_data = data_object.scoring_sim_data(num_paths=number_scoring_paths)
#Stock_Matrix_GBM_test=test_data[:,1:]

#Stock_Matrix_GBM_test
#test_data_1 = test_data[:,1:]

#test_data_1
#Beta_list
#steps_value_1 = steps_value-1


Stock_Matrix_GBM_test = SimulateGBM(S0=S0_value, r=r_value, sd=sd_value, T=T_value, 
paths=10000,steps=steps_value, reduce_variance=False)

option_price_lsm(scoring_data=Stock_Matrix_GBM_test, num_steps=steps_value-1,
payoff=payoff_func, rate=r_value, expiry=T_value, Beta_list=trained_lsm, k=k_value)


price_of_option2 = LSMtest.option_price(scoring_data=GBMtest, Beta_list=Beta_list1,
                                        k=4)
Beta_list1
trained_lsm
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


#np.exp(-rate * t)

import pandas as pd

price_of_option2 = LSMtest.option_price(scoring_data=GBMtest, Beta_list=Beta_list,
                                        k=4)
trained_lsm[48]
Beta_list1[48]

trained_lsm = LSMtest.train_LSM(num_paths_train=10000,K=40,k=4, 
    training_data=Stock_Matrix_GBM) 
trained_lsm[0]
Beta_list1[0]


########################################################
#######################################################
########################################################


plt.figure(figsize=(15, 12))
plt.subplots_adjust(hspace=0.7)
plt.suptitle("Expected Continutaion value in LSM", fontsize=16, y=0.98)

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
  
  indext = 10
  for index, i in enumerate(reversed(range(steps-1))):

          # Find in the money paths
          in_the_money_n = np.where(K-Stn[:, i] > 0)[0]
          out_of_money_n = np.asarray(list(set(np.arange(paths)) - set(in_the_money_n)))
          

          X = laguerre_polynomials(Stn[in_the_money_n, i], k)
          Y = cashFlow[in_the_money_n, i + 1]/np.exp(r*dt)

          A = np.dot(X.T, X)
          b = np.dot(X.T, Y)
          Beta = np.dot(np.linalg.pinv(A), b)

          cont_value[in_the_money_n,i] =  np.dot(X, Beta)
          
          if i in np.arange(0,50,5):  
            ax=plt.subplot(5, 2, indext)
          #plt.subplot(211)        
            indext = indext -1 
            xs, ys = zip(*sorted(zip(Stn[in_the_money_n, i], cont_value[in_the_money_n,i])))
            #plt.plot(Stn[in_the_money_n, i], Y, "bs", xs, 
            #        ys,"r--",np.arange(25,41),40-np.arange(25,41), "g--", 
            #        markersize=1, label="log10")
            l1 = ax.plot(Stn[in_the_money_n, i], Y, "bs", markersize=1, 
            label = "continuation value")
            l2 = ax.plot(xs, ys,"r--",  
            label = "Conditional Expectaion (Regresion)", linewidth=2)
            l3 = ax.plot(np.arange(20,41),40-np.arange(20,41), "g--",  
            label = "Immediate Payoff", linewidth=2)
            
          #plt.set_title('district_{}_change'.format(i))
            plt.title('Time Step=' + str(i+1))
            plt.xlabel("Price of the Underlying Asset (S)")
            plt.ylabel("Continuation value , V(S)")
            ax.set_ylim([0,30])
            ax.set_xlim([10,40])
            if indext ==0:
                ax.legend(loc="lower left")
            #ax.legend([l1], "hda")
            #ax[1].legend(loc=(1.1, 0.5))

          try:
              cont_value[out_of_money_n,i] =  cont_value[out_of_money_n, i + 1]/np.exp(r*dt)
          except:
              pass

          decision[:, i] = np.where(np.maximum(K-Stn[:, i], 0)  - cont_value[:,i] >= 0, 1, 0)
          cashFlow[:, i] =  np.maximum(K-Stn[:, i], cont_value[:,i])
          
          Beta_list.update({i: Beta}) 

                  
  return Beta_list



S0_value = 36
r_value = 0.06
sd_value = 0.4
T_value = 1
paths_value = 10000
steps_value = 50

K_value = 40
k_value = 4

Stock_Matrix_GBM = SimulateGBM(S0=S0_value, r=r_value, sd=sd_value, T=T_value, 
paths=paths_value,steps=steps_value, reduce_variance=True)

np.shape(Stock_Matrix_GBM)[0]

Beta_list1 = lsm_policy(S0=S0_value, 
                                         K=K_value, r=r_value, paths=paths_value,
                                         sd=sd_value, T=T_value, steps=steps_value, 
                                         Stock_Matrix=Stock_Matrix_GBM,
                                         k=k_value,
                                         reduce_variance=True)
#plt.show()


# better regression equation of the 

def option_price_lsm(
        scoring_data: np.ndarray,
        num_steps,
        payoff,
        rate,
        expiry,
        Beta_list,
        k
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
                    XX=laguerre_polynomials_ind(path[step],k)
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


strike = 40
def payoff_func(_: float, s: float) -> float:
        return max(strike - s, 0.)


Stock_Matrix_GBM_test = SimulateGBM(S0=S0_value, r=r_value, sd=sd_value, T=T_value, 
paths=10000,steps=steps_value, reduce_variance=True)

option_price_lsm(scoring_data=Stock_Matrix_GBM_test, num_steps=steps_value-1,
payoff=payoff_func, rate=r_value, expiry=T_value, Beta_list=Beta_list1, k=k_value)




import matplotlib.pyplot as plt

prices_xaxis: np.ndarray = np.arange(40.0)

for i,step in enumerate(np.arange(0,48,8)):
    
    continue_price_list = []

    for price in prices_xaxis:
    
        XX=laguerre_polynomials_ind(price,k=4)
        continue_price: float = np.dot(XX, Beta_list1[step])
        continue_price_list.append((continue_price))
    ax=plt.subplot(3, 2, i+1)
    #plt.subplot(211)          
    plt.plot(prices_xaxis,continue_price_list,"bs")
    #plt.set_title('district_{}_change'.format(i))
    plt.title('Time Step=' + str(i))
    plt.xlabel("Price of the Underlying Asset (S)")
    plt.ylabel("Continuation value , V(S)")

plt.show()
for i, j in enumerate(np.arange(0,48,8)):
    print(i)