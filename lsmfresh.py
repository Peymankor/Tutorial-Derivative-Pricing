from audioop import reverse
from dataclasses import dataclass, replace
from re import I
from typing import Callable, Dict, Sequence, Tuple, List
import numpy as np
from pandas import test
from scipy.stats import norm

from random import randrange
from numpy.polynomial.laguerre import lagval
from basis_fun import laguerre_polynomials, laguerre_polynomials_ind
from price_model import SimulateGBM

from rich import print, pretty
pretty.install()

import matplotlib.pyplot as plt



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
sd_value = 0.2
T_value = 1
paths_value = 100
steps_value = 50

K_value = 40
k_value = 4

Stock_Matrix_GBM = SimulateGBM(S0=S0_value, r=r_value, sd=sd_value, T=T_value, 
paths=paths_value,steps=steps_value, reduce_variance=True)

Stock_Matrix_GBM
Stock_Matrix_GBM
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



##################################

num_steps = 48
expiry = 1


strike = 40
prices: np.ndarray = np.arange(0., strike + 0.1, 0.1)
prices
num_steps = 48
expiry = 1

x = []
y = []

for step in range(num_steps):
    
    print(step)
    cp = np.array([np.dot(laguerre_polynomials_ind(p,k=4), 
                Beta_list1[step]) for p in prices])

    ep = np.array([max(strike - p, 0) for p in prices])


    ll: Sequence[float] = [p for p, c, e in zip(prices, cp, ep)
                               if e > c]
    print(ll)
    if len(ll) > 0:
        x.append(step * expiry / num_steps)
        y.append(max(ll))

final: Sequence[Tuple[float, float]] = \
        [(p, max(strike - p, 0)) for p in prices]
x.append(expiry)
y.append(max(p for p, e in final if e > 0))
y
x


from rl.gen_utils.plot_funcs import plot_list_of_curves


plot_list_of_curves(
        list_of_x_vals=[x],
        list_of_y_vals=[y],
        list_of_colors=["b"],
        list_of_curve_labels=["LSPI"],
        x_label="Time",
        y_label="Underlying Price",
        title="LSPI, Binary Tree Exercise Boundaries")


    #listcomp =listcont>listexe
    #print(listcomp)
    #priceofbound = np.where(listcomp==True)[0][0]

    #prices[priceofbound]

import matplotlib.pyplot as plt

###################################
x: List[float] = []
y: List[float] = []

strike = 40
prices: np.ndarray = np.arange(0., strike + 0.1, 0.1)

num_steps = 48
expiry = 1


for step in range(num_steps):

    contpriceinloop = []
    exerpriceinloop = []

    for price in prices:

        XX=laguerre_polynomials_ind(price,k=4)
        continue_price: float = np.dot(XX, Beta_list1[step])
        contpriceinloop.append((continue_price))
        exepriceinloop = np.array([max(strike - p, 0) for p in prices])


prices: np.ndarray = np.arange(0., strike + 0.1, 1)
listcont = np.array([np.dot(laguerre_polynomials_ind(p,k=4), 
                Beta_list1[step]) for p in prices])

listexe = np.array([max(strike - p, 0) for p in prices])

listcomp =listcont>listexe

priceofbound = np.where(listcomp==True)[0][0]

prices[priceofbound]

contpriceinloop = []
exerpriceinloop = []

step = 10
prices: np.ndarray = np.arange(0., strike + 0.1, 0.1)


prices: np.ndarray = np.arange(0., strike + 0.1, 5)

prices
contpriceinloop = []
exerpriceinloop = []

for price in prices:
    
    XX=laguerre_polynomials_ind(price,k=4)
    continue_price: float = np.dot(XX, Beta_list1[step])
    contpriceinloop.append((continue_price))
    exepriceinloop = np.array([max(strike - p, 0) for p in prices])
prices
contpriceinloop
exepriceinloop


prices



strike = 40
prices: np.ndarray = np.arange(0., strike + 0.1, 0.1)
exepriceinloop1 = np.array([max(strike - p, 0) for p in prices])

exepriceinloop1

def continuation_curve(
   
    t: float,
    prices: Sequence[float]
) -> np.ndarray:

    XX=laguerre_polynomials_ind(price,k=4)
    continue_price: float = np.dot(XX, Beta_list1[step])
    continue_price_list.append((continue_price))
    

    return func.evaluate([(t, p) for p in prices])


for step in range(num_steps):
        t: float = step * expiry / num_steps
        cp: np.ndarray = continuation_curve(
            func=func,
            t=t,
            prices=prices
        )
        ep: np.ndarray = exercise_curve(
            strike=strike,
            t=t,
            prices=prices
        )
        ll: Sequence[float] = [p for p, c, e in zip(prices, cp, ep)
                               if e > c]
        if len(ll) > 0:
            x.append(t)
            y.append(max(ll))
    final: Sequence[Tuple[float, float]] = \
        [(p, max(strike - p, 0)) for p in prices]
    x.append(expiry)
    y.append(max(p for p, e in final if e > 0))
    
    return x, y


















####################



def continuation_curve(
    func: FunctionApprox[Tuple[float, float]],
    t: float,
    prices: Sequence[float]
) -> np.ndarray:
    return func.evaluate([(t, p) for p in prices])


def exercise_curve(
    strike: float,
    t: float,
    prices: Sequence[float]
) -> np.ndarray:
    return np.array([max(strike - p, 0) for p in prices])


def put_option_exercise_boundary(
    func: FunctionApprox[Tuple[float, float]],
    expiry: float,
    num_steps: int,
    strike: float
) -> Tuple[Sequence[float], Sequence[float]]:
    x: List[float] = []
    y: List[float] = []
    prices: np.ndarray = np.arange(0., strike + 0.1, 0.1)
    for step in range(num_steps):
        t: float = step * expiry / num_steps
        cp: np.ndarray = continuation_curve(
            func=func,
            t=t,
            prices=prices
        )
        ep: np.ndarray = exercise_curve(
            strike=strike,
            t=t,
            prices=prices
        )
        ll: Sequence[float] = [p for p, c, e in zip(prices, cp, ep)
                               if e > c]
        if len(ll) > 0:
            x.append(t)
            y.append(max(ll))
    final: Sequence[Tuple[float, float]] = \
        [(p, max(strike - p, 0)) for p in prices]
    x.append(expiry)
    y.append(max(p for p, e in final if e > 0))
    return x, y