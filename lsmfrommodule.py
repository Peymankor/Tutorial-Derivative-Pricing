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

from lsmfresh import lsm_policy

S0_value = 36
r_value = 0.06
sd_value = 0.2
T_value = 1
paths_value = 100000
steps_value = 50

K_value = 40
k_value = 4

Stock_Matrix_GBM = SimulateGBM(S0=S0_value, r=r_value, sd=sd_value, T=T_value, 
paths=paths_value,steps=steps_value, reduce_variance=True)


Beta_list1 = lsm_policy(S0=S0_value, 
                                         K=K_value, r=r_value, paths=paths_value,
                                         sd=sd_value, T=T_value, steps=steps_value, 
                                         Stock_Matrix=Stock_Matrix_GBM,
                                         k=k_value,
                                         reduce_variance=True)







num_steps = 48
expiry = 1


strike = 40
prices: np.ndarray = np.arange(0., strike + 0.1, 0.1)
prices
num_steps = 48
expiry = 1

x = []
y = []

for step in range(2,num_steps,1):
    
    print(step)
    cp = np.array([np.dot(laguerre_polynomials_ind(p,k=4), 
                Beta_list1[step]) for p in prices])

    ep = np.array([max(strike - p, 0) for p in prices])


    ll: Sequence[float] = [p for p, c, e in zip(prices, cp, ep)
                               if e > c]
    #print(ll)
    if len(ll) > 0:
        x.append(step * expiry / num_steps)
        y.append(max(ll))

final: Sequence[Tuple[float, float]] = \
        [(p, max(strike - p, 0)) for p in prices]
x.append(expiry)
y.append(max(p for p, e in final if e > 0))



from rl.gen_utils.plot_funcs import plot_list_of_curves


plot_list_of_curves(
        list_of_x_vals=[x],
        list_of_y_vals=[y],
        list_of_colors=["b"],
        list_of_curve_labels=["LSPI"],
        x_label="Time",
        y_label="Underlying Price",
        title="LSPI, Binary Tree Exercise Boundaries")