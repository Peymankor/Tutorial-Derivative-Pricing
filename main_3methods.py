
########### import packages, from python library

import numpy as np
from typing import Sequence, Tuple
from pprint import pprint
import matplotlib.pyplot as plt


########### Import method for Binomial method

from Bin_policy import BinPolicy

########### Import method for LSM

from LSM_policy import OptimalExerciseLSM
from basis_fun import laguerre_polynomials_ind

########### Import Method for Q-Learning

from RL_policy import fitted_dql_put_option, OptimalExerciseRL
from function_approx import DNNApprox, FunctionApprox

########### Initial Value for Put Option

S0_value = 36
r_value = 0.06
sd_value = 0.2
T_value = 1

steps_value = 50
paths_number_train = 100000
K_value = 40
k_value = 4

strike = 40

########### Define the payoff_func, and cont curve 

def payoff_func(_: float, s: float) -> float:
        return max(strike - s, 0.)


def continuation_curve(
        func: FunctionApprox[Tuple[float, float]],
        step: int,
        prices: Sequence[float],
        expiry: int,
        num_steps: int
    ) -> np.ndarray:
        t: float = step * expiry / num_steps
        return func.evaluate([(t, p) for p in prices])

######### Binomial Method ##############################

Binomial_class = BinPolicy(spot_price=S0_value, strike=strike, expiry=T_value,
                                rate=r_value, vol=sd_value, num_steps=steps_value)

cont_value_bin_list = Binomial_class.continuation_value_list()

Option_price_binomial = Binomial_class.option_price() 
print("Value of Option in Binomial Lattice Meethod: \nOption Price:")
print(Option_price_binomial)

############### LSM method ################################


LSM_class = OptimalExerciseLSM(spot_price=S0_value, payoff=payoff_func,expiry=T_value,
                                         rate=r_value, vol=sd_value,num_steps=steps_value)

train_data_v = LSM_class.GBMprice_training(num_paths_train=paths_number_train)

lsm_policy_v = LSM_class.train_LSM(training_data=train_data_v, 
                             num_paths_train=paths_number_train, K=K_value, k=k_value)


test_data_v = LSM_class.scoring_sim_data(num_paths_test=10000)

Option_price_LSM = LSM_class.option_price(scoring_data=test_data_v,Beta_list=lsm_policy_v,
                                            k=k_value)
print("Value of Option in LSM Method: \nOption Price:")
print(Option_price_LSM)

#################### Q-learning Method #########################


TrainingDataType = Tuple[int, float, float]

num_scoring_paths = 5000

num_training_paths = 5000
spot_price_frac_val = 0
dql_training_iters = 1000000

#num_scoring_paths = 10000


RL_calss = OptimalExerciseRL(
        spot_price=S0_value,
        payoff=payoff_func,
        expiry=T_value,
        rate=r_value,
        vol=sd_value,
        num_steps=steps_value
    )
training_data: Sequence[TrainingDataType] = RL_calss.training_sim_data(
        num_paths=num_training_paths,
        spot_price_frac=spot_price_frac_val
    )

fdql: DNNApprox[Tuple[float, float]] = fitted_dql_put_option(
        obj=RL_calss,
        strike=strike,
        expiry=T_value,
        training_data=training_data,
        training_iters=dql_training_iters
    )


scoring_data: np.ndarray = RL_calss.scoring_sim_data(
        num_paths=num_scoring_paths
    )

dql_opt_price: float = RL_calss.option_price(
        scoring_data=scoring_data,
        func=fdql
    )
print("Value of Option in RL Method: \nOption Price:")
print(dql_opt_price)


#######################################################################
prices_xaxis: np.ndarray = np.arange(20,41,1)

plt.figure(figsize=(15, 12))
plt.subplots_adjust(hspace=0.7)
#np.array([20,25,30,40,45,49])

for i,step in enumerate(np.array([20,25,30,40,45,48])):
    
    continue_price_list = []

    for price in prices_xaxis:
    
        XX=laguerre_polynomials_ind(price,k=4)
        continue_price: float = np.dot(XX, lsm_policy_v[step])
        continue_price_list.append((continue_price))
    
    dql_cont = continuation_curve(fdql, step=step,
            prices=prices_xaxis, expiry=1, num_steps=50)
    
    prices_x = [t[1] for t in cont_value_bin_list if t[0]==step]
    cont_values_binomial = [t[2] for t in cont_value_bin_list if t[0]==step]
    

    ax=plt.subplot(3, 2, i+1)
    l1 = ax.plot(prices_xaxis, dql_cont, "b--", markersize=1, 
            label = "Deep Q")
    l2 = ax.plot(prices_xaxis, continue_price_list,"r--",  
            label = "LSM", linewidth=2)
    l3 = ax.plot(prices_x, cont_values_binomial,"g--",  
            label = "Binomial", linewidth=2)
    #ax=plt.subplot(3, 2, i+1)
    plt.title('Time Step=' + str(step+1))
    plt.xlabel("Price of the Underlying Asset (S)")
    plt.ylabel("Continuation value , V(S)")
    if i ==0:
                ax.legend(loc="upper right")
                ax.set_ylim([0,30])
    ax.set_xlim([10,40])        
    #l1=ax.plot(prices_xaxis,dql_cont,"r--")
    #plt.set_title('district_{}_change'.format(i))
    #plt.title('Time Step=' + str(step))
    #plt.xlabel("Price of the Underlying Asset (S)")
    #plt.ylabel("Continuation value , V(S)")

plt.show()
