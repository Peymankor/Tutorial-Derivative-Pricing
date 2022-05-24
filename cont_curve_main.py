
import numpy as np

S0_value = 36
r_value = 0.06
sd_value = 0.2
T_value = 1
#paths_value = 100000
steps_value = 50

K_value = 40
k_value = 4

strike = 40


def payoff_func(_: float, s: float) -> float:
        return max(strike - s, 0.)


######### Binomial Method ##############################

from cont_curve_methods import binomial_put

value_of_binomial, cont_value_bin_list = binomial_put(S=S0_value, K=K_value, T=T_value, 
                                            r=r_value, vol=sd_value, N=steps_value)


############### LSM method ################################


from LSM_policybased import OptimalExerciseLSM

paths_number_train = 1000

LSMclass = OptimalExerciseLSM(spot_price=S0_value, payoff=payoff_func,expiry=T_value,
                                         rate=r_value, vol=sd_value,num_steps=steps_value)
train_data_v = LSMclass.GBMprice_training(num_paths_train=paths_number_train)

lsm_policy_v = LSMclass.train_LSM(training_data=train_data_v, 
                             num_paths_train=paths_number_train, K=K_value, k=k_value)

################################################################

from rl_optexe import fitted_dql_put_option, OptimalExerciseRL
from typing import Sequence, Tuple
from function_approx import DNNApprox, FunctionApprox
from pprint import pprint

TrainingDataType = Tuple[int, float, float]

num_training_paths = 5000
spot_price_frac_val = 0
dql_training_iters = 100000

#num_scoring_paths = 10000


RLcalss = OptimalExerciseRL(
        spot_price=S0_value,
        payoff=payoff_func,
        expiry=T_value,
        rate=r_value,
        vol=sd_value,
        num_steps=steps_value
    )
training_data: Sequence[TrainingDataType] = RLcalss.training_sim_data(
        num_paths=num_training_paths,
        spot_price_frac=spot_price_frac_val
    )

fdql: DNNApprox[Tuple[float, float]] = fitted_dql_put_option(
        obj=RLcalss,
        strike=strike,
        expiry=T_value,
        training_data=training_data,
        training_iters=dql_training_iters
    )

#######################################################################

from cont_curve_methods import continuation_curve
from basis_fun import laguerre_polynomials_ind
import matplotlib.pyplot as plt

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
    
    #lspi_cont = continuation_curve(flspi, step=step, prices=prices_xaxis, 
    #        expiry=T_value, num_steps=50)

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

#cont_value_binoimial

#[t[2] for t in cont_value_binoimial if t[0]==49]
#dql_cont = continuation_curve(fdql, step=step,
#            prices=prices_xaxis, expiry=1, num_steps=50)