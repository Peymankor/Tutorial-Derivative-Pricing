######### LSM ################

from LSM_policybased import OptimalExerciseLSM
from basis_fun import laguerre_polynomials_ind
import matplotlib.pyplot as plt
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

paths_number_train = 1000

LSMclass = OptimalExerciseLSM(spot_price=S0_value, payoff=payoff_func,expiry=T_value,
                                         rate=r_value, vol=sd_value,num_steps=steps_value)
train_data_v = LSMclass.GBMprice_training(num_paths_train=paths_number_train)

lsm_policy_v = LSMclass.train_LSM(training_data=train_data_v, 
                             num_paths_train=paths_number_train, K=K_value, k=k_value)


# lsm_policy_v


prices_xaxis: np.ndarray = np.arange(30,40,1)

#prices_xaxis


# for i,step in enumerate(np.arange(0,48,8)):
    
#     continue_price_list = []

#     for price in prices_xaxis:
    
#         XX=laguerre_polynomials_ind(price,k=4)
#         continue_price: float = np.dot(XX, lsm_policy_v[step])
#         continue_price_list.append((continue_price))
    
#     ax=plt.subplot(3, 2, i+1)
#     #plt.subplot(211)          
#     l1=ax.plot(prices_xaxis,continue_price_list,"r--")
#     #plt.set_title('district_{}_change'.format(i))
#     plt.title('Time Step=' + str(step))
#     #plt.xlabel("Price of the Underlying Asset (S)")
#     #plt.ylabel("Continuation value , V(S)")

# plt.show()


######################################################################################
######################################################################################
######################################################################################

from rl_optexe import OptimalExerciseRL

RLcalss = OptimalExerciseRL(
        spot_price=S0_value,
        payoff=payoff_func,
        expiry=T_value,
        rate=r_value,
        vol=sd_value,
        num_steps=steps_value
    )



# Neural Network Approximation

from rl_optexe import fitted_dql_put_option
from typing import Sequence, Tuple
from function_approx import DNNApprox, FunctionApprox
from pprint import pprint

TrainingDataType = Tuple[int, float, float]

num_training_paths = 10000
spot_price_frac_val = 0
dql_training_iters = 100000
num_scoring_paths = 10000

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


def continuation_curve(
        func: FunctionApprox[Tuple[float, float]],
        step: int,
        prices: Sequence[float],
        expiry: int,
        num_steps: int
    ) -> np.ndarray:
        t: float = step * expiry / num_steps
        return func.evaluate([(t, p) for p in prices])


scoring_data: np.ndarray = RLcalss.scoring_sim_data(
        num_paths=num_scoring_paths
    )
#continuation_curve(fdql, step=10,prices=prices_xaxis, expiry=1, num_steps=50 )

dql_opt_price: float = RLcalss.option_price(
        scoring_data=scoring_data,
        func=fdql
    )
print(f"DQL Option Price = {dql_opt_price:.3f}")
#scoring_data


prices_xaxis: np.ndarray = np.arange(20,40,1)



# for i,step in enumerate(np.arange(0,48,8)):
    
#     continue_price_list = []

#     for price in prices_xaxis:
    
#         XX=laguerre_polynomials_ind(price,k=4)
#         continue_price: float = np.dot(XX, lsm_policy_v[step])
#         continue_price_list.append((continue_price))
    
#     dql_cont = continuation_curve(fdql, step=step,
#             prices=prices_xaxis, expiry=1, num_steps=50 )
#     ax=plt.subplot(3, 2, i+1)
#     l1 = ax.plot(prices_xaxis, dql_cont, "b--", markersize=1, 
#             label = "continuation value")
#     l2 = ax.plot(prices_xaxis, continue_price_list,"r--",  
#             label = "Conditional Expectaion (Regresion)", linewidth=2)
#     #ax=plt.subplot(3, 2, i+1)
#     plt.title('Time Step=' + str(step+1))        
#     #l1=ax.plot(prices_xaxis,dql_cont,"r--")
#     #plt.set_title('district_{}_change'.format(i))
#     #plt.title('Time Step=' + str(step))
#     #plt.xlabel("Price of the Underlying Asset (S)")
#     #plt.ylabel("Continuation value , V(S)")

# plt.show()

########################################################
############### LSTD #####################################

from rl_optexe import fitted_lspi_put_option
from function_approx import DNNApprox, LinearFunctionApprox
import time

lspi_training_iters: int = 100
split_val: int = 1000

#start_time = time.time()

flspi: LinearFunctionApprox[Tuple[float, float]] = fitted_lspi_put_option(
        obj=RLcalss,
        strike=strike,
        expiry=T_value,
        training_data=training_data,
        training_iters=lspi_training_iters,
        split=split_val
    )
#print("lspi_training_iters: " + str(lspi_training_iters))
#print("split_val: " + str(split_val))
#print("--- %s seconds ---" % (time.time() - start_time))
############################################################

for i,step in enumerate(np.arange(0,48,8)):
    
    continue_price_list = []

    for price in prices_xaxis:
    
        XX=laguerre_polynomials_ind(price,k=4)
        continue_price: float = np.dot(XX, lsm_policy_v[step])
        continue_price_list.append((continue_price))
    
    dql_cont = continuation_curve(fdql, step=step,
            prices=prices_xaxis, expiry=1, num_steps=50 )
    
    lspi_cont = continuation_curve(flspi, step=step, prices=prices_xaxis, 
            expiry=T_value, num_steps=50)

    ax=plt.subplot(3, 2, i+1)
    l1 = ax.plot(prices_xaxis, dql_cont, "b--", markersize=1, 
            label = "Deep Q")
    l2 = ax.plot(prices_xaxis, continue_price_list,"r--",  
            label = "LSM", linewidth=2)
    l3 = ax.plot(prices_xaxis, lspi_cont,"g--",  
            label = "LSPI", linewidth=2)
    #ax=plt.subplot(3, 2, i+1)
    plt.title('Time Step=' + str(step+1))
    plt.xlabel("Price of the Underlying Asset (S)")
    plt.ylabel("Continuation value , V(S)")
    if i ==0:
                ax.legend(loc="upper right")        
    #l1=ax.plot(prices_xaxis,dql_cont,"r--")
    #plt.set_title('district_{}_change'.format(i))
    #plt.title('Time Step=' + str(step))
    #plt.xlabel("Price of the Underlying Asset (S)")
    #plt.ylabel("Continuation value , V(S)")

plt.show()

LSPI_opt_price: float = RLcalss.option_price(
        scoring_data=scoring_data,
        func=flspi
    )
print(f"LSPI Price = {LSPI_opt_price:.3f}")

#scoring_data
########################################################################
########################################################################

def binomial_put(S, K, T, r, vol, N):
    
    # Number of time steps
    dt = T/N
    
    # u value 
    u =  np.exp(vol * np.sqrt(dt))
    
    # 1/u value
    d = 1/u
    
    # 
    q = (np.exp(r * dt) - d)/(u - d)
    
    # probability of q
    
    # 
    C = {}
    
    #payoff function for put options    
    for m in range(0, N+1):
        C[(N, m)] = max(K - S * (u ** (2*m - N)), 0) 
    
    
    for k in range(N-1, -1, -1):

        for m in range(0,k+1):

            future_value = np.exp(-r * dt) * (q * C[(k+1, m+1)] + (1-q) * C[(k+1, m)])
            exercise_value =  max(K - S * (u ** (2*m-k)),0)
            C[(k, m)] = max(future_value, exercise_value)
    
    return C[(0,0)]
    #C

binomial_put(S=S0_value, K=K_value, T=T_value, r=r_value, vol=sd_value, N=steps_value)