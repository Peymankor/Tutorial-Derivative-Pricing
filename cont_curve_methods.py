
import numpy as np
from typing import Sequence, Tuple
from function_approx import DNNApprox, FunctionApprox


######### Binomial Method ##############################

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
    #price_array = []
    future_value_list = []
    
    #payoff function for put options    
    for m in range(0, N+1):
        C[(N, m)] = max(K - S * (u ** (2*m - N)), 0) 
    
    
    for k in range(N-1, -1, -1):

        for m in range(0,k+1):

            future_value = np.exp(-r * dt) * (q * C[(k+1, m+1)] + (1-q) * C[(k+1, m)])
            exercise_value =  max(K - S * (u ** (2*m-k)),0)
            
            #price_array.append((k,S * (u ** (2*m-k))))
            future_value_list.append((k,S * (u ** (2*m-k)),future_value))

            C[(k, m)] = max(future_value, exercise_value)
    
    return C[(0,0)], future_value_list

#########################################################

#paths: np.ndarray = np.empty([num_paths, self.num_steps + 1])

from dataclasses import dataclass

@dataclass(frozen=True)
class BinPolicy:

    spot_price: float
    strike: float
    expiry: float
    rate: float
    vol: float
    num_steps: int

    def option_price(self) -> float:

        # Number of time steps
        dt = self.expiry/self.num_steps
    
        # u value 
        u =  np.exp(self.vol * np.sqrt(dt))
    
        # 1/u value
        d = 1/u
    
        # probability of q 
        q = (np.exp(self.rate * dt) - d)/(u - d)
    
        # probability of q
    
        # 
        C = {}
        #price_array = []
        
        #future_value_list = []
    
        #payoff function for put options    
        for m in range(0, self.num_steps+1):

            C[(self.num_steps, m)] = max(self.strike - 
                self.spot_price * (u ** (2*m - self.num_steps)), 0) 
    
    
        for k in range(self.num_steps-1, -1, -1):

            for m in range(0,k+1):

                future_value = np.exp(-self.rate * dt) * (q * C[(k+1, m+1)] + 
                                (1-q) * C[(k+1, m)])
                exercise_value =  max(self.strike - self.spot_price * (u ** (2*m-k)),0)
            

                C[(k, m)] = max(future_value, exercise_value)
    
        return C[(0,0)]


    def continuation_value_list(self):

        # Number of time steps
        dt = self.expiry/self.num_steps
    
        # u value 
        u =  np.exp(self.vol * np.sqrt(dt))
    
        # 1/u value
        d = 1/u
    
        # probability of q 
        q = (np.exp(self.rate * dt) - d)/(u - d)
    
        # probability of q
    
        # 
        C = {}
        #price_array = []
        
        future_value_list = []
    
        #payoff function for put options    
        for m in range(0, self.num_steps+1):

            C[(self.num_steps, m)] = max(self.strike - 
                self.spot_price * (u ** (2*m - self.num_steps)), 0) 
    
    
        for k in range(self.num_steps-1, -1, -1):

            for m in range(0,k+1):

                exercise_value =  max(self.strike - self.spot_price * (u ** (2*m-k)),0)

                future_value = np.exp(-self.rate * dt) * (q * C[(k+1, m+1)] + (1-q) * C[(k+1, m)])
            
                future_value_list.append((k,self.spot_price* (u ** (2*m-k)),future_value))

                C[(k, m)] = max(future_value, exercise_value)
        
        return future_value_list



import numpy as np
test = BinPolicy(spot_price=36, strike=40, expiry=1,
            rate=0.06, vol=0.2, num_steps=5)


test.option_price()
test.continuation_value_list()
####################### Q-Learning Method ####################

def continuation_curve(
        func: FunctionApprox[Tuple[float, float]],
        step: int,
        prices: Sequence[float],
        expiry: int,
        num_steps: int
    ) -> np.ndarray:
        t: float = step * expiry / num_steps
        return func.evaluate([(t, p) for p in prices])


###########################################################################


S0_value = 36
r_value = 0.06
sd_value = 0.2
T_value = 1
#paths_value = 100000
steps_value = 5

K_value = 40
k_value = 4

strike = 40

ddd , v=binomial_put(S=S0_value, K=K_value, T=T_value, r=r_value,
                vol=sd_value, N=5)

ddd
v