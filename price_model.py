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
    
    #return St[:,1:]
    return St

######## Test ##############################
S0_value = 36
r_value = 0.06
sd_value = 0.2
T_value = 1
paths_value = 100
steps_value = 10

Test_GBM = SimulateGBM(S0=S0_value, r=r_value, sd=sd_value, T=T_value, paths=paths_value,
steps=steps_value)

Test_GBM
########## Plot the Data ################

Time_steps = np.arange(0, steps_value+1)
plt.style.use('ggplot')
#plt.figure(figsize=(14, 10))
plt.plot(Time_steps, Test_GBM.T, '--')
plt.xticks(np.arange(0,11, 1.0))
plt.xlabel("Time Step", fontsize=16)
plt.ylabel("Underlying Price ($)", fontsize=16)
plt.title("100 paths simulation of GBM model with data in Table 1, Ex1 ", fontsize=20)
#plt.show()
#plt.xlabel.set_color('black')

plt.savefig('filename.png', dpi=300)
