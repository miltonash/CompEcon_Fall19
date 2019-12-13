###############################################################################
# File name: execute.py                                                       #
# Author: Milton Straw                                                        #
# Date created: 12/4/2019                                                     #
# Date last modified: 12/11/2019                                              #
# Instructor: Dr. Jason DeBacker                                              #
# Course: ECON 815                                                            #
###############################################################################

'''
This script solves my dynamic programming problem from Problem Set 7 by calling
functions defined in `functions.py`. It then plots the value function as a
function of a state variable.
'''

# Import packages and scripts, listed alphabetically
import functions
import matplotlib.pyplot as plt
import numpy as np

# Parameters
beta = 0.95
w_0 = 1000000
I_0 = 1500000
C = 250000
rho = 0.1
alpha = 0.7
gamma = 0.3

# Create grid for k
lb_m = 0.4
ub_m = 2.0
size_m = 100
m_grid = np.linspace(lb_m, ub_m, size_m)

# Create grid for depreciation delta
epsilon = np.random.uniform(0, 0.1)
delta_grid = rho + epsilon


'''
-------------------------------------------------------------------------------
Value Function Iteration
------------------------|------------------------------------------------------
Call the `vfi()` function from `functions.py` imported above.
-------------------------------------------------------------------------------
'''
vfi(V_params)


'''
-------------------------------------------------------------------------------
Extract the decision rule.
------------------------|------------------------------------------------------
opt  = vector, the optimal choice of c for each c
-------------------------------------------------------------------------------
'''
optI = k_grid - optK # optimal investment


'''
-------------------------------------------------------------------------------
Visualization
-------------------------------------------------------------------------------
'''
# Plot value function 
plt.figure()
plt.plot(k_grid[1:], VF[1:])
plt.xlabel('Quality as a Fn of Depreciated Equipment Stock')
plt.ylabel('Value Function')
plt.title('Value Function - deterministic investment in quality')
plt.show()