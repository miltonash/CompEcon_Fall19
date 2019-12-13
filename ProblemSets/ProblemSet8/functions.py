###############################################################################
# File name: functions.py                                                     #
# Author: Milton Straw                                                        #
# Date created: 12/4/2019                                                     #
# Date last modified: 12/11/2019                                              #
# Instructor: Dr. Jason DeBacker                                              #
# Course: ECON 815                                                            #
###############################################################################

'''
This script defines the necessary functions used to solve my dynamic
programming problem outlined in Problem Set 7. Ths script will be called by
`execute.py`.
'''

# Import packages, listed alphabetically
import numba
import numpy as np
from scipy import interpolate
from scipy.optimize import fminbound


# Define utility function
def utility(pi, k, R):
    '''
    Utility is the log of weighted profits, alpha*pi, plus weighted quality,
    gamma*k. The choice variable, I, is constrained by revenue (as a function
    of quality), fixed operating cost, initial investment in equipment stock,
    and initial cash endowment.
    '''
    
    Iprime = R - C + I_0 + w_0
    R = np.log(k)
    k = I + 
    U = np.log((alpha * pi) + (gamma * k))

    return U


# Define Bellman operator function for value function iteration
# @numba.jit()
def bellman_operator(V, k_grid, params):
    '''
    The Bellman operator computes the updated value function TV on the grid
    points.
    '''
    beta, C, rho, alpha, gamma = params
    
    # Apply cubic interpolation to V
    V_func = interpolate.interp1d(k_grid, V, kind='cubic', fill_value='extrapolate')

    # Initialize array for operator and policy function
    TV = np.empty_like(V)
    optW = np.empty_like(TV)

    # == set TV[i] = max_I' { u(alpha pi + gamma k) + beta V(C',k')} == #
    for i, k in enumerate(k_grid):
        def objective(kprime):
            return - utility(pi, k, R) - beta * V_func(kprime)
        kprime_star = fminbound(objective, 1e-6, k - 1e-6)
        optK[i] = kprime_star
        TV[i] = - objective(kprime_star)
        
    return TV, optK
    
    
# Define Value Function Iteration
def vfi(V_params):
    '''
    VFtol     = scalar, tolerance required for value function to converge
    VFdist    = scalar, distance between last two value functions, as the
                distance closes we are getting closer to converging
    VFmaxiter = integer, maximum number of iterations for value function, to
                prevent an endless loop that might never return a solution
    V         = vector, the value functions at each iteration
    Vmat      = matrix, the value for each possible combination of I and I'
    Vstore    = matrix, stores V at each iteration 
    VFiter    = integer, current iteration number
    V_params  = tuple, contains parameters to pass into Bellman operator: beta, 
                C, rho, alpha, gamma
    TV        = vector, the value function after applying the Bellman operator
    PF        = vector, indicies of choices of I' for all I 
    VF        = vector, the "true" value function
    '''
    VFtol = 1e-5
    VFdist = 7.0 
    VFmaxiter = 2000 
    V = np.zeros(size_I)
    # Initialize Vstore array
    Vstore = np.zeros((size_I, VFmaxiter))
    VFiter = 1 
    V_params = (beta, C, rho, alpha, gamma)
    
    while VFdist > VFtol and VFiter < VFmaxiter:
        Vstore[:, VFiter] = V
        TV, optK = bellman_operator(V, k_grid, V_params)
        # Check distance
        VFdist = (np.absolute(V - TV)).max() 
        print('Iteration ', VFiter, ', distance = ', VFdist)
        V = TV
        VFiter += 1

    if VFiter < VFmaxiter:
        print('Value function converged after ', VFiter, ' iterations.')
    else:
        print('Value function failed to converge.')

    VF = V