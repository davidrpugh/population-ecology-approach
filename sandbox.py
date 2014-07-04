"""

TODO:
    
    1) Re-parameterize endogenous variables so that they are not defined for
    0, 1 and then look for a root. This should correspond to an interior 
    equilibrium. Use logistic function to perform transformation of vars.
    2) Set up non-linear optimization problem with inequality constraints and 
    solve for feasible point.
    3) Refactor code into an OOP framework.
    4) Need to automate stability analysis (which requires understanding 
    stability of non-hyperbolic fixed points!)
    5) Automate parameter sweep (but for which parameters?)
    7) Selfish Gene - Dawkins; Evolution and Theory of Games - J.M. Smith
    
"""
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, optimize
import sympy as sp


# solve the nonlinear system using optimization
constraints = [{'type':'eq', 'fun': get_steady_state, 'jac':get_steady_state_jac, 'args':(model_params,)}]
eps = 1e-1
bounds = [(eps, 1-eps), (eps, 1-eps), (eps, 1-eps), (eps, 1-eps), (eps, 1-eps), 
          (eps, 1-eps)]

result2 = optimize.minimize(lambda x, params: 1.0,
                            x0=initial_guess,
                            args=(model_params,),
                            method='SLSQP',
                            constraints=constraints,
                            bounds=bounds,
                            )

# check local stability? Need a check for stability with unit eigenvalues
# Consider imposing restrictions that require strictly positive frequencies!
mga = 1 - result.x[:3].sum()
fga = 1 - result.x[3:].sum()
steady_state_vals = np.hstack((result.x[:3], mga, result.x[3:], fga))
eig_vals, eig_vecs = linalg.eig(get_F_jac(steady_state_vals, model_params))
eig_vals_modulus = np.absolute(eig_vals)
print(eig_vals_modulus)
print(np.less(eig_vals_modulus, 1.0))

##### Multi-start for root finder ####

N = 100
prng = np.random.RandomState(42)
initial_males = prng.dirichlet(np.ones(4), size=N)
initial_females = prng.dirichlet(np.ones(4), size=N)

# array of initial guesses for root finder
initial_guesses = np.hstack((initial_males[:,:-1], initial_females[:,:-1])) 

steady_states = np.empty((N, 8))

# looping ove NumPy arrays is generally inefficient!
for i in range(N):
    
    # extract initial guess
    tmp_initial_guess = initial_guesses[i]
    
    # solve the nonlinear system
    tmp_result = optimize.root(get_steady_state, 
                               args=(model_params,), 
                               x0=tmp_initial_guess, 
                               jac=get_steady_state_jac, 
                               method='hybr', 
	                           tol=1e-12)
    
    if tmp_result.success:
        mga = 1 - tmp_result.x[:3].sum()
        fga = 1 - tmp_result.x[3:].sum()
        steady_states[i] = np.hstack((tmp_result.x[:3], mga, tmp_result.x[3:], fga)) 
    else:
        steady_states[i] = np.nan
   
fig, ax = plt.subplots()
ind = np.arange(-0.5, 7.5) 
#for i in range(N):
ax.bar(left=ind, height=steady_states[78], width=1.0, alpha=0.5)
male_labels = ('$m_{GA}$', '$m_{Ga}$', '$m_{gA}$', '$m_{ga}$')
female_labels = ('$f_{GA}$', '$f_{Ga}$', '$f_{gA}$', '$f_{ga}$')

ax.set_xlim(-1.0, 8.5)
ax.set_xticks(np.arange(8))
ax.set_xticklabels(male_labels + female_labels) 
ax.set_ylim(0,1)
plt.show()