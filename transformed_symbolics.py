from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, optimize
import sympy as sp

import model

##### Steady state computation #####

mod = model.Model()

# define some new variables
sp.var('mGA_tilde, mGa_tilde, mgA_tilde, mga_tilde')
sp.var('fGA_tilde, fGa_tilde, fgA_tilde, fga_tilde')

# define a transformation of variables using the logistic function
logistic_transform = {'mGA':(1 / (1 + sp.exp(-mGA_tilde))),
                      'mGa':(1 / (1 + sp.exp(-mGa_tilde))),
                      'mgA':(1 / (1 + sp.exp(-mgA_tilde))),
                      'mga':(1 / (1 + sp.exp(-mga_tilde))),
                      'fGA':(1 / (1 + sp.exp(-fGA_tilde))),
                      'fGa':(1 / (1 + sp.exp(-fGa_tilde))), 
                      'fgA':(1 / (1 + sp.exp(-fgA_tilde))),
                      'fga':(1 / (1 + sp.exp(-fga_tilde)))}

new_endog_vars = sp.Matrix([mGA_tilde, mGa_tilde, mgA_tilde, mga_tilde, fGA_tilde, fGa_tilde, fgA_tilde, fga_tilde])
new_steady_state_system = mod._symbolic_simulation_system.subs(logistic_transform)
new_args = (mGA_tilde, mGa_tilde, mgA_tilde, mga_tilde, fGA_tilde, fGa_tilde, fgA_tilde, fga_tilde, dA, da, eA, ea, PiaA, PiAA, Piaa, PiAa)

_steady_state_symbolic = new_steady_state_system - new_endog_vars
_steady_state_numeric = sp.lambdify(new_args, _steady_state_symbolic, modules=['numpy'])

def get_steady_state(X, params):    
    out = _steady_state_numeric(*X, **params)
    return np.array(out).flatten()
                   
# compute the Jacobian of the steady state system   
_steady_state_symbolic_jac = _steady_state_symbolic.jacobian(new_endog_vars)

# wrap the system of equations and Jacobian for the solver
_steady_state_numeric_jac = sp.lambdify(new_args, _steady_state_symbolic_jac, modules=['numpy'])

def get_steady_state_jac(X, params):    
    out = _steady_state_numeric_jac(*X, **params)
    return np.array(out)
    
# dictionary of parameter values
model_params = {'dA':0.25, 'da':0.75, 'eA':0.25, 'ea':0.5, 'PiaA':6.0, 'PiAA':5.0, 
                'Piaa':4.0, 'PiAa':3.0}
  
# initial guess for the solver       
mGA0 = 0.05
mGa0 = 0.05
mgA0 = 0.05
mga0 = 1 - mGA0 - mGa0 - mgA0
 
fGA0 = mGA0
fGa0 = mGa0
fgA0 = mgA0
fga0 = 1 - fGA0 - fGa0 - fgA0

initial_guess = np.array([mGA0, mGa0, mgA0, mga0, fGA0, fGa0, fgA0, fga0])
transform_initial_guess = np.log(initial_guess / (1 - initial_guess))

# solve the nonlinear system
result = optimize.root(get_steady_state, 
                       args=(model_params,), 
                       x0=transform_initial_guess, 
                       jac=get_steady_state_jac, 
                       method='hybr', 
                       tol=1e-12)
print(result.x)

def feasiblity_constraint_1(X, params):
    """Require male population shares to sum to 1."""
    inverse_transform = 1 / (1 + np.exp(-X))
    return np.sum(inverse_transform[:4]) - 1

def feasiblity_constraint_2(X, params):
    """Require female population shares to sum to 1."""
    inverse_transform = 1 / (1 + np.exp(-X))
    return np.sum(inverse_transform[4:]) - 1

eq_cons = [{'type':'eq', 'fun':feasiblity_constraint_1, 'args':(model_params,)},
           {'type':'eq', 'fun':feasiblity_constraint_2, 'args':(model_params,)}]

# solve the nonlinear system
result2 = optimize.minimize(lambda X, params: np.sum(get_steady_state(X, params)**2), 
                           args=(model_params,), 
                           x0=transform_initial_guess, 
                           method='SLSQP',
                           constraints=eq_cons, 
                           tol=1e-12)
print(result2.x)           
# check local stability? Need a check for stability with unit eigenvalues
# Consider imposing restrictions that require strictly positive frequencies!
# mga = 1 - result.x[:3].sum()
# fga = 1 - result.x[3:].sum()
# steady_state_vals = np.hstack((result.x[:3], mga, result.x[3:], fga))
# eig_vals, eig_vecs = linalg.eig(get_F_jac(steady_state_vals, model_params))
# eig_vals_modulus = np.absolute(eig_vals)
# print(eig_vals_modulus)
# print(np.less(eig_vals_modulus, 1.0))

# ##### Simulate a trajectory from the model #####

# # For simulation need to use same initial conditions for both males and females
# T = 100
# mga0 = 1 - initial_guess[:3].sum()
# fga0 = 1 - initial_guess[3:].sum()
# initial_cond = np.hstack((initial_guess[:3], mga0, initial_guess[3:], fga0))
# traj = np.maximum(initial_cond[:,np.newaxis], 0.0)

# for t in range(T):
#     step = F(traj[:,-1], model_params)
#     traj = np.hstack((traj, step[:,np.newaxis]))
              
# fig, axes = plt.subplots(1, 2, figsize=(12, 8))

# axes[0].plot(traj[0], label=r'$m_{GA}$')
# axes[0].plot(traj[1], label=r'$m_{Ga}$')
# axes[0].plot(traj[2], label=r'$m_{gA}$')
# axes[0].plot(traj[3], label=r'$m_{ga}$')

# axes[0].set_ylim(0, 1)
# axes[0].set_ylabel('Population shares', family='serif', fontsize=15)
# axes[0].set_title('Males', family='serif', fontsize=20)
# axes[0].legend(loc=0, frameon=False)

# axes[1].plot(traj[4], label=r'$f_{GA}$')
# axes[1].plot(traj[5], label=r'$f_{Ga}$')
# axes[1].plot(traj[6], label=r'$f_{gA}$')
# axes[1].plot(traj[7], label=r'$f_{ga}$')

# axes[1].set_ylim(0, 1)
# axes[1].set_title('Females', family='serif', fontsize=20)
# axes[1].legend(loc=0, frameon=False)

# #fig.suptitle('Initial condition "ga"', x=0.5, y=0.975, family='serif', fontsize=25)
# #fig.savefig('initial_condition_4.png')

# ##### Multi-start for root finder ####

# N = 100
# prng = np.random.RandomState(42)
# initial_males = prng.dirichlet(np.ones(4), size=N)
# initial_females = prng.dirichlet(np.ones(4), size=N)

# # array of initial guesses for root finder
# initial_guesses = np.hstack((initial_males[:,:-1], initial_females[:,:-1])) 

# steady_states = np.empty((N, 8))

# # looping ove NumPy arrays is generally inefficient!
# for i in range(N):
    
#     # extract initial guess
#     tmp_initial_guess = initial_guesses[i]
    
#     # solve the nonlinear system
#     tmp_result = optimize.root(get_steady_state, 
#                                args=(model_params,), 
#                                x0=tmp_initial_guess, 
#                                jac=get_steady_state_jac, 
#                                method='hybr', 
#                            tol=1e-12)
    
#     if tmp_result.success:
#         mga = 1 - tmp_result.x[:3].sum()
#         fga = 1 - tmp_result.x[3:].sum()
#         steady_states[i] = np.hstack((tmp_result.x[:3], mga, tmp_result.x[3:], fga)) 
#     else:
#         steady_states[i] = np.nan
   
# fig, ax = plt.subplots()
# ind = np.arange(-0.5, 7.5) 
# #for i in range(N):
# ax.bar(left=ind, height=steady_states[78], width=1.0, alpha=0.5)
# male_labels = ('$m_{GA}$', '$m_{Ga}$', '$m_{gA}$', '$m_{ga}$')
# female_labels = ('$f_{GA}$', '$f_{Ga}$', '$f_{gA}$', '$f_{ga}$')

# ax.set_xlim(-1.0, 8.5)
# ax.set_xticks(np.arange(8))
# ax.set_xticklabels(male_labels + female_labels) 
# ax.set_ylim(0,1)
# plt.show()