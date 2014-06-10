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
    6) Set up private github repo to share code with Mark and Paul.
    7) Selfish Gene - Dawkins; Evolution and Theory of Games - J.M. Smith
    
"""
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg, optimize
import sympy as sp

# shares of male children
mGA, mGa, mgA, mga = sp.var('mGA, mGa, mgA, mga')

# shares of female children
fGA, fGa, fgA, fga = sp.var('fGA, fGa, fgA, fga')

# population shares by phenotype
mG = mGA + mGa
mg = mgA + mga
fA = fGA + fgA
fa = fGa + fga

# female signaling probabilities
dA, da = sp.var('dA, da') 

# male screening probabilities
eA, ea = sp.var('eA, ea')

# probability that male gets matched with preferred feature
SGA = (dA * fA) / (dA * fA + (1 - eA) * (1 - da) * fa)
SGa = 1 - SGA
Sga = (da * fa) / (da * fa + (1 - ea) * (1 - dA) * fA)
SgA = 1 - Sga

# payoff parameters (from a Prisoner's dilemma)
PiaA, PiAA, Piaa, PiAa = sp.var('PiaA, PiAA, Piaa, PiAa')

# recurrence relation for mGA
eqn1 = (mGA * SGA**2 * ((fGA + 0.5 * fgA) / fA) + 
        mGA * SGa**2 * ((0.5 * fGa + 0.25 * fga) / fa) + 
        2 * mGA * SGA * SGa * (((fGA + 0.5 * fgA) / fA) * (PiAa / (PiAa + PiaA)) + ((0.5 * fGa + 0.25 * fga) / fa) * (PiaA / (PiAa + PiaA))) + 
        mGa * SGA**2 * ((0.5 * fGA + 0.25 * fgA) / fA) +
        2 * mGa * SGA * SGa * (((0.5 * fGA + 0.25 * fgA) / fA) * (PiAa / (PiAa + PiaA))) + 
        mgA * SgA**2 * (0.5 * fGA / fA) + 
        mgA * Sga**2 * (0.25 * fGa / fa) + 
        2 * mgA * SgA * Sga * ((0.5 * fGA / fA) * (PiAa / (PiAa + PiaA)) + (0.25 * fGa / fa) * (PiaA / (PiAa + PiaA))) + 
        mga * SgA**2 * (0.25 * fGA / fA) +
        2 * mga * SgA * Sga * ((0.25 * fGA / fA) * (PiAa / (PiAa + PiaA))))

# recurrence relation for mGa
eqn2 = (mGA * SGa**2 * ((0.5 * fGa + 0.25 * fga) / fa) + 
        2 * mGA * SGA * SGa * (((0.5 * fGa + 0.25 * fga) / fa) * (PiaA / (PiAa + PiaA))) + 
        mGa * SGA**2 * ((0.5 * fGA + 0.25 * fgA) / fA) +
        mGa * SGa**2 * ((fGa + 0.5 * fga) / fa) + 
        2 * mGa * SGA * SGa * (((0.5 * fGA + 0.25 * fgA) / fA) * (PiAa / (PiAa + PiaA)) + ((fGa + 0.5 * fga) / fa) * (PiaA / (PiAa + PiaA))) + 
        mgA * Sga**2 * (0.25 * fGa / fa) + 
        2 * mgA * SgA * Sga * ((0.25 * fGa / fa) * (PiaA / (PiAa + PiaA))) +
        mga * SgA**2 * (0.25 * fGA / fA) + 
        mga * Sga**2 * (0.5 * fGa / fa) +
        2 * mga * SgA * Sga * ((0.25 * fGA / fA) * (PiAa / (PiAa + PiaA)) + (0.5 * fGa / fa) * (PiaA / (PiAa + PiaA))))

# recurrence relation for mgA                
eqn3 = (mGA * SGA**2 * (0.5 * fgA / fA) + 
        mGA * SGa**2 * (0.25 * fga / fa) + 
        2 * mGA * SGA * SGa * ((0.5 * fgA / fA) * (PiAa / (PiAa + PiaA)) + (0.25 * fga / fa) * (PiaA / (PiAa + PiaA))) +
        mGa * SGA**2 * (0.25 * fgA / fA) + 
        2 * mGa * SGA * SGa * ((0.25 * fgA / fA) * (PiAa / (PiAa + PiaA))) + 
        mgA * SgA**2 * ((0.5 * fGA + fgA) / fA) + 
        mgA * Sga**2 * ((0.25 * fGa + 0.5 * fga) / fa) + 
        2 * mgA * SgA * Sga * (((0.5 * fGA + fgA) / fA) * (PiAa / (PiAa + PiaA)) + ((0.25 * fGa + 0.5 * fga) / fa) * (PiaA / (PiAa + PiaA))) + 
        mga * SgA**2 * ((0.25 * fGA + 0.5 * fgA) / fA) + 
        2 * mga * SgA * Sga * (((0.25 * fGA + 0.5 * fgA) / fA) * (PiAa / (PiAa + PiaA))))

# recurrence relation for mga                 
eqn4 = (mGA * SGa**2 * (0.25 * fga / fa) + 
        2 * mGA * SGA * SGa * ((0.25 * fga / fa) * (PiaA / (PiAa + PiaA))) + 
        mGa * SGA**2 * (0.25 * fgA / fA) + 
        mGa * SGa**2 * (0.5 * fga / fa) + 
        2 * mGa * SGA * SGa * ((0.25 * fgA / fA) * (PiAa / (PiAa + PiaA)) + (0.5 * fga / fa) * (PiaA / (PiAa + PiaA))) + 
        mgA * Sga**2 * ((0.25 * fGa + 0.5 * fga) / fa) + 
        2 * mgA * SgA * Sga * (((0.25 * fGa + 0.5 * fga) / fa) * (PiaA / (PiAa + PiaA))) + 
        mga * SgA**2 * ((0.25 * fGA + 0.5 * fgA) / fA) + 
        mga * Sga**2 * ((0.5 * fGa + fga) / fa) + 
        2 * mga * SgA * Sga * (((0.25 * fGA + 0.5 * fgA) / fA) * (PiAa / (PiAa + PiaA)) + ((0.5 * fGa + fga) / fa) * (PiaA / (PiAa + PiaA))))
                   
# total female children in next generation
Nprime = ((mG * SGA**2 + mg * SgA**2) * 2 * PiAA +
          (mG * SGa**2 + mg * Sga**2) * 2 * Piaa + 
          2 * (mG * SGA * SGa + mg * SgA * Sga) * (PiAa + PiaA))
          
# recurrence relation for fGA
eqn5 = (mGA * SGA**2 * (((fGA + 0.5 * fgA) / fA) * (2 * PiAA / Nprime)) + 
        mGA * SGa**2 * (((0.5 * fGa + 0.25 * fga) / fa) * (2 * Piaa / Nprime)) + 
        2 * mGA * SGA * SGa * (((fGA + 0.5 * fgA) / fA) * (PiAa / Nprime) + ((0.5 * fGa + 0.25 * fga) / fa) * (PiaA / Nprime)) + 
        mGa * SGA**2 * (((0.5 * fGA + 0.25 * fgA) / fA) * (2 * PiAA / Nprime)) + 
        2 * mGa * SGA * SGa * (((0.5 * fGA + 0.25 * fgA) / fA) * (PiAa / Nprime)) + 
        mgA * SgA**2 * ((0.5 * fGA / fA) * (2 * PiAA / Nprime)) + 
        mgA * Sga**2 * ((0.25 * fGa / fa) * (2 * Piaa / Nprime)) +
        2 * mgA * SgA * Sga * ((0.5 * fGA / fA) * (PiAa / Nprime) + (0.25 * fGa / fa) * (PiaA / Nprime)) + 
        mga * SgA**2 * ((0.25 * fGA / fA) * (2 * PiAA / Nprime)) + 
        2 * mga * SgA * Sga * ((0.25 * fGA / fA) * (PiAa / Nprime)))

# recurrence relation for fGa                   
eqn6 = (mGA * SGa**2 * ((0.5 * fGa + 0.25 * fga) / fa) * (2 * Piaa / Nprime) +
        2 * mGA * SGA * SGa * (((0.5 * fGa + 0.25 * fga) / fa) * (PiaA / Nprime)) +
        mGa * SGA**2 * ((0.5 * fGA + 0.25 * fgA) / fA) * (2 * PiAA / Nprime) +
        mGa * SGa**2 * ((fGa + 0.5 * fga) / fa) * (2 * Piaa / Nprime) +
        2 * mGa * SGA * SGa * ((0.5 * fGA + 0.25 * fgA) / fA * PiAa / Nprime + (fGa + 0.5 * fga) / fa * PiaA / Nprime) +
        mgA * Sga**2 * 0.25 * fGa / fa * (2 * Piaa / Nprime) +
        2 * mgA * SgA * Sga * (0.25 * fGa / fa) * (PiaA / Nprime)	+
        mga * SgA**2 * (0.25 * fGA / fA) * (2 * PiAA / Nprime) +
        mga * Sga**2 * (0.5 * fGa / fa) * (2 * Piaa / Nprime) +
        2 * mga * SgA * Sga * ((0.25 * fGA / fA) * (PiAa / Nprime) + (0.5 * fGa / fa) * (PiaA / Nprime)))
        
# recurrence relation for fgA
eqn7 =(mGA * SGA**2 * ((0.5 * fgA / fA) * (2 * PiAA / Nprime)) +
       mGA * SGa**2 * ((0.25 * fga / fa) * (2 * Piaa / Nprime)) + 
       2 * mGA * SGA * SGa * ((0.5 * fgA / fA) * (PiAa / Nprime) + (0.25 * fga / fa) * (PiaA / Nprime)) +
       mGa * SGA**2 * ((0.25 * fgA / fA) * (2 * PiAA / Nprime)) + 
       2 * mGa * SGA * SGa * ((0.25 * fgA / fA) * (PiAa / Nprime)) + 
       mgA * SgA**2 * (((0.5 * fGA + fgA) / fA) * (2 * PiAA / Nprime)) + 
       mgA * Sga**2 * (((0.25 * fGa + 0.5 * fga) / fa) * (2 * Piaa / Nprime)) + 
       2 * mgA * SgA * Sga * (((0.5 * fGA + fgA) / fA) * (PiAa / Nprime) + ((0.25 * fGa + 0.5 * fga) / fa) * (PiaA / Nprime)) +
       mga * SgA**2 * (((0.25 * fGA + 0.5 * fgA) / fA) * (2 * PiAA / Nprime)) +
       2 * mga * SgA * Sga * (((0.25 * fGA + 0.5 * fgA) / fA) * (PiAa / Nprime)))
       
# recurrence relation for fga
eqn8 = (mGA * SGa**2 * ((0.25 * fga / fa) * (2 * Piaa / Nprime)) + 
        2 * mGA * SGA * SGa * ((0.25 * fga / fa) * (PiaA / Nprime)) + 
        mGa * SGA**2 * ((0.25 * fgA / fA) * (2 * PiAA / Nprime)) +
        mGa * SGa**2 * ((0.5 * fga / fa) * (2 * Piaa / Nprime)) + 
        2 * mGa * SGA * SGa * ((0.25 * fgA / fA) * (PiAa / Nprime) + (0.5 * fga / fa) * (PiaA / Nprime)) +
        mgA * Sga**2 * (((0.25 * fGa + 0.5 * fga) / fa) * (2 * Piaa / Nprime)) +
        2 * mgA * SgA * Sga * (((0.25 * fGa + 0.5 * fga) / fa) * (PiaA / Nprime)) + 
        mga * SgA**2 * (((0.25 * fGA + 0.5 * fgA) / fA) * (2 * PiAA / Nprime)) +
        mga * Sga**2 * (((0.5 * fGa + fga) / fa) * (2 * Piaa / Nprime)) +
        2 * mga * SgA * Sga * (((0.25 * fGA + 0.5 * fgA) / fA) * (PiAa / Nprime) + ((0.5 * fGa + fga) / fa) * (PiaA / Nprime)))

##### System of equations for simulation purposes #####

# define the system of nonlinear equations
endog_vars = sp.Matrix([mGA, mGa, mgA, mga, fGA, fGa, fgA, fga])
_symbolic_F = sp.Matrix([eqn1, eqn2, eqn3, eqn4, eqn5, eqn6, eqn7, eqn8])

# need to lambdify both the system of equations
args = (mGA, mGa, mgA, mga, fGA, fGa, fgA, fga, dA, da, eA, ea, PiaA, PiAA, Piaa, PiAa)
_numeric_F = sp.lambdify(args, _symbolic_F, modules=['numpy'])

def F(X, params):
    out = _numeric_F(*X, **params)
    return np.array(out).flatten()

# compute the Jacobian of the non-linear dynamical system   
_symbolic_F_jac = _symbolic_F.jacobian(endog_vars)

# wrap the Jacobian F (used for local stability analysis)
_numeric_F_jac = sp.lambdify(args, _symbolic_F_jac, modules=['numpy'])

def get_F_jac(X, params):
    out = _numeric_F_jac(*X, **params)
    return np.array(out)

##### Steady state computation #####

# for steady state only need to solve system of 6 equations
endog_vars = sp.Matrix([mGA, mGa, mgA, fGA, fGa, fgA])
_steady_state_symbolic = sp.Matrix([eqn1, eqn2, eqn3, eqn5, eqn6, eqn7]) - endog_vars
_steady_state_numeric = sp.lambdify(args, _steady_state_symbolic, modules=['numpy'])

def get_steady_state(X, params):
    
    # need to reassemble the vectors of endog variables
    mga = 1 - X[:3].sum()
    fga = 1 - X[3:].sum()
    endog_vars = np.hstack((X[:3], mga, X[3:], fga))
    
    out = _steady_state_numeric(*endog_vars, **params)
    return np.array(out).flatten()
                   
# compute the Jacobian of the steady state system   
_steady_state_symbolic_jac = _steady_state_symbolic.jacobian(endog_vars)

# wrap the system of equations and Jacobian for the solver
_steady_state_numeric_jac = sp.lambdify(args, _steady_state_symbolic_jac, modules=['numpy'])

def get_steady_state_jac(X, params):
    # need to reassemble the vectors of endog variables
    mga = 1 - X[:3].sum()
    fga = 1 - X[3:].sum()
    endog_vars = np.hstack((X[:3], mga, X[3:], fga))
    
    out = _steady_state_numeric_jac(*endog_vars, **params)
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

initial_guess = np.array([mGA0, mGa0, mgA0, fGA0, fGa0, fgA0])

# solve the nonlinear system
result = optimize.root(get_steady_state, 
                       args=(model_params,), 
                       x0=initial_guess, 
                       jac=get_steady_state_jac, 
                       method='hybr', 
                       tol=1e-12)
print(result.x)

# check local stability? Need a check for stability with unit eigenvalues
# Consider imposing restrictions that require strictly positive frequencies!
mga = 1 - result.x[:3].sum()
fga = 1 - result.x[3:].sum()
steady_state_vals = np.hstack((result.x[:3], mga, result.x[3:], fga))
eig_vals, eig_vecs = linalg.eig(get_F_jac(steady_state_vals, model_params))
eig_vals_modulus = np.absolute(eig_vals)
print(eig_vals_modulus)
print(np.less(eig_vals_modulus, 1.0))

##### Simulate a trajectory from the model #####

# For simulation need to use same initial conditions for both males and females
T = 100
mga0 = 1 - initial_guess[:3].sum()
fga0 = 1 - initial_guess[3:].sum()
initial_cond = np.hstack((initial_guess[:3], mga0, initial_guess[3:], fga0))
traj = np.maximum(initial_cond[:,np.newaxis], 0.0)

for t in range(T):
    step = F(traj[:,-1], model_params)
    traj = np.hstack((traj, step[:,np.newaxis]))
              
fig, axes = plt.subplots(1, 2, figsize=(12, 8))

axes[0].plot(traj[0], label=r'$m_{GA}$')
axes[0].plot(traj[1], label=r'$m_{Ga}$')
axes[0].plot(traj[2], label=r'$m_{gA}$')
axes[0].plot(traj[3], label=r'$m_{ga}$')

axes[0].set_ylim(0, 1)
axes[0].set_ylabel('Population shares', family='serif', fontsize=15)
axes[0].set_title('Males', family='serif', fontsize=20)
axes[0].legend(loc=0, frameon=False)

axes[1].plot(traj[4], label=r'$f_{GA}$')
axes[1].plot(traj[5], label=r'$f_{Ga}$')
axes[1].plot(traj[6], label=r'$f_{gA}$')
axes[1].plot(traj[7], label=r'$f_{ga}$')

axes[1].set_ylim(0, 1)
axes[1].set_title('Females', family='serif', fontsize=20)
axes[1].legend(loc=0, frameon=False)

#fig.suptitle('Initial condition "ga"', x=0.5, y=0.975, family='serif', fontsize=25)
#fig.savefig('initial_condition_4.png')

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