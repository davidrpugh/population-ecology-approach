"""
It would seem that the solver generically converges to unstable equilibria
(i.e., there is always at least one eigenvalue greater than one. This means,
that at least for the moment, the only mechanism we have for finding stable
equilibria is to simulate the model from some initial condition.

"""
import sys
sys.path.append('../../')

import numpy as np

from package import model

# females signal randomly, and males screen imperfectly
eps = 1e-3
params = {'dA': 0.5, 'da': 0.5, 'eA': 1 - eps, 'ea': 1 - eps,
          'PiaA': 6.0, 'PiAA': 5.0, 'Piaa': 4.0, 'PiAa': 3.0}

# create an array of initial guesses for root finder
N = 250
prng = np.random.RandomState(42)
initial_males = prng.dirichlet(np.ones(4), size=N)
initial_females = initial_males
initial_guesses = np.hstack((initial_males, initial_females))

# create an instance of the model
example = model.Model(params=params,
                      solver_kwargs={'tol': 1e-12})

for i in range(N):

    # extract initial guess
    example.initial_guess = initial_guesses[i]

    # print the eigenvalues
    print example.eigenvalues

    if example.isstable:
        print "Equilibrium is stable!"
        print example.initial_guess
    elif not example.isstable and not example.isunstable:
        print "Stability indeterminate: there is a unit eigenvalue!"
        print example.initial_guess
    elif example.isunstable:
        print "Equilibrium is unstable!"
