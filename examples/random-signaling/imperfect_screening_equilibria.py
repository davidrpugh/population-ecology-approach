import sys
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt

from package import model

# females send random signals, and males screen imperfectly
eps = 1e-3
params = {'dA': 0.5, 'da': 0.5, 'eA': 1.0 - eps, 'ea': 1.0 - eps,
          'PiaA': 6.0, 'PiAA': 5.0, 'Piaa': 4.0, 'PiAa': 3.0}

# create an array of initial guesses for root finder
N = 100
prng = np.random.RandomState(42)
initial_males = prng.dirichlet(np.ones(4), size=N)
initial_females = initial_males
initial_guesses = np.hstack((initial_males, initial_females))

# create an instance of the model
example = model.Model(params=params,
                      solver_kwargs={'tol': 1e-12})

# initialize a storage container
steady_states = np.empty((N, 8))

# compute the steady state of the model for different initial guesses
fig, ax = plt.subplots()
ind = np.arange(-0.5, 7.5)

for i in range(N):

    # extract initial guess
    example.initial_guess = initial_guesses[i]

    # solve the nonlinear system
    tmp_result = example.steady_state

    if tmp_result.success and np.isclose(tmp_result.fun, 0.0, atol=1e-6):
        steady_states[i] = tmp_result.x
        bar_list = ax.bar(left=ind, height=tmp_result.x, width=1.0, alpha=0.05)

        # Set different color for Altruistic genotypes
        bar_list[0].set_color('r')
        bar_list[2].set_color('r')
        bar_list[4].set_color('r')
        bar_list[6].set_color('r')

    else:
        steady_states[i] = np.nan

    print 'Done with %i out of %i.' % (i, N)

# axes , labels, title, etc
ax.set_xlim(-1.0, 8.5)
ax.set_xticks(np.arange(8))
male_labels = ('$m_{GA}$', '$m_{Ga}$', '$m_{gA}$', '$m_{ga}$')
female_labels = ('$f_{GA}$', '$f_{Ga}$', '$f_{gA}$', '$f_{ga}$')
ax.set_xticklabels(male_labels + female_labels)
ax.set_ylim(0, 1)
ax.set_title('Putative equilibrium population shares', fontsize=20)

# save and display the figure
fig.savefig('../../images/random-signaling/imperfect_screening_equilibria.png')
plt.show()
