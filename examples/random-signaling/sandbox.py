import sys
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt

from package import model

# fix the initial condition
prng = np.random.RandomState(42)
initial_males = prng.dirichlet(np.ones(4), size=1)
initial_females = initial_males
initial_condition = np.hstack((initial_males, initial_females))

# define an array of screening probabilities
N = 21
screening_probs = np.linspace(0, 1, N)

# create an instance of the model
example = model.Model()

# storage container
results = np.empty((N, N, 8))

for i, eA in enumerate(screening_probs):
    for j, ea in enumerate(screening_probs):

        # females send random signals, but males screen imperfectly
        tmp_params = {'dA': 0.5, 'da': 0.5, 'eA': eA, 'ea': ea,
                      'PiaA': 6.0, 'PiAA': 5.0, 'Piaa': 4.0, 'PiAa': 3.0}

        # simulate the model to find the equilibrium
        example.params = tmp_params
        tmp_traj = example.simulate(initial_condition, T=1000)

        # store the results
        results[i, j, :] = tmp_traj[:, -1]


# create the plot for male population shares
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(results[:, :, 4], origin='lower', extent=[0, 1, 0, 1],
                  interpolation='gaussian', vmin=0, vmax=1)
axes[0, 0].set_ylabel('$e_A$', fontsize=20, rotation='horizontal')
axes[0, 0].set_title('$m_{GA}$', fontsize=20)

axes[0, 1].imshow(results[:, :, 5], origin='lower', extent=[0, 1, 0, 1],
                  interpolation='gaussian', vmin=0, vmax=1)
axes[0, 1].set_title('$m_{Ga}$', fontsize=20)

axes[1, 0].imshow(results[:, :, 6], origin='lower', extent=[0, 1, 0, 1],
                  interpolation='gaussian', vmin=0, vmax=1)
axes[1, 0].set_xlabel('$e_a$', fontsize=20, rotation='horizontal')
axes[1, 0].set_ylabel('$e_A$', fontsize=20, rotation='horizontal')
axes[1, 0].set_title('$m_{gA}$', fontsize=20)

axes[1, 1].imshow(results[:, :, 7], origin='lower', extent=[0, 1, 0, 1],
                  interpolation='gaussian', vmin=0, vmax=1)
axes[1, 1].set_xlabel('$e_a$', fontsize=20, rotation='horizontal')
axes[1, 1].set_title('$m_{ga}$', fontsize=20)

plt.show()


# create the plot for female population shares
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

axes[0, 0].imshow(results[:, :, 4], origin='lower', extent=[0, 1, 0, 1],
                  interpolation='gaussian', vmin=0, vmax=1)
axes[0, 0].set_ylabel('$e_A$', fontsize=20, rotation='horizontal')
axes[0, 0].set_title('$f_{GA}$', fontsize=20)

axes[0, 1].imshow(results[:, :, 5], origin='lower', extent=[0, 1, 0, 1],
                  interpolation='gaussian', vmin=0, vmax=1)
axes[0, 1].set_title('$f_{Ga}$', fontsize=20)

axes[1, 0].imshow(results[:, :, 6], origin='lower', extent=[0, 1, 0, 1],
                  interpolation='gaussian', vmin=0, vmax=1)
axes[1, 0].set_xlabel('$e_a$', fontsize=20, rotation='horizontal')
axes[1, 0].set_ylabel('$e_A$', fontsize=20, rotation='horizontal')
axes[1, 0].set_title('$f_{gA}$', fontsize=20)

axes[1, 1].imshow(results[:, :, 7], origin='lower', extent=[0, 1, 0, 1],
                  interpolation='gaussian', vmin=0, vmax=1)
axes[1, 1].set_xlabel('$e_a$', fontsize=20, rotation='horizontal')
axes[1, 1].set_title('$f_{ga}$', fontsize=20)

plt.show()
