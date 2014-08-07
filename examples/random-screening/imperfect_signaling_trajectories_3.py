import sys
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt

from package import model

# females send precise signals, but males screen randomly
params = {'dA': 0.25, 'da': 0.25, 'eA': 0.0, 'ea': 0.0, 'PiaA': 6.0, 'PiAA': 5.0,
          'Piaa': 4.0, 'PiAa': 3.0}

# create an array of initial guesses for root finder
N = 500
prng = np.random.RandomState(42)
initial_males = prng.dirichlet(np.ones(4), size=N)
initial_females = initial_males
initial_guesses = np.hstack((initial_males, initial_females))

# create an instance of the model
example = model.Model(params=params,
                      solver_kwargs={'tol': 1e-12})

fig, axes = plt.subplots(1, 2, figsize=(12, 8))

for i in range(N):

    # extract initial guess
    example.initial_guess = initial_guesses[i]
    tmp_traj = example.simulate(initial_condition=example.initial_guess, T=150)

    # male allele trajectories
    m_A, = axes[0].plot(tmp_traj[0:3:2].sum(axis=0), color='b', alpha=0.05)
    m_a, = axes[0].plot(tmp_traj[1:4:2].sum(axis=0), color='r', alpha=0.05)

    # female allele trajectories
    f_A, = axes[1].plot(tmp_traj[4:7:2].sum(axis=0), color='b', alpha=0.05)
    f_a, = axes[1].plot(tmp_traj[5::2].sum(axis=0), color='r', alpha=0.05)

# axes, labels, title, legend, etc
axes[0].set_ylim(0, 1)
axes[0].set_ylabel('Population shares', family='serif', fontsize=15)
axes[0].set_title('Males', family='serif', fontsize=20)
legend_0 = axes[0].legend([m_A, m_a], [r'$m_{A}$', r'$m_{a}$'], loc=0,
                          frameon=False)

# want legend lines to be solid!
for line_obj in legend_0.legendHandles:
    line_obj.set_alpha(1.0)

axes[1].set_ylim(0, 1)
axes[1].set_title('Females', family='serif', fontsize=20)
legend_1 = axes[1].legend([f_A, f_a], [r'$f_{A}$', r'$f_{a}$'], loc=0,
                          frameon=False)

# want legend lines to be solid!
for line_obj in legend_1.legendHandles:
    line_obj.set_alpha(1.0)

# save and display the figure
fig.savefig('../../images/random-screening/imperfect_signaling_trajectories_3.png')
plt.show()
