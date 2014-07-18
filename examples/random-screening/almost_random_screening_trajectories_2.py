import sys
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt

from package import model

# females send precise signals, but males screen almost randomly
eps = 1e-3
params = {'dA': 1.0, 'da': 1.0, 'eA': eps, 'ea': eps, 'PiaA': 6.0, 'PiAA': 5.0,
          'Piaa': 4.0, 'PiAa': 3.0}

# create an array of initial guesses for root finder
N = 250
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

    # solve the nonlinear system
    tmp_result = example.steady_state

    if tmp_result.success and np.isclose(tmp_result.fun, 0):

        # extract initial guess
        example.initial_guess = tmp_result.x
        tmp_traj = example.simulate(initial_condition=example.initial_guess,
                                    T=500)

        # male allele trajectories
        m_GA, = axes[0].plot(tmp_traj[0], color='b', alpha=0.05)
        m_Ga, = axes[0].plot(tmp_traj[1], color='g', alpha=0.05)
        m_gA, = axes[0].plot(tmp_traj[2], color='r', alpha=0.05)
        m_ga, = axes[0].plot(tmp_traj[3], color='c', alpha=0.05)

        # female allele trajectories
        f_GA, = axes[1].plot(tmp_traj[4], color='b', alpha=0.05)
        f_Ga, = axes[1].plot(tmp_traj[5], color='g', alpha=0.05)
        f_gA, = axes[1].plot(tmp_traj[6], color='r', alpha=0.05)
        f_ga, = axes[1].plot(tmp_traj[7], color='c', alpha=0.05)

    else:
        continue

# axes, labels, title, legend, etc
axes[0].set_ylim(0, 1)
axes[0].set_ylabel('Population shares', family='serif', fontsize=15)
axes[0].set_title('Males', family='serif', fontsize=20)
legend_0 = axes[0].legend([m_GA, m_Ga, m_gA, m_ga],
                          [r'$m_{GA}$', r'$m_{Ga}$', r'$m_{gA}$', r'$m_{ga}$'],
                          loc=0, frameon=False)

# want legend lines to be solid!
for line_obj in legend_0.legendHandles:
    line_obj.set_alpha(1.0)

axes[1].set_ylim(0, 1)
axes[1].set_title('Females', family='serif', fontsize=20)
legend_1 = axes[1].legend([f_GA, f_Ga, f_gA, f_ga],
                          [r'$f_{GA}$', r'$f_{Ga}$', r'$f_{gA}$', r'$f_{ga}$'],
                          loc=0, frameon=False)

# want legend lines to be solid!
for line_obj in legend_1.legendHandles:
    line_obj.set_alpha(1.0)

fig.savefig('../../images/random-screening/almost_random_screenining_trajectories_2.png')
plt.show()
