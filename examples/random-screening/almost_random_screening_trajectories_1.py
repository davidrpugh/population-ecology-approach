import sys
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt

from package import model

# females send precise signals, but males screen almost randomly
eps = 1e-3
params = {'c': 2.0, 'dA': 0.75, 'da': 0.75, 'eG': eps, 'eg': eps,
          'PiaA': 9.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}

# define an array of initial conditions S_A = mGA = fGA
N_initial = 75
eps = 1e-3
mGA0 = np.linspace(eps, 1 - eps, N_initial)

# create an instance of the model
example = model.Model(params=params,
                      solver_kwargs={'tol': 1e-12})

fig, axes = plt.subplots(1, 2, figsize=(12, 8))

for i in range(N_initial):

    # extract initial guess
    example.initial_condition = mGA0[i]
    tmp_traj = example.simulate(T=500)

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

# axes, labels, title, legend, etc
axes[0].set_ylim(0, 1)
axes[0].set_ylabel('Population shares', family='serif', fontsize=15)
axes[0].set_title('Males', family='serif', fontsize=20)
axes[1].set_title('Females', family='serif', fontsize=20)

# add a legend to the second subplot
legend_1 = axes[1].legend([f_GA, f_Ga, f_gA, f_ga],
                          [r'$GA$', r'$Ga$', r'$gA$', r'$ga$'],
                          loc=0, frameon=False,
                          bbox_to_anchor=(1.0, 1.0))

# want legend lines to be solid!
for line_obj in legend_1.legendHandles:
    line_obj.set_alpha(1.0)

# save and display the figure
#fig.savefig('../../images/random-screening/almost_random_screening_trajectories_1.png')
plt.show()
