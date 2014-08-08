import sys
sys.path.append('../../')


import matplotlib.pyplot as plt
import numpy as np

from package import model

# define an array of initial conditions S = mGA = fGA
N = 21
mGA0 = np.linspace(0, 1, N).reshape((N, 1))
mga0 = 1 - mGA0
initial_males = np.hstack((mGA0, np.zeros((N, 2)), mga0))
initial_females = initial_males
initial_conditions = np.hstack((initial_males, initial_females))

# define some signaling probs
signaling_probs = np.linspace(0, 1, N)

# storage container
results = np.empty((N, N, 8))

for i, dA in enumerate(signaling_probs):
    for j, initial_condition in enumerate(initial_conditions):

        # fix the model parameters
        tmp_params = {'dA': dA, 'da': dA, 'eA': 0.0, 'ea': 0.0,
                      'PiaA': 6.0, 'PiAA': 5.0, 'Piaa': 4.0, 'PiAa': 3.0}

        # create an instance of the model
        tmp_example = model.Model(params=tmp_params)

        # simulate the model to find the equilibrium
        tmp_traj = tmp_example.simulate(initial_condition, rtol=1e-4)

        # store the results
        results[i, j, :] = tmp_traj[:, -1]

### make the contour plot for
fig, ax = plt.subplots(1, 1, figsize=(8, 6))

for i, dA in enumerate(signaling_probs[::4]):

    # equilibrium share of female altruists
    equilibrium_female_altruists = results[i, :, 4:7:2].sum(axis=1)

    ax.plot(mGA0.flatten(), equilibrium_female_altruists, 'o-', alpha=0.5,
            label='$d_A=d_a=%g$' % dA)

# labels, title, legend, etc
ax.grid('on')
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel('$m_{GA}=f_{GA}$', fontsize=20)
ax.set_ylabel('$f_{A}$', fontsize=20, rotation='horizontal')

fig_title = ("Equilibrium share of 'Altruistic' females, $f_{A}$,\n" +
             "for various initial conditions")
ax.set_title(fig_title)

ax.legend(loc=0, frameon=False, bbox_to_anchor=(1.0, 1.0))

fig.savefig('../../images/random-screening/imperfect_signaling_sweep_6.png',
            bbox_inches='tight')
plt.show()

### make the contour plot for
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
fGA = initial_conditions[:, 4]

# equilibrium share of female altruists
female_altruists = results[:, :, 4:7:2].sum(axis=-1)

# plot the interpolated values
mappable = ax.imshow(female_altruists, extent=[0, 1, 0, 1], origin='lower',
                     vmin=0, vmax=1)

# labels, title, etc
ax.set_xlabel('$d_A=d_a$', fontsize=20)
ax.set_ylabel('$f_{GA}$', fontsize=20, rotation='horizontal')
fig_title = ("Equilibrium share of 'Altruistic' females, $f_{A}$,\n" +
             "for various initial conditions and signaling probs")
ax.set_title(fig_title)

# add a color bar
fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/random-screening/imperfect_signaling_sweep_7.png')
plt.show()
