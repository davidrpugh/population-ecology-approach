import sys
sys.path.append('../../')

import numpy as np
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt

from package import model

# fix the model parameters
params = {'dA': 0.25, 'da': 0.25, 'eA': 0.0, 'ea': 0.0, 'PiaA': 6.0,
          'PiAA': 5.0, 'Piaa': 4.0, 'PiAa': 3.0}

# define an array of initial conditions
N = 5000
prng = np.random.RandomState(42)
initial_males = prng.dirichlet(np.ones(4), size=N)
initial_females = initial_males
initial_conditions = np.hstack((initial_males, initial_females))

# create an instance of the model
example = model.Model(params=params)

# storage container
results = np.empty((N, 8))

for i, initial_condition in enumerate(initial_conditions):

    # simulate the model to find the equilibrium
    tmp_traj = example.simulate(initial_condition, rtol=1e-4)

    # store the results
    results[i, :] = tmp_traj[:, -1]

### make the contour plot for
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
mGA = initial_conditions[:, 0]
mga = initial_conditions[:, 3]

# equilibrium share of female altruists
female_altruists = results[:, 4:7:2].sum(axis=1)

# interpolation grids
N = 100
grid = np.linspace(0, 1, N)
xi, yi = np.meshgrid(grid, grid)
interpolated_shares = griddata(mga, mGA, female_altruists, xi, yi)

# plot the interpolated values
mappable = ax.imshow(interpolated_shares.T, extent=[0, 1, 0, 1], origin='lower',
                     vmin=0, vmax=1)
ax.plot(grid, 1 - grid, 'k--')

# labels, title, etc
ax.set_xlabel('$f_{ga}$', fontsize=20)
ax.set_ylabel('$f_{GA}$', fontsize=20, rotation='horizontal')
fig_title = ("Equilibrium share of 'Altruistic' females\n" +
             "for various initial conditions")
ax.set_title(fig_title)

# add a color bar
fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/random-screening/imperfect_signaling_sweep_5.png')
plt.show()
