import sys
sys.path.append('../../')


import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from package import model

# fix the model parameters
params = {'dA': 0.25, 'da': 0.25, 'eA': 0.0, 'ea': 0.0, 'PiaA': 6.0,
          'PiAA': 5.0, 'Piaa': 4.0, 'PiAa': 3.0}

# define an array of initial conditions
N = 500
prng = np.random.RandomState(2)
pure_males = prng.dirichlet(np.ones(2), size=N)
initial_males = np.zeros((N, 4))
initial_males[:, 0:4:3] = pure_males
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
fGA = initial_conditions[:, 4]
fga = initial_conditions[:, -1]

# equilibrium share of female altruists
female_altruists = results[:, 4:7:2].sum(axis=1)

# interpolation grid
ni = 100
grid = np.linspace(0, 1, ni)
fgai, fGAi = np.meshgrid(grid, grid)

interpolated_shares = interpolate.griddata(points=(fga, fGA),
                                           values=female_altruists,
                                           xi=(fgai, fGAi),
                                           method='nearest',
                                           fill_value=np.nan,
                                           )

# plot the interpolated values
mappable = ax.imshow(np.where(fgai + fGAi <= 1.0, interpolated_shares, np.nan),
                     extent=[0, 1, 0, 1], origin='lower', vmin=0, vmax=1)
ax.plot(grid, 1 - grid, 'k--')

# labels, title, etc
ax.set_xlabel('$f_{ga}$', fontsize=20)
ax.set_ylabel('$f_{GA}$', fontsize=20, rotation='horizontal')
fig_title = ("Equilibrium share of 'Altruistic' females, $f_{A}$,\n" +
             "for various initial conditions")
ax.set_title(fig_title)

# add a color bar
fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/random-screening/imperfect_signaling_sweep_6.png')
plt.show()
