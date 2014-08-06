import sys
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt

from package import model

# fix the initial condition
prng = np.random.RandomState(123)
initial_males = prng.dirichlet(np.ones(4), size=1)
initial_females = initial_males
initial_condition = np.hstack((initial_males, initial_females))

# define an array of screening probabilities
N = 21
signaling_probs = np.linspace(0, 1, N)

# create an instance of the model
example = model.Model()

# storage container
results = np.empty((N, N, 8))

for i, dA in enumerate(signaling_probs):
    for j, da in enumerate(signaling_probs):

        # females imperfect signals, but males screen randomly
        tmp_params = {'dA': dA, 'da': da, 'eA': 0, 'ea': 0,
                      'PiaA': 6.0, 'PiAA': 5.0, 'Piaa': 4.0, 'PiAa': 3.0}

        # simulate the model to find the equilibrium
        example.params = tmp_params
        tmp_traj = example.simulate(initial_condition, rtol=1e-4)

        # store the results
        results[i, j, :] = tmp_traj[:, -1]


### create the plot for male population shares
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# colobar will be indexed off of this subplot
mappable = axes[0, 0].imshow(results[:, :, 0], origin='lower',
                             extent=[0, 1, 0, 1], interpolation='gaussian',
                             vmin=0, vmax=1)

# contour lines indicate the initial condition
CS0 = axes[0, 0].contour(results[:, :, 0], levels=[initial_condition[0, 0]],
                         colors='white', origin='lower', extent=[0, 1, 0, 1])
axes[0, 0].clabel(CS0, inline=1, fontsize=10)

# labels, title, etc
axes[0, 0].set_ylabel('$d_A$', fontsize=20, rotation='horizontal')
axes[0, 0].set_title('$m_{GA}$', fontsize=20)

# repeat for the other subplots
axes[0, 1].imshow(results[:, :, 1], origin='lower', extent=[0, 1, 0, 1],
                  interpolation='gaussian', vmin=0, vmax=1)
CS1 = axes[0, 1].contour(results[:, :, 1], levels=[initial_condition[0, 1]],
                         colors='white', origin='lower', extent=[0, 1, 0, 1])
axes[0, 1].clabel(CS1, inline=1, fontsize=10)
axes[0, 1].set_title('$m_{Ga}$', fontsize=20)

axes[1, 0].imshow(results[:, :, 2], origin='lower', extent=[0, 1, 0, 1],
                  interpolation='gaussian', vmin=0, vmax=1)
CS2 = axes[1, 0].contour(results[:, :, 2], levels=[initial_condition[0, 2]],
                         colors='white', origin='lower', extent=[0, 1, 0, 1])
axes[1, 0].clabel(CS2, inline=1, fontsize=10)

axes[1, 0].set_xlabel('$d_a$', fontsize=20, rotation='horizontal')
axes[1, 0].set_ylabel('$d_A$', fontsize=20, rotation='horizontal')
axes[1, 0].set_title('$m_{gA}$', fontsize=20)

axes[1, 1].imshow(results[:, :, 3], origin='lower', extent=[0, 1, 0, 1],
                  interpolation='gaussian', vmin=0, vmax=1)
CS3 = axes[1, 1].contour(results[:, :, 3], levels=[initial_condition[0, 3]],
                         colors='white', origin='lower', extent=[0, 1, 0, 1])
axes[1, 1].clabel(CS3, inline=1, fontsize=10)
axes[1, 1].set_xlabel('$d_a$', fontsize=20, rotation='horizontal')
axes[1, 1].set_title('$m_{ga}$', fontsize=20)

# add a color bar
fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

# title for the plot
title = 'Equilibrium population shares for\n various signaling probabilities'
fig.suptitle(title, x=0.5, y=0.98, fontsize=20)

# save and display the figure
fig.savefig('../../images/random-screening/imperfect_signaling_sweep_1.png')
plt.show()


### create the plot for female population shares
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# colobar will be indexed off of this subplot
mappable = axes[0, 0].imshow(results[:, :, 4], origin='lower',
                             extent=[0, 1, 0, 1], interpolation='gaussian',
                             vmin=0, vmax=1)

# contour lines indicate the initial condition
CS0 = axes[0, 0].contour(results[:, :, 4], levels=[initial_condition[0, 4]],
                         colors='white', origin='lower', extent=[0, 1, 0, 1])
axes[0, 0].clabel(CS0, inline=1, fontsize=10)

# labels, title, etc
axes[0, 0].set_ylabel('$d_A$', fontsize=20, rotation='horizontal')
axes[0, 0].set_title('$f_{GA}$', fontsize=20)

# repeat for the other subplots
axes[0, 1].imshow(results[:, :, 5], origin='lower', extent=[0, 1, 0, 1],
                  interpolation='gaussian', vmin=0, vmax=1)
CS1 = axes[0, 1].contour(results[:, :, 5], levels=[initial_condition[0, 5]],
                         colors='white', origin='lower', extent=[0, 1, 0, 1])
axes[0, 1].clabel(CS1, inline=1, fontsize=10)
axes[0, 1].set_title('$f_{Ga}$', fontsize=20)

axes[1, 0].imshow(results[:, :, 6], origin='lower', extent=[0, 1, 0, 1],
                  interpolation='gaussian', vmin=0, vmax=1)
CS2 = axes[1, 0].contour(results[:, :, 6], levels=[initial_condition[0, 6]],
                         colors='white', origin='lower', extent=[0, 1, 0, 1])
axes[1, 0].clabel(CS2, inline=1, fontsize=10)

axes[1, 0].set_xlabel('$d_a$', fontsize=20, rotation='horizontal')
axes[1, 0].set_ylabel('$d_A$', fontsize=20, rotation='horizontal')
axes[1, 0].set_title('$f_{gA}$', fontsize=20)

axes[1, 1].imshow(results[:, :, 7], origin='lower', extent=[0, 1, 0, 1],
                  interpolation='gaussian', vmin=0, vmax=1)
CS3 = axes[1, 1].contour(results[:, :, 7], levels=[initial_condition[0, 7]],
                         colors='white', origin='lower', extent=[0, 1, 0, 1])
axes[1, 1].clabel(CS3, inline=1, fontsize=10)
axes[1, 1].set_xlabel('$d_a$', fontsize=20, rotation='horizontal')
axes[1, 1].set_title('$f_{ga}$', fontsize=20)

# add a color bar
fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

# title for the plot
title = 'Equilibrium population shares for\n various signaling probabilities'
fig.suptitle(title, x=0.5, y=0.98, fontsize=20)

# save and display the figure
fig.savefig('../../images/random-screening/imperfect_signaling_sweep_2.png')
plt.show()


### create the plot for male population shares for allele A
fig, axes = plt.subplots(1, 2, figsize=(12, 6), squeeze=False)

# colobar will be indexed off of this subplot
mappable = axes[0, 0].imshow(results[:, :, :3:2].sum(axis=-1), origin='lower',
                             extent=[0, 1, 0, 1], interpolation='gaussian',
                             vmin=0, vmax=1)

# contour lines indicate the initial condition
CS0 = axes[0, 0].contour(results[:, :, :3:2].sum(axis=-1),
                         levels=[initial_condition[0, 0]],
                         colors='white', origin='lower', extent=[0, 1, 0, 1])
axes[0, 0].clabel(CS0, inline=1, fontsize=10)

# labels, title, etc
axes[0, 0].set_ylabel('$d_A$', fontsize=20, rotation='horizontal')
axes[0, 0].set_title('$m_{A}$', fontsize=20)

# repeat for the other subplots
axes[0, 1].imshow(results[:, :, 1:4:2].sum(axis=-1), origin='lower',
                  extent=[0, 1, 0, 1], interpolation='gaussian',
                  vmin=0, vmax=1)
CS1 = axes[0, 1].contour(results[:, :, 1:4:2].sum(axis=-1),
                         levels=[initial_condition[0, 1]],
                         colors='white', origin='lower', extent=[0, 1, 0, 1])
axes[0, 1].clabel(CS1, inline=1, fontsize=10)
axes[0, 1].set_title('$m_{a}$', fontsize=20)

# add a color bar
fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

# title for the plot
title = 'Equilibrium population shares for\n various signaling probabilities'
fig.suptitle(title, x=0.5, y=0.98, fontsize=20)

# save and display the figure
fig.savefig('../../images/random-screening/imperfect_signaling_sweep_3.png')
plt.show()


### create the plot for female population shares for allele a
fig, axes = plt.subplots(1, 2, figsize=(12, 6), squeeze=False)

# colobar will be indexed off of this subplot
mappable = axes[0, 0].imshow(results[:, :, 4:7:2].sum(axis=-1), origin='lower',
                             extent=[0, 1, 0, 1], interpolation='gaussian',
                             vmin=0, vmax=1)

# contour lines indicate the initial condition
CS0 = axes[0, 0].contour(results[:, :, 4:7:2].sum(axis=-1),
                         levels=[initial_condition[0, 0]],
                         colors='white', origin='lower', extent=[0, 1, 0, 1])
axes[0, 0].clabel(CS0, inline=1, fontsize=10)

# labels, title, etc
axes[0, 0].set_ylabel('$d_A$', fontsize=20, rotation='horizontal')
axes[0, 0].set_title('$f_{A}$', fontsize=20)

# repeat for the other subplots
axes[0, 1].imshow(results[:, :, 5:8:2].sum(axis=-1), origin='lower',
                  extent=[0, 1, 0, 1], interpolation='gaussian',
                  vmin=0, vmax=1)
CS1 = axes[0, 1].contour(results[:, :, 5:8:2].sum(axis=-1),
                         levels=[initial_condition[0, 1]],
                         colors='white', origin='lower', extent=[0, 1, 0, 1])
axes[0, 1].clabel(CS1, inline=1, fontsize=10)
axes[0, 1].set_title('$f_{a}$', fontsize=20)

# add a color bar
fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

# title for the plot
title = 'Equilibrium population shares for\n various signaling probabilities'
fig.suptitle(title, x=0.5, y=0.98, fontsize=20)

# save and display the figure
fig.savefig('../../images/random-screening/imperfect_signaling_sweep_4.png')
plt.show()
