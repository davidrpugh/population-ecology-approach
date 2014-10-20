import sys
sys.path.append('../../')

import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

from package import sandbox

# number of female children of particular genotype
girls = sym.DeferredVector('f')

# number of male adults of particular genotype
men = sym.DeferredVector('M')

# Female signaling probabilities
female_signaling_probs = sym.var('dA, da')
dA, da = female_signaling_probs

# Male screening probabilities
male_screening_probs = sym.var('eG, eg')
eG, eg = male_screening_probs

# Female population by phenotype.
altruistic_girls = girls[0] + girls[2]
selfish_girls = girls[1] + girls[3]

# Conditional phenotype matching probabilities
true_altruistic_girls = dA * altruistic_girls
false_altruistic_girls = (1 - da) * selfish_girls
mistaken_for_altruistic_girls = (1 - eG) * false_altruistic_girls
altruist_adoption_pool = true_altruistic_girls + mistaken_for_altruistic_girls

true_selfish_girls = da * selfish_girls
false_selfish_girls = (1 - dA) * altruistic_girls
mistaken_for_selfish_girls = (1 - eg) * false_selfish_girls
selfish_adoption_pool = true_selfish_girls + mistaken_for_selfish_girls

SGA = true_altruistic_girls / altruist_adoption_pool
SGa = 1 - SGA
Sga = true_selfish_girls / selfish_adoption_pool
SgA = 1 - Sga

# females send precise signals, but males screen almost randomly
eps = 1e-3
params = {'c': 1.0, 'dA': 0.75, 'da': 0.75, 'eG': eps, 'eg': eps,
          'PiaA': 9.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}

# define an array of initial conditions S_A = mGA = fGA
N_initial = 5
eps = 1e-3
mGA0 = np.linspace(eps, 1 - eps, N_initial)

# create an instance of the model
example = sandbox.OneMaleTwoFemalesModel(params=params,
                                         SGA=SGA,
                                         Sga=Sga)

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
