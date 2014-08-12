import matplotlib.pyplot as plt
import numpy as np

from parameter_sweep import initial_conditions, signaling_probs

results = np.load("parameter_sweep.npy")


fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    almost_perfect_screening = results[:, :, 9, 9, i]
    female_altruists = almost_perfect_screening[:, :, 4:7:2].sum(axis=-1)

    # plot the equilibrium proportion of female altruists
    ax = fig.add_subplot(2, 3, i+1)
    mappable = ax.imshow(female_altruists, extent=[0, 1, 0, 1], origin='lower',
                         vmin=0, vmax=1)
    ax.set_xlabel('$d_a$', fontsize=20)
    ax.set_ylabel('$d_A$', fontsize=20, rotation='horizontal')

    # add a subplot title
    mGA0 = initial_condition[0]
    mga0 = initial_condition[3]
    sub_title = '$m_{GA,0}=f_{GA,0}= %g$,\n $m_{ga,0}=f_{ga,0}=%g$'
    ax.set_title(sub_title % (mGA0, mga0))

fig_title = ("Equilibrium share of 'altruistic' females, $f_A$,\n" +
             "for various signaling probabilities and almost perfect screening ($e_A={}, e_a={}$)")
fig.suptitle(fig_title.format(signaling_probs[9], signaling_probs[9]),
             fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/parameter-sweeps/signaling/figure-7.png')
plt.show()
