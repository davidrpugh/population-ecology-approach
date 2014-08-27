import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import parameter_sweep

# load the results file
results = np.load("parameter_sweep.npy")

initial_conditions = parameter_sweep.initial_conditions
signaling_probs = parameter_sweep.signaling_probs

# define the desired color map
cmap = mpl.cm.hot

### various screening probabilities and useless signaling
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    useless_signaling = results[0, 0, :, :, i]
    female_altruists = useless_signaling[:, :, 4:7:2].sum(axis=-1)

    # plot the equilibrium proportion of female altruists
    ax = fig.add_subplot(2, 3, i+1)
    mappable = ax.imshow(female_altruists, extent=[0, 1, 0, 1], origin='lower',
                         vmin=0, vmax=1, cmap=cmap)
    ax.set_xlabel('$e_a$', fontsize=20)
    ax.set_ylabel('$e_A$', fontsize=20, rotation='horizontal')

    # add a subplot title
    mGA0 = initial_condition[0]
    mga0 = initial_condition[3]
    sub_title = '$m_{GA,0}=f_{GA,0}= %g$,\n $m_{ga,0}=f_{ga,0}=%g$'
    ax.set_title(sub_title % (mGA0, mga0))

fig_title = ("Equilibrium share of 'altruistic' females, $f_A$,\n" +
             "for various screening probabilities and useless signaling ($d_A=d_a={}$)")
fig.suptitle(fig_title.format(signaling_probs[0]), fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/parameter-sweeps/screening/figure-0.png')
plt.show()


### various screening probabilities and imperfect signaling
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    idx = np.searchsorted(signaling_probs, 0.25)
    imperfect_signaling = results[idx, idx, :, :, i]
    female_altruists = imperfect_signaling[:, :, 4:7:2].sum(axis=-1)

    # plot the equilibrium proportion of female altruists
    ax = fig.add_subplot(2, 3, i+1)
    mappable = ax.imshow(female_altruists, extent=[0, 1, 0, 1], origin='lower',
                         vmin=0, vmax=1, cmap=cmap)
    ax.set_xlabel('$e_a$', fontsize=20)
    ax.set_ylabel('$e_A$', fontsize=20, rotation='horizontal')

    # add a subplot title
    mGA0 = initial_condition[0]
    mga0 = initial_condition[3]
    sub_title = '$m_{GA,0}=f_{GA,0}= %g$,\n $m_{ga,0}=f_{ga,0}=%g$'
    ax.set_title(sub_title % (mGA0, mga0))

fig_title = ("Equilibrium share of 'altruistic' females, $f_A$,\n" +
             "for various screening probabilities and imperfect signaling ($d_A=d_a={}$)")
fig.suptitle(fig_title.format(signaling_probs[idx]), fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/parameter-sweeps/screening/figure-1.png')
plt.show()


### various screening probabilities and random signaling
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    idx = np.searchsorted(signaling_probs, 0.5)
    random_signaling = results[idx, idx, :, :, i]
    female_altruists = random_signaling[:, :, 4:7:2].sum(axis=-1)

    # plot the equilibrium proportion of female altruists
    ax = fig.add_subplot(2, 3, i+1)
    mappable = ax.imshow(female_altruists, extent=[0, 1, 0, 1], origin='lower',
                         vmin=0, vmax=1, cmap=cmap)
    ax.set_xlabel('$e_a$', fontsize=20)
    ax.set_ylabel('$e_A$', fontsize=20, rotation='horizontal')

    # add a subplot title
    mGA0 = initial_condition[0]
    mga0 = initial_condition[3]
    sub_title = '$m_{GA,0}=f_{GA,0}= %g$,\n $m_{ga,0}=f_{ga,0}=%g$'
    ax.set_title(sub_title % (mGA0, mga0))

fig_title = ("Equilibrium share of 'altruistic' females, $f_A$,\n" +
             "for various screening probabilities and random signaling ($d_A=d_a={}$)")
fig.suptitle(fig_title.format(signaling_probs[idx]), fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/parameter-sweeps/screening/figure-2.png')
plt.show()


### various screening probabilities and imperfect signaling
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    idx = np.searchsorted(signaling_probs, 0.75)
    imperfect_signaling = results[idx, idx, :, :, i]
    female_altruists = imperfect_signaling[:, :, 4:7:2].sum(axis=-1)

    # plot the equilibrium proportion of female altruists
    ax = fig.add_subplot(2, 3, i+1)
    mappable = ax.imshow(female_altruists, extent=[0, 1, 0, 1], origin='lower',
                         vmin=0, vmax=1, cmap=cmap)
    ax.set_xlabel('$e_a$', fontsize=20)
    ax.set_ylabel('$e_A$', fontsize=20, rotation='horizontal')

    # add a subplot title
    mGA0 = initial_condition[0]
    mga0 = initial_condition[3]
    sub_title = '$m_{GA,0}=f_{GA,0}= %g$,\n $m_{ga,0}=f_{ga,0}=%g$'
    ax.set_title(sub_title % (mGA0, mga0))

fig_title = ("Equilibrium share of 'altruistic' females, $f_A$,\n" +
             "for various screening probabilities and imperfect signaling ($d_A=d_a={}$)")
fig.suptitle(fig_title.format(signaling_probs[idx]), fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/parameter-sweeps/screening/figure-3.png')
plt.show()


### various screening probabilities and perfect signaling
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    perfect_signaling = results[-1, -1, :, :, i]
    female_altruists = perfect_signaling[:, :, 4:7:2].sum(axis=-1)

    # plot the equilibrium proportion of female altruists
    ax = fig.add_subplot(2, 3, i+1)
    mappable = ax.imshow(female_altruists, extent=[0, 1, 0, 1], origin='lower',
                         vmin=0, vmax=1, cmap=cmap)
    ax.set_xlabel('$e_a$', fontsize=20)
    ax.set_ylabel('$e_A$', fontsize=20, rotation='horizontal')

    # add a subplot title
    mGA0 = initial_condition[0]
    mga0 = initial_condition[3]
    sub_title = '$m_{GA,0}=f_{GA,0}= %g$,\n $m_{ga,0}=f_{ga,0}=%g$'
    ax.set_title(sub_title % (mGA0, mga0))

fig_title = ("Equilibrium share of 'altruistic' females, $f_A$,\n" +
             "for various screening probabilities and perfect signaling ($d_A=d_a={}$)")
fig.suptitle(fig_title.format(signaling_probs[-1]), fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/parameter-sweeps/screening/figure-4.png')
plt.show()


### various screening probabilities and asymmetric signaling
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    lower_idx = np.searchsorted(signaling_probs, 0.25)
    upper_idx = np.searchsorted(signaling_probs, 0.75)
    asymmetric_signaling = results[upper_idx, lower_idx, :, :, i]
    female_altruists = asymmetric_signaling[:, :, 4:7:2].sum(axis=-1)

    # plot the equilibrium proportion of female altruists
    ax = fig.add_subplot(2, 3, i+1)
    mappable = ax.imshow(female_altruists, extent=[0, 1, 0, 1], origin='lower',
                         vmin=0, vmax=1, cmap=cmap)
    ax.set_xlabel('$e_a$', fontsize=20)
    ax.set_ylabel('$e_A$', fontsize=20, rotation='horizontal')

    # add a subplot title
    mGA0 = initial_condition[0]
    mga0 = initial_condition[3]
    sub_title = '$m_{GA,0}=f_{GA,0}= %g$,\n $m_{ga,0}=f_{ga,0}=%g$'
    ax.set_title(sub_title % (mGA0, mga0))

fig_title = ("Equilibrium share of 'altruistic' females, $f_A$,\n" +
             "for various screening probabilities and asymmetric signaling ($d_A={}, d_a={}$)")
fig.suptitle(fig_title.format(signaling_probs[upper_idx], signaling_probs[lower_idx]),
             fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/parameter-sweeps/screening/figure-5.png')
plt.show()


### various screening probabilities and asymmetric signaling
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    lower_idx = np.searchsorted(signaling_probs, 0.25)
    upper_idx = np.searchsorted(signaling_probs, 0.75)
    asymmetric_signaling = results[lower_idx, upper_idx, :, :, i]
    female_altruists = asymmetric_signaling[:, :, 4:7:2].sum(axis=-1)

    # plot the equilibrium proportion of female altruists
    ax = fig.add_subplot(2, 3, i+1)
    mappable = ax.imshow(female_altruists, extent=[0, 1, 0, 1], origin='lower',
                         vmin=0, vmax=1, cmap=cmap)
    ax.set_xlabel('$e_a$', fontsize=20)
    ax.set_ylabel('$e_A$', fontsize=20, rotation='horizontal')

    # add a subplot title
    mGA0 = initial_condition[0]
    mga0 = initial_condition[3]
    sub_title = '$m_{GA,0}=f_{GA,0}= %g$,\n $m_{ga,0}=f_{ga,0}=%g$'
    ax.set_title(sub_title % (mGA0, mga0))

fig_title = ("Equilibrium share of 'altruistic' females, $f_A$,\n" +
             "for various screening probabilities and asymmetric signaling ($d_A={}, d_a={}$)")
fig.suptitle(fig_title.format(signaling_probs[lower_idx], signaling_probs[upper_idx]),
             fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/parameter-sweeps/screening/figure-6.png')
plt.show()
