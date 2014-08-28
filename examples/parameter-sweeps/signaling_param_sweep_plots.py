import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import parameter_sweep
reload(parameter_sweep)

# load the results file
results = np.load("parameter_sweep.npy")

initial_conditions = parameter_sweep.initial_conditions
screening_probs = parameter_sweep.screening_probs

# define the desired color map
cmap = mpl.cm.hot

### various signaling probabilities and useless screening
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    useless_screening = results[:, :, 0, 0, i]
    female_altruists = useless_screening[:, :, 4:7:2].sum(axis=-1)

    # plot the equilibrium proportion of female altruists
    ax = fig.add_subplot(2, 3, i+1)
    mappable = ax.imshow(female_altruists, extent=[0, 1, 0, 1], origin='lower',
                         vmin=0, vmax=1, cmap=cmap)
    ax.set_xlabel('$d_a$', fontsize=20)
    ax.set_ylabel('$d_A$', fontsize=20, rotation='horizontal')

    # add a subplot title
    mGA0 = initial_condition[0]
    mga0 = initial_condition[3]
    sub_title = '$m_{GA,0}=f_{GA,0}= %g$,\n $m_{ga,0}=f_{ga,0}=%g$'
    ax.set_title(sub_title % (mGA0, mga0))

fig_title = ("Equilibrium share of 'altruistic' females, $f_A$,\n" +
             "for various signaling probabilities and almost random screening ($e_A=e_a={}$)")
fig.suptitle(fig_title.format(screening_probs[0]), fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/parameter-sweeps/signaling/figure-0.png')
plt.show()


### various signaling probabilities and imperfect screening
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    idx = np.searchsorted(screening_probs, 0.25)
    imperfect_screening = results[:, :, idx, idx, i]
    female_altruists = imperfect_screening[:, :, 4:7:2].sum(axis=-1)

    # plot the equilibrium proportion of female altruists
    ax = fig.add_subplot(2, 3, i+1)
    mappable = ax.imshow(female_altruists, extent=[0, 1, 0, 1], origin='lower',
                         vmin=0, vmax=1, cmap=cmap)
    ax.set_xlabel('$d_a$', fontsize=20)
    ax.set_ylabel('$d_A$', fontsize=20, rotation='horizontal')

    # add a subplot title
    mGA0 = initial_condition[0]
    mga0 = initial_condition[3]
    sub_title = '$m_{GA,0}=f_{GA,0}= %g$,\n $m_{ga,0}=f_{ga,0}=%g$'
    ax.set_title(sub_title % (mGA0, mga0))

fig_title = ("Equilibrium share of 'altruistic' females, $f_A$,\n" +
             "for various signaling probabilities and imperfect screening ($e_A=e_a={}$)")
fig.suptitle(fig_title.format(screening_probs[idx]), fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/parameter-sweeps/signaling/figure-1.png')
plt.show()


### various signaling probabilities and random screening
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    idx = np.searchsorted(screening_probs, 0.5)
    random_screening = results[:, :, idx, idx, i]
    female_altruists = random_screening[:, :, 4:7:2].sum(axis=-1)

    # plot the equilibrium proportion of female altruists
    ax = fig.add_subplot(2, 3, i+1)
    mappable = ax.imshow(female_altruists, extent=[0, 1, 0, 1], origin='lower',
                         vmin=0, vmax=1, cmap=cmap)
    ax.set_xlabel('$d_a$', fontsize=20)
    ax.set_ylabel('$d_A$', fontsize=20, rotation='horizontal')

    # add a subplot title
    mGA0 = initial_condition[0]
    mga0 = initial_condition[3]
    sub_title = '$m_{GA,0}=f_{GA,0}= %g$,\n $m_{ga,0}=f_{ga,0}=%g$'
    ax.set_title(sub_title % (mGA0, mga0))

fig_title = ("Equilibrium share of 'altruistic' females, $f_A$,\n" +
             "for various signaling probabilities and imperfect screening ($e_A=e_a={}$)")
fig.suptitle(fig_title.format(screening_probs[idx]), fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/parameter-sweeps/signaling/figure-2.png')
plt.show()


### various signaling probabilities and imperfect screening
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    idx = np.searchsorted(screening_probs, 0.75)
    imperfect_screening = results[:, :, idx, idx, i]
    female_altruists = imperfect_screening[:, :, 4:7:2].sum(axis=-1)

    # plot the equilibrium proportion of female altruists
    ax = fig.add_subplot(2, 3, i+1)
    mappable = ax.imshow(female_altruists, extent=[0, 1, 0, 1], origin='lower',
                         vmin=0, vmax=1, cmap=cmap)
    ax.set_xlabel('$d_a$', fontsize=20)
    ax.set_ylabel('$d_A$', fontsize=20, rotation='horizontal')

    # add a subplot title
    mGA0 = initial_condition[0]
    mga0 = initial_condition[3]
    sub_title = '$m_{GA,0}=f_{GA,0}= %g$,\n $m_{ga,0}=f_{ga,0}=%g$'
    ax.set_title(sub_title % (mGA0, mga0))

fig_title = ("Equilibrium share of 'altruistic' females, $f_A$,\n" +
             "for various signaling probabilities and imperfect screening ($e_A=e_a={}$)")
fig.suptitle(fig_title.format(screening_probs[idx]), fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/parameter-sweeps/signaling/figure-3.png')
plt.show()


### various signaling probabilities and perfect screening
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    perfect_screening = results[:, :, -1, -1, i]
    female_altruists = perfect_screening[:, :, 4:7:2].sum(axis=-1)

    # plot the equilibrium proportion of female altruists
    ax = fig.add_subplot(2, 3, i+1)
    mappable = ax.imshow(female_altruists, extent=[0, 1, 0, 1], origin='lower',
                         vmin=0, vmax=1, cmap=cmap)
    ax.set_xlabel('$d_a$', fontsize=20)
    ax.set_ylabel('$d_A$', fontsize=20, rotation='horizontal')

    # add a subplot title
    mGA0 = initial_condition[0]
    mga0 = initial_condition[3]
    sub_title = '$m_{GA,0}=f_{GA,0}= %g$,\n $m_{ga,0}=f_{ga,0}=%g$'
    ax.set_title(sub_title % (mGA0, mga0))

fig_title = ("Equilibrium share of 'altruistic' females, $f_A$,\n" +
             "for various signaling probabilities and almost perfect screening ($e_A=e_a={}$)")
fig.suptitle(fig_title.format(screening_probs[-1]), fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/parameter-sweeps/signaling/figure-4.png')
plt.show()


### various signaling probabilities and asymmetric screening
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    lower_idx = np.searchsorted(screening_probs, 0.25)
    upper_idx = np.searchsorted(screening_probs, 0.75)
    asymmetric_screening = results[:, :, upper_idx, lower_idx, i]
    female_altruists = asymmetric_screening[:, :, 4:7:2].sum(axis=-1)

    # plot the equilibrium proportion of female altruists
    ax = fig.add_subplot(2, 3, i+1)
    mappable = ax.imshow(female_altruists, extent=[0, 1, 0, 1], origin='lower',
                         vmin=0, vmax=1, cmap=cmap)
    ax.set_xlabel('$d_a$', fontsize=20)
    ax.set_ylabel('$d_A$', fontsize=20, rotation='horizontal')

    # add a subplot title
    mGA0 = initial_condition[0]
    mga0 = initial_condition[3]
    sub_title = '$m_{GA,0}=f_{GA,0}= %g$,\n $m_{ga,0}=f_{ga,0}=%g$'
    ax.set_title(sub_title % (mGA0, mga0))

fig_title = ("Equilibrium share of 'altruistic' females, $f_A$,\n" +
             "for various signaling probabilities and asymmetric screening ($e_A={}, e_a={}$)")
fig.suptitle(fig_title.format(screening_probs[upper_idx], screening_probs[lower_idx]),
             fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/parameter-sweeps/signaling/figure-5.png')
plt.show()


### various signaling probabilities and asymmetric screening
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    lower_idx = np.searchsorted(screening_probs, 0.25)
    upper_idx = np.searchsorted(screening_probs, 0.75)
    asymmetric_screening = results[:, :, lower_idx, upper_idx, i]
    female_altruists = asymmetric_screening[:, :, 4:7:2].sum(axis=-1)

    # plot the equilibrium proportion of female altruists
    ax = fig.add_subplot(2, 3, i+1)
    mappable = ax.imshow(female_altruists, extent=[0, 1, 0, 1], origin='lower',
                         vmin=0, vmax=1, cmap=cmap)
    ax.set_xlabel('$d_a$', fontsize=20)
    ax.set_ylabel('$d_A$', fontsize=20, rotation='horizontal')

    # add a subplot title
    mGA0 = initial_condition[0]
    mga0 = initial_condition[3]
    sub_title = '$m_{GA,0}=f_{GA,0}= %g$,\n $m_{ga,0}=f_{ga,0}=%g$'
    ax.set_title(sub_title % (mGA0, mga0))

fig_title = ("Equilibrium share of 'altruistic' females, $f_A$,\n" +
             "for various signaling probabilities and asymmetric screening ($e_A={}, e_a={}$)")
fig.suptitle(fig_title.format(screening_probs[lower_idx], screening_probs[upper_idx]),
             fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../../images/parameter-sweeps/signaling/figure-6.png')
plt.show()
