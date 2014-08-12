import matplotlib.pyplot as plt
import numpy as np

from parameter_sweep import initial_conditions, screening_probs

results = np.load("parameter_sweep.npy")


### various signaling probabilities and useless screening
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    useless_screening = results[:, :, 0, 0, i]
    female_altruists = useless_screening[:, :, 4:7:2].sum(axis=-1)

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
             "for various signaling probabilities and useless screening ($e_A=e_a={}$)")
fig.suptitle(fig_title.format(screening_probs[0]), fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../images/parameter-sweeps/signaling/figure-0.png')
plt.show()


### various signaling probabilities and imperfect screening
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    imperfect_screening = results[:, :, 2, 2, i]
    female_altruists = imperfect_screening[:, :, 4:7:2].sum(axis=-1)

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
             "for various signaling probabilities and imperfect screening ($e_A=e_a={}$)")
fig.suptitle(fig_title.format(screening_probs[2]), fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../images/parameter-sweeps/signaling/figure-1.png')
plt.show()


### various signaling probabilities and random screening
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    random_screening = results[:, :, 5, 5, i]
    female_altruists = random_screening[:, :, 4:7:2].sum(axis=-1)

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
             "for various signaling probabilities and random screening ($e_A=e_a={}$)")
fig.suptitle(fig_title.format(screening_probs[5]), fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../images/parameter-sweeps/signaling/figure-2.png')
plt.show()


### various signaling probabilities and imperfect screening
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    imperfect_screening = results[:, :, 8, 8, i]
    female_altruists = imperfect_screening[:, :, 4:7:2].sum(axis=-1)

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
             "for various signaling probabilities and imperfect screening ($e_A=e_a={}$)")
fig.suptitle(fig_title.format(screening_probs[8]), fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../images/parameter-sweeps/signaling/figure-3.png')
plt.show()


### various signaling probabilities and perfect screening
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    perfect_screening = results[:, :, -1, -1, i]
    female_altruists = perfect_screening[:, :, 4:7:2].sum(axis=-1)

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
             "for various signaling probabilities and perfect screening ($e_A=e_a={}$)")
fig.suptitle(fig_title.format(screening_probs[-1]), fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../images/parameter-sweeps/signaling/figure-4.png')
plt.show()


### various signaling probabilities and asymmetric screening
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    asymmetric_screening = results[:, :, 8, 2, i]
    female_altruists = asymmetric_screening[:, :, 4:7:2].sum(axis=-1)

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
             "for various signaling probabilities and asymmetric screening ($e_A={}, e_a={}$)")
fig.suptitle(fig_title.format(screening_probs[8], screening_probs[2]),
             fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../images/parameter-sweeps/signaling/figure-5.png')
plt.show()


### various signaling probabilities and asymmetric screening
fig = plt.figure(figsize=(18, 12))

for i, initial_condition in enumerate(initial_conditions):

    asymmetric_screening = results[:, :, 2, 8, i]
    female_altruists = asymmetric_screening[:, :, 4:7:2].sum(axis=-1)

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
             "for various signaling probabilities and asymmetric screening ($e_A={}, e_a={}$)")
fig.suptitle(fig_title.format(screening_probs[2], screening_probs[8]),
             fontsize=20, family='serif')

# add a color bar
fig.subplots_adjust(right=0.8)
cax = fig.add_axes([0.85, 0.1, 0.075, 0.8])
fig.colorbar(mappable, cax=cax)

fig.savefig('../images/parameter-sweeps/signaling/figure-6.png')
plt.show()
