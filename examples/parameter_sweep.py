import sys
sys.path.append('../../')


import numpy as np

from package import model

# define a model object
mod = model.Model()

# define an array of initial conditions S_A = mGA = fGA
step = 0.25
mGA0 = np.arange(0, 1, step)
mga0 = 1 - mGA0

tup = (mGA0[:, np.newaxis], np.zeros((mGA0.size, 2)), mga0[:, np.newaxis])
initial_males = np.hstack(tup)
initial_females = initial_males

initial_conditions = np.hstack((initial_males, initial_females))

# define some signaling and screening params
step = 0.1
signaling_probs = np.arange(0, 1+step, step)
screening_probs = np.arange(0.5, 1+step, step)

# storage container for results
N_signal = signaling_probs.size
N_screen = screening_probs.size
N_initial = initial_conditions.shape[0]
results_shape = (N_signal, N_signal, N_screen, N_screen, N_initial, 8)
results = np.empty(results_shape)

for i, dA in enumerate(signaling_probs):
    for j, da in enumerate(signaling_probs):
        for k, eA in enumerate(screening_probs):
            for l, ea in enumerate(screening_probs):
                for m, initial_condition in enumerate(initial_conditions):

                    # fix the model parameters
                    tmp_params = {'dA': dA, 'da': da, 'eA': eA, 'ea': ea,
                                  'PiaA': 6.0, 'PiAA': 5.0, 'Piaa': 4.0, 'PiAa': 3.0}
                    mod.params = tmp_params

                    # simulate the model to find the equilibrium
                    tmp_traj = mod.simulate(initial_condition, rtol=1e-4)

                    # store the results
                    results[i, j, k, l, m, :] = tmp_traj[:, -1]
