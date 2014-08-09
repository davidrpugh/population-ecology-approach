import sys
sys.path.append('../../')


import numpy as np

from package import model

# define an array of initial conditions S = mGA = fGA
N = 5
mGA0 = np.linspace(0, 1, N).reshape((N, 1))
mga0 = 1 - mGA0
initial_males = np.hstack((mGA0, np.zeros((N, 2)), mga0))
initial_females = initial_males
initial_conditions = np.hstack((initial_males, initial_females))

# define some signaling probs
signaling_probs = np.linspace(0, 1, N)
screening_probs = signaling_probs

# storage container
results = np.empty((N, N, N, N, N, 8))

for i, dA in enumerate(signaling_probs):
    for j, da in enumerate(signaling_probs):
        for k, eA in enumerate(screening_probs):
            for l, ea in enumerate(screening_probs):
                for m, initial_condition in enumerate(initial_conditions):

                    # fix the model parameters
                    tmp_params = {'dA': dA, 'da': da, 'eA': eA, 'ea': ea,
                                  'PiaA': 6.0, 'PiAA': 5.0, 'Piaa': 4.0, 'PiAa': 3.0}

                    # create an instance of the model
                    tmp_example = model.Model(params=tmp_params)

                    # simulate the model to find the equilibrium
                    tmp_traj = tmp_example.simulate(initial_condition, rtol=1e-4)

                    # store the results
                    results[i, j, k, l, m, :] = tmp_traj[:, -1]
