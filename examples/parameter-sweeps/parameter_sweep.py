import sys
sys.path.append('../../')

import numpy as np

from package import model

# define a model object
mod = model.Model()

# define an array of initial conditions S_A = mGA = fGA
N_initial = 5
eps = 1e-3
mGA0 = np.linspace(eps, 1 - eps, N_initial)
mga0 = 1 - mGA0

tup = (mGA0[:, np.newaxis], np.zeros((mGA0.size, 2)), mga0[:, np.newaxis])
initial_males = np.hstack(tup)
initial_females = initial_males

initial_conditions = np.hstack((initial_males, initial_females))

# define some signaling and screening params
N_probs = 7
signaling_probs = np.linspace(eps, 1 - eps, N_probs)
screening_probs = signaling_probs

# storage container for results
tmp_shape = 4 * (N_probs,) + (N_initial, 8)
results = np.empty(tmp_shape)


def main():
    """Runs the parameter sweep."""
    counter = 0
    for i, dA in enumerate(signaling_probs):
        for j, da in enumerate(signaling_probs):
            for k, eA in enumerate(screening_probs):
                for l, ea in enumerate(screening_probs):
                    for m, initial_condition in enumerate(initial_conditions):

                        # fix the model parameters
                        tmp_params = {'dA': dA, 'da': da, 'eA': eA, 'ea': ea,
                                      'PiaA': 10.0, 'PiAA': 3.0, 'Piaa': 2.0,
                                      'PiAa': 0.0}
                        mod.params = tmp_params

                        # simulate the model to find the equilibrium
                        tmp_traj = mod.simulate(initial_condition, rtol=1e-4)

                        # store the results
                        results[i, j, k, l, m, :] = tmp_traj[:, -1]

                        counter += 1

                        if counter % 100 == 0:
                            print('Done with {} out of {}'.format(counter, results.size / 8))

    np.save("parameter_sweep", results)


if __name__ == '__main__':
    main()
