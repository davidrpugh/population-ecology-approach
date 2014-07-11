import matplotlib.pyplot as plt
import numpy as np 

import model

params = {'dA':0.95, 'da':0.5, 'eA':0.5, 'ea':0.5, 'PiaA':6.0, 'PiAA':5.0, 
          'Piaa':4.0, 'PiAa':3.0}

# initial guess for the solver       
mGA0 = 0.05
mGa0 = 0.05
mgA0 = 0.05
mga0 = 1 - mGA0 - mGa0 - mgA0
 
fGA0 = mGA0
fGa0 = mGa0
fgA0 = mgA0
fga0 = 1 - fGA0 - fGa0 - fgA0

tmp_initial_guess = np.array([mGA0, mGa0, mgA0, mga0, fGA0, fGa0, fgA0, fga0])

N = 10
steady_states = np.empty((N, 8))
signalling_probs = np.linspace(0.5, 1, N)

for i, dA in enumerate(signalling_probs):
    for j, da in enumerate(signalling_probs):

        tmp_params = {'dA':dA, 'da':da, 'eA':0.5, 'ea':0.5, 
                      'PiaA':6.0, 'PiAA':5.0, 'Piaa':4.0, 'PiAa':3.0}
         
        tmp_model = model.Model(initial_guess=tmp_initial_guess,
                                params=tmp_params,
                                solver_kwargs={'tol': 1e-12})

        if tmp_model.steady_state.success:
            steady_states[i] = tmp_model.steady_state.x
            tmp_initial_guess = tmp_model.steady_state.x
        else:
            steady_states[i] = np.nan

    
def equilibrium_population_shares(initial_guess, params):
    """Plot the equilibrium, or steady state, population shares."""
    tmp_model = model.Model(initial_guess=initial_guess,
                            params=params,
                            solver_kwargs={'tol': 1e-12})

    fig, ax = plt.subplots()
    ind = np.arange(-0.5, 7.5) 

    ax.bar(left=ind, height=tmp_model.steady_state.x, width=1.0, alpha=0.5)
    male_labels = ('$m_{GA}$', '$m_{Ga}$', '$m_{gA}$', '$m_{ga}$')
    female_labels = ('$f_{GA}$', '$f_{Ga}$', '$f_{gA}$', '$f_{ga}$')

    ax.set_xlim(-1.0, 8.5)
    ax.set_xticks(np.arange(8))
    ax.set_xticklabels(male_labels + female_labels) 
    ax.set_ylim(0, 1)




