import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

import model

params = {'dA':0.25, 'da':0.75, 'eA':0.25, 'ea':0.5, 'PiaA':6.0, 'PiAA':5.0, 
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

initial_guess = np.array([mGA0, mGa0, mgA0, mga0, fGA0, fGa0, fgA0, fga0])

example = model.Model(initial_guess=initial_guess,
                      params=params,
                      solver_kwargs={'tol': 1e-12})

# simulate the model
initial_condition = initial_guess
traj = example.simulate(initial_condition, T=100)

fig, axes = plt.subplots(1, 2, figsize=(12, 8))

axes[0].plot(traj[0], label=r'$m_{GA}$')
axes[0].plot(traj[1], label=r'$m_{Ga}$')
axes[0].plot(traj[2], label=r'$m_{gA}$')
axes[0].plot(traj[3], label=r'$m_{ga}$')

axes[0].set_ylim(0, 1)
axes[0].set_ylabel('Population shares', family='serif', fontsize=15)
axes[0].set_title('Males', family='serif', fontsize=20)
axes[0].legend(loc=0, frameon=False)

axes[1].plot(traj[4], label=r'$f_{GA}$')
axes[1].plot(traj[5], label=r'$f_{Ga}$')
axes[1].plot(traj[6], label=r'$f_{gA}$')
axes[1].plot(traj[7], label=r'$f_{ga}$')

axes[1].set_ylim(0, 1)
axes[1].set_title('Females', family='serif', fontsize=20)
axes[1].legend(loc=0, frameon=False)

plt.show()
    

