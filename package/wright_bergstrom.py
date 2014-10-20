import numpy as np
import sympy as sym

import families

# number of female children of particular genotype
girls = sym.DeferredVector('f')

# number of male adults of particular genotype
men = sym.DeferredVector('M')

# Male screening probabilities
e = sym.var('e')

# Female population by phenotype.
altruistic_girls = girls[0] + girls[2]
selfish_girls = girls[1] + girls[3]

# conditional phenotype matching probabilities (a la Wright/Bergstrom)
SGA = e + (1 - e) * altruistic_girls / (altruistic_girls + selfish_girls)
SGa = 1 - SGA
Sga = e + (1 - e) * selfish_girls / (altruistic_girls + selfish_girls)
SgA = 1 - Sga

# females send precise signals, but males screen almost randomly
eps = 0.5
params = {'c': 5.0, 'e': eps,
          'PiaA': 9.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}

# define an array of initial conditions S_A = mGA = fGA
N_initial = 5
eps = 1e-3
mGA0 = np.linspace(eps, 1 - eps, N_initial)

# create an instance of the model
example = families.OneMaleTwoFemales(params=params,
                                     SGA=SGA,
                                     Sga=Sga)
