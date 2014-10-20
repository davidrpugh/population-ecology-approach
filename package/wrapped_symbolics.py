"""
Converts symbolic expressions defining the model into callable NumPy functions.

"""
import numpy as np
import sympy as sp

import symbolics

# extract the important variables
c = symbolics.fecundity_factor
dA, da = symbolics.female_signaling_probs
eA, ea = symbolics.male_screening_probs
PiaA, PiAA, Piaa, PiAa = symbolics.prisoners_dilemma_payoffs

# wrapped symbolic expressions for sizes of the adoption pools
tmp_args = (symbolics.girls, c, dA, da, eA, ea, PiaA, PiAA, Piaa, PiAa)
altruist_adoption_pool = sp.lambdify(tmp_args,
                                     symbolics.altruist_adoption_pool,
                                     modules=[{'ImmutableMatrix': np.array}, "numpy"])
selfish_adoption_pool = sp.lambdify(tmp_args,
                                    symbolics.selfish_adoption_pool,
                                    modules=[{'ImmutableMatrix': np.array}, "numpy"])

# wraps the symbolic system and its Jacobian for use in numerical analysis
tmp_args = (symbolics.men, symbolics.girls, c, dA, da, eA, ea, PiaA, PiAA, Piaa, PiAa)
model_system = sp.lambdify(tmp_args,
                           symbolics.model_system,
                           modules=[{'ImmutableMatrix': np.array}, "numpy"])
model_jacobian = sp.lambdify(tmp_args,
                             symbolics.model_jacobian,
                             modules=[{'ImmutableMatrix': np.array}, "numpy"])

# wraps the symbolic residual and its Jacobian for use in numerical analysis
residual = sp.lambdify(tmp_args,
                       symbolics.residual,
                       modules=[{'ImmutableMatrix': np.array}, "numpy"])
residual_jacobian = sp.lambdify(tmp_args,
                                symbolics.residual_jacobian,
                                modules=[{'ImmutableMatrix': np.array}, "numpy"])
