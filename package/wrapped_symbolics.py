"""
Converts symbolic expressions defining the model into callable NumPy functions.

"""
import numpy as np
import sympy as sp

import sandbox

# extract the important variables
c = sandbox.fecundity_factor
dA, da = sandbox.female_signaling_probs
eA, ea = sandbox.male_screening_probs
PiaA, PiAA, Piaa, PiAa = sandbox.prisoners_dilemma_payoffs

# wrapped symbolic expressions for sizes of the adoption pools
tmp_args = (sandbox.girls, c, dA, da, eA, ea, PiaA, PiAA, Piaa, PiAa)
altruist_adoption_pool = sp.lambdify(tmp_args,
                                     sandbox.altruist_adoption_pool,
                                     modules=[{'ImmutableMatrix': np.array}, "numpy"])
selfish_adoption_pool = sp.lambdify(tmp_args,
                                    sandbox.selfish_adoption_pool,
                                    modules=[{'ImmutableMatrix': np.array}, "numpy"])

# wraps the symbolic system and its Jacobian for use in numerical analysis
tmp_args = (sandbox.men, sandbox.girls, c, dA, da, eA, ea, PiaA, PiAA, Piaa, PiAa)
model_system = sp.lambdify(tmp_args,
                           sandbox.model_system,
                           modules=[{'ImmutableMatrix': np.array}, "numpy"])
model_jacobian = sp.lambdify(tmp_args,
                             sandbox.model_jacobian,
                             modules=[{'ImmutableMatrix': np.array}, "numpy"])

# wraps the symbolic residual and its Jacobian for use in numerical analysis
residual = sp.lambdify(tmp_args,
                       sandbox.residual,
                       modules=[{'ImmutableMatrix': np.array}, "numpy"])
residual_jacobian = sp.lambdify(tmp_args,
                                sandbox.residual_jacobian,
                                modules=[{'ImmutableMatrix': np.array}, "numpy"])
