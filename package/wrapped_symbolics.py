"""
Converts symbolic expressions defining the model into callable NumPy functions.

"""
import sympy as sp

import sandbox

# extract the important variables
dA, da = sandbox.female_signaling_probs
eA, ea = sandbox.male_screening_probs
PiaA, PiAA, Piaa, PiAa = sandbox.prisoners_dilemma_payoffs

# wraps the symbolic system and its Jacobian for use in numerical analysis
tmp_args = (sandbox.men, sandbox.girls, dA, da, eA, ea, PiaA, PiAA, Piaa, PiAa)
model_system = sp.lambdify(tmp_args, sandbox.model_system, 'numpy')
model_jacobian = sp.lambdify(tmp_args, sandbox.model_jacobian, 'numpy')

# wraps the symbolic residual and its Jacobian for use in numerical analysis
residual = sp.lambdify(tmp_args, sandbox.residual, 'numpy')
residual_jacobian = sp.lambdify(tmp_args, sandbox.residual_jacobian, 'numpy')
