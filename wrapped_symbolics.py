"""
Converts symbolic expressions defining the model into callable NumPy functions.

"""
import sympy as sp

import symbolics

# extract the important variables
fGA, fGa, fgA, fga = symbolics.female_allele_shares
mGA, mGa, mgA, mga = symbolics.male_allele_shares
dA, da = symbolics.female_signaling_probs
eA, ea = symbolics.male_screening_probs
PiaA, PiAA, Piaa, PiAa = symbolics.matching_payoffs

# wraps the symbolic model system and its Jacobian for use in numerical analysis
tmp_args = (mGA, mGa, mgA, mga, fGA, fGa, fgA, fga, dA, da, eA, ea, PiaA, PiAA, Piaa, PiAa)
model_system = sp.lambdify(tmp_args, symbolics.model_system, modules='numpy')
model_jacobian = sp.lambdify(tmp_args, symbolics.model_jacobian, modules='numpy')

# wraps the symbolic residual and its Jacobian for use in numerical analysis
residual = sp.lambdify(tmp_args, symbolics.residual, modules='numpy')
residual_jacobian = sp.lambdify(tmp_args, symbolics.residual_jacobian, modules='numpy')
