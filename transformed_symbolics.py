"""
Applies a logistic transformation of variables to the original model.

"""
import sympy as sp 

import symbolics

# define some new variables
sp.var('mGA_tilde, mGa_tilde, mgA_tilde, mga_tilde')
sp.var('fGA_tilde, fGa_tilde, fgA_tilde, fga_tilde')

# define a transformation of variables using the logistic function
logistic_transform = {'mGA':(1 / (1 + sp.exp(-mGA_tilde))),
                      'mGa':(1 / (1 + sp.exp(-mGa_tilde))),
                      'mgA':(1 / (1 + sp.exp(-mgA_tilde))),
                      'mga':(1 / (1 + sp.exp(-mga_tilde))),
                      'fGA':(1 / (1 + sp.exp(-fGA_tilde))),
                      'fGa':(1 / (1 + sp.exp(-fGa_tilde))), 
                      'fgA':(1 / (1 + sp.exp(-fgA_tilde))),
                      'fga':(1 / (1 + sp.exp(-fga_tilde)))}

# symbolic system of equations for model simulation
endog_vars = sp.Matrix([mGA_tilde, mGa_tilde, mgA_tilde, mga_tilde, 
                        fGA_tilde, fGa_tilde, fgA_tilde, fga_tilde])
model_system = symbolics.model_system.subs(logistic_transform)
                   
# symbolic model Jacobian for stability analysis
model_jacobian = model_system.jacobian(endog_vars)

# steady state of the model makes residual zero
residual = model_system - sp.Matrix(endog_vars)

# residual Jacobian is an input to steady state solver
residual_jacobian = residual.jacobian(endog_vars)
