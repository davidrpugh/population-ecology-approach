"""
Defines the model classes.

"""
import numpy as np
from scipy import optimize
import sympy as sp

from traits.api import (Array, cached_property, Dict, Float, HasPrivateTraits, 
                        Property, Str)

import wrapped_symbolics
import wrapped_transformed_symbolics


class Model(HasPrivateTraits):
    """Base class representing the model of Pugh-Schaefer-Seabright."""

    _bound_constraints = Property

    _equality_constraints = Property

    _female_alleles_constraint = Property

    _initial_guess = Array

    _male_alleles_constraint = Property

    initial_guess = Property(Array) 

    params = Dict(Str, Float)

    solver_kwargs = Dict(Str, Float)

    steady_state = Property(depends_on=['_initial_guess, params'])

    def _get__bound_constraints(self):
        """Population shares must be in [0,1]."""
        raise NotImplementedError

    def _get__equality_constraints(self):
        """Population shares of male and female alleles must sum to one."""
        return [self._male_alleles_constraint, self._female_alleles_constraint]

    def _get__female_alleles_constraint(self):
        """Female allele population shares must sum to one."""
        raise NotImplementedError

    def _get__male_alleles_constraint(self):
        """Male allele population shares must sum to one."""
        raise NotImplementedError

    def _get_initial_guess(self):
        """Return initial guess of the equilibrium population shares."""
        return self._initial_guess

    @cached_property
    def _get_steady_state(self):
        """Compute the steady state for the model."""
        result = optimize.minimize(self._objective, 
                                   x0=self._initial_guess,
                                   method='SLSQP', 
                                   jac=self._jacobian,
                                   bounds=self._bound_constraints,
                                   constraints=self._equality_constraints,
                                   **self.solver_kwargs)
        return result

    def _set_initial_guess(self, value):
        """Specify the initial guess of the equilibrium population shares."""
        self._initial_guess = value

    def _jacobian(self, X):
        """Jacobian of the objective function."""
        jac = np.sum(self._residual(X) * self._residual_jacobian(X), axis=0)
        return jac

    def _objective(self, X):
        """Objective function used to solve for the model steady state."""
        obj = 0.5 * np.sum(self._residual(X)**2)
        return obj

    def _residual(self, X):
        """Model steady state is a root of this non-linear system."""
        raise NotImplementedError

    def _residual_jacobian(self, X):
        """Returns the Jacobian of the model residual."""
        raise NotImplementedError

    def F(self, X):
        """Equation of motion for population allele shares."""
        out = wrapped_symbolics.model_system(*X, **self.params)
        return np.array(out).flatten()

    def F_jacobian(self, X):
        """Jacobian for equation of motion."""
        jac = wrapped_symbolics.model_jacobian(*X, **self.params)
        return np.array(jac)

    def simulate(self, initial_condition, T=10):
        """Simulates a run of the model given some initial_condition."""

        # set up the trajectory array
        traj = np.empty((8, T))
        traj[:,0] = initial_condition

        # run the simulation
        for t in range(1,T):
            traj[:,t] = self.F(traj[:,t-1])

        return traj
    

class StandardModel(Model):
    """Class representing the model of Pugh-Schaefer-Seabright."""

    def _get__bound_constraints(self):
        """Population shares must be in [0,1]."""
        eps = 1e-15
        return [(eps, 1 - eps) for i in range(8)]

    def _get__female_alleles_constraint(self):
        """Female allele population shares must sum to one."""
        cons = lambda X: 1 - np.sum(X[4:])
        return {'type': 'eq', 'fun': cons}

    def _get__male_alleles_constraint(self):
        """Male allele population shares must sum to one."""
        cons = lambda X: 1 - np.sum(X[:4])
        return {'type': 'eq', 'fun': cons}

    def _residual(self, X):
        """Model steady state is a root of this non-linear system."""
        resid = wrapped_symbolics.residual(*X, **self.params)
        return np.array(resid)

    def _residual_jacobian(self, X):
        """Returns the Jacobian of the model residual."""
        jac = wrapped_symbolics.residual_jacobian(*X, **self.params)
        return np.array(jac)

 
class TransformedModel(Model):
    """Transformed version of the Pugh-Schaefer-Seabright model."""

    def _get__bound_constraints(self):
        """Population shares must be in [0,1]."""
        return [(None, None) for i in range(8)]

    def _get__equality_constraints(self):
        """Population shares of male and female alleles must sum to one."""
        return [self._male_alleles_constraint, self._female_alleles_constraint]

    def _get__female_alleles_constraint(self):
        """Female allele population shares must sum to one."""
        cons = lambda X: 1 - np.sum((1 / (1 + np.exp(-X))[4:]))
        return {'type': 'eq', 'fun': cons}

    def _get__initial_guess(self):
        """Apply logistic transformation of variables to initial guess."""
        transform_initial_guess = np.log(initial_guess / (1 - initial_guess))
        return transform_initial_guess

    def _get__male_alleles_constraint(self):
        """Male allele population shares must sum to one."""
        cons = lambda X: 1 - np.sum((1 / (1 + np.exp(-X))[:4]))
        return {'type': 'eq', 'fun': cons}

    def _residual(self, X):
        """Model steady state is a root of this non-linear system."""
        resid = wrapped_transformed_symbolics.residual(*X, **self.params)
        return np.array(resid)

    def _residual_jacobian(self, X):
        """Returns the Jacobian of the model residual."""
        jac = wrapped_transformed_symbolics.residual_jacobian(*X, **self.params)
        return np.array(jac)
