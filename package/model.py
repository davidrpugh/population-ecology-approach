"""
Defines the model classes.

"""
import numpy as np
from scipy import linalg, optimize

from traits.api import (Array, Bool, cached_property, Dict, Float,
                        HasPrivateTraits, Property, Str)

import wrapped_symbolics


class Model(HasPrivateTraits):
    """Base class representing the model of Pugh-Schaefer-Seabright."""

    _bound_constraints = Property

    _equality_constraints = Property

    _female_alleles_constraint = Property

    _initial_guess = Array

    _male_alleles_constraint = Property

    eigenvalues = Property

    initial_condition = Property(Float)

    initial_guess = Property(Array)

    isunstable = Property(Bool)

    isstable = Property(Bool)

    params = Dict(Str, Float)

    solver_kwargs = Dict(Str, Float)

    steady_state = Property(depends_on=['_initial_guess, params'])

    def _get__bound_constraints(self):
        """Population shares must be in [0,1]."""
        eps = 1e-15
        return [(eps, 1 - eps) for i in range(8)]

    def _get__equality_constraints(self):
        """Population shares of male and female alleles must sum to one."""
        return [self._male_alleles_constraint, self._female_alleles_constraint]

    def _get__female_alleles_constraint(self):
        """Female allele population shares must sum to one."""
        cons = lambda X: 1 - np.sum(X[4:])
        return {'type': 'eq', 'fun': cons}

    def _get__male_alleles_constraint(self):
        """Male allele population shares must sum to one."""
        cons = lambda X: 1 - np.sum(X[:4])
        return {'type': 'eq', 'fun': cons}

    def _get_eigenvalues(self):
        """Return the eigenvalues of the Jacobian evaluated at equilibrium."""
        evaluated_jac = self.F_jacobian(self.steady_state.x)
        eigen_vals, eigen_vecs = linalg.eig(evaluated_jac)
        return eigen_vals

    def _get_initial_condition(self):
        """Return initial condition for a simulation."""
        mGA0 = self._initial_condition
        mga0 = 1 - mGA0

        initial_males = np.array([mGA0, 0, 0, mga0])

        # f_GA(0)=mGA0*Pi_AA and f_ga(0)=mga0*Pi_aa.
        initial_females = np.array([self.params['PiAA'] * mGA0, 0.0,
                                    0.0, self.params['Piaa'] * mga0])

        initial_condition = np.hstack((initial_males, initial_females))
        return initial_condition

    def _get_initial_guess(self):
        """Return initial guess of the equilibrium population shares."""
        return self._initial_guess

    def _get_isunstable(self):
        """Return True if the steady state of the model is unstable."""
        return np.any(np.greater(np.abs(self.eigenvalues), 1.0))

    def _get_isstable(self):
        """Return True if the steady state of the model is stable."""
        return np.all(np.less(np.abs(self.eigenvalues), 1.0))

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

    def _set_initial_condition(self, value):
        """Specify the initial condition for a simulation."""
        self._initial_condition = value

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
        resid = wrapped_symbolics.residual(X[:4], X[4:], **self.params)
        return np.array(resid)

    def _residual_jacobian(self, X):
        """Returns the Jacobian of the model residual."""
        jac = wrapped_symbolics.residual_jacobian(X[:4], X[4:], **self.params)
        return np.array(jac)

    def _simulate_fixed_trajectory(self, initial_condition, T):
        """Simulates a trajectory of fixed length."""
        # set up the trajectory array
        traj = np.empty((8, T))
        traj[:, 0] = initial_condition

        # run the simulation
        for t in range(1, T):
            current_shares = traj[:, t-1]
            new_shares = self.F(current_shares)
            traj[:, t] = new_shares

        return traj

    def _simulate_variable_trajectory(self, initial_condition, rtol):
        """Simulates a trajectory of variable length."""
        # set up the trajectory array
        traj = initial_condition.reshape((8, 1))

        # initialize delta
        delta = np.ones(8)

        # run the simulation
        while np.any(np.greater(delta, rtol)):
            current_shares = traj[:, -1]
            new_shares = self.F(current_shares)
            delta = np.abs(new_shares - current_shares)

            # update the trajectory
            traj = np.hstack((traj, new_shares[:, np.newaxis]))

        return traj

    def F(self, X):
        """Equation of motion for population allele shares."""
        out = wrapped_symbolics.model_system(X[:4], X[4:], **self.params)
        return out.ravel()

    def F_jacobian(self, X):
        """Jacobian for equation of motion."""
        jac = wrapped_symbolics.model_jacobian(X[:4], X[4:], **self.params)
        return np.array(jac)

    def simulate(self, T=None, rtol=None):
        """Simulates a run of the model given some initial_condition."""
        if T is not None:
            traj = self._simulate_fixed_trajectory(self.initial_condition, T)
        elif rtol is not None:
            traj = self._simulate_variable_trajectory(self.initial_condition, rtol)
        else:
            raise ValueError("One of 'T' or 'rtol' must be specified.")
        return traj


class DepletionError(Exception):
    pass
