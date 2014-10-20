import numpy as np
from scipy import optimize
import sympy as sym


class Solver(object):
    """Base class for steady state solvers."""

    __numeric_jacobian = None

    __numeric_residual = None

    _modules = [{'ImmutableMatrix': np.array}, "numpy"]

    def __init__(self, family):
        """
        Create an instance of the Solver class.

        Parameters
        ----------
        family : families.family
            Instance of the families.Family class defining a family unit.

        """
        self.family = family

    @property
    def _numeric_residual_jacobian(self):
        """
        Vectorized function for numerically evaluating the Jacobian matrix.

        :getter: Return the current function.
        :type: function

        """
        if self.__numeric_jacobian is None:
            self.__numeric_jacobian = sym.lambdify(self.family._symbolic_args,
                                                   self._symbolic_residual_jacobian,
                                                   self._modules)
        return self.__numeric_jacobian

    @property
    def _numeric_residual(self):
        """
        Vectorized function for numerically evaluating the model residual.

        :getter: Return the current function.
        :type: function

        """
        if self.__numeric_residual is None:
            self.__numeric_residual = sym.lambdify(self.family._symbolic_args,
                                                   self._symbolic_residual,
                                                   self._modules)
        return self.__numeric_residual

    @property
    def _symbolic_residual(self):
        """
        Symbolic representation of the model residual.

        :getter: Return the model residual.
        :type: sympy.Matrix

        """
        resid = (self.family._symbolic_system -
                 sym.Matrix(self.family._symbolic_vars))
        return resid

    @property
    def _symbolic_residual_jacobian(self):
        """
        Symbolic representation of the Jacobian matrix of partial derivatives.

        :getter: Return Jacobian matrix of partial derivatives.
        :type: sympy.Matrix

        """
        return self._symbolic_residual.jacobian(self.family._symbolic_vars)

    @property
    def initial_guess(self):
        """
        Initial guess for the solver.

        :getter: Return the current initial guess.
        :setter: Set a new initial guess.
        :type: numpy.ndarray

        """
        return self._initial_guess

    @initial_guess.setter
    def initial_guess(self, value):
        """Set a new initial guess."""
        self._initial_guess = value

    def residual_jacobian(self, X):
        """
        Jacobian matrix of partial derivatives for the system of non-linear
        equations defining the steady state.

        Parameters
        ----------
        X : numpy.ndarray

        Returns
        -------
        jac : numpy.ndarray
            Jacobian matrix of partial derivatives.

        """
        jac = self._numeric_residual_jacobian(X[:4], X[4:], **self.family.params)
        return jac

    def residual(self, X):
        """
        System of non-linear equations defining the model steady state.

        Parameters
        ----------
        X : numpy.ndarray

        Returns
        -------
        residual : numpy.ndarray
            Value of the model residual given current values of endogenous
            variables and parameters.

        """
        residual = self._numeric_residual(X[:4], X[4:], **self.family.params)
        return residual.ravel()

    def solve(self, *args, **kwargs):
        raise NotImplementedError


class LeastSquaresSolver(Solver):
    """Solve a system of non-linear equations by minimization."""

    @property
    def _bounds(self):
        return self._men_bound_constraints + self._girls_bound_constraints

    @property
    def _constraints(self):
        return [self._men_equality_constraint]

    @property
    def _girls_bound_constraints(self):
        """Numbers of girls with a given genotype must be non-negative."""
        return [(0, None) for i in range(4)]

    @property
    def _men_bound_constraints(self):
        """Shares of men with a given genotype must be in [0, 1]."""
        return [(0, 1) for i in range(4)]

    @property
    def _men_equality_constraint(self):
        """Shares of men with a given genotype must sum to one."""
        cons = lambda X: 1 - np.sum(X[:4])
        return {'type': 'eq', 'fun': cons}

    def _objective(self, X):
        obj = 0.5 * np.sum(self.residual(X)**2)
        return obj

    def _objective_jacobian(self, X):
        jac = np.sum(self.residual(X) * self.residual_jacobian(X), axis=0)
        return jac

    def solve(self, **kwargs):
        """
        Solve the system of non-linear equations describing the equilibrium.

        Parameters
        ----------
        kwargs : dict
            Dictionary of optional solver parameters.

        Returns
        -------
        result :

        """
        result = optimize.minimize(self._objective,
                                   x0=self.initial_guess,
                                   jac=self._objective_jacobian,
                                   method='SLSQP',
                                   bounds=self._bounds,
                                   constraints=self._constraints,
                                   **kwargs
                                   )
        return result


class RootFinder(Solver):
    """Solve a system of non-linear equations by root finding."""

    def solve(self, method='hybr', with_jacobian=False, **kwargs):
        """
        Solve the system of non-linear equations describing the equilibrium.

        Parameters
        ----------
        method : str
        with_jacobian : boolean (default=False)

        Returns
        -------
        result :

        """
        if with_jacobian:
            kwargs['jac'] = self.jacobian
        else:
            kwargs['jac'] = False

        result = optimize.root(self.residual,
                               x0=self.initial_guess,
                               method=method,
                               **kwargs
                               )
        return result
