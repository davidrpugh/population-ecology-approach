class Solver(object):
    """Base class for steady state solvers."""

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
    def _symbolic_residual(self):
        """
        Symbolic matrix representing the model residual.

        :getter: Return the model residual.
        :type: sympy.Matrix

        """
        return self.family._symbolic_system - self.family._symbolic_vars

    @property
    def _symbolic_jacobian(self):
        """
        Jacobian matrix of partial derivatives.

        :getter: Return Jacobian matrix of partial derivatives.
        :type: sympy.Matrix

        """
        return self._symbolic_residual.jacobian(self.family._symbolic_vars)
