"""
Classes representing the equilibrium system of non-linear equations.

@author : David R. Pugh
@date : 2014-11-08

"""
import sympy as sym


class Equilibrium(object):

    __symbolic_jacobian_wrt_params = None

    __symbolic_jacobian_wrt_vars = None

    def __init__(self, family):
        self.family = family

    @property
    def _symbolic_residual(self):
        residual = (self.family._symbolic_system -
                    sym.Matrix(self.family._symbolic_vars))
        return residual

    @property
    def _symbolic_jacobian_wrt_params(self):
        if self.__symbolic_jacobian_wrt_params is None:
            params = sym.var(list(self.family.params.keys()))
            self.__symbolic_jacobian_wrt_params = self.derive_jacobian(params)
        return self.__symbolic_jacobian_wrt_params

    @property
    def _symbolic_jacobian_wrt_vars(self):
        if self.__symbolic_jacobian_wrt_vars is None:
            variables = self.family._symbolic_vars
            self.__symbolic_jacobian_wrt_vars = self.derive_jacobian(variables)
        return self.__symbolic_jacobian_wrt_vars

    def derive_jacobian(self, variables):
        return self._symbolic_residual.jacobian(variables)
