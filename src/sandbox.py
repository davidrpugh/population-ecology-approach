import collections

import numpy as np
import sympy as sym

from quantecon import ivp

import models

# declare generic symbolic variables
t, X = sym.symbols('t'), sym.DeferredVector('X')


class Simulator(object):
    """Class representing a simulator for the Model."""

    __numeric_jacobian = None

    __numeric_system = None

    _modules = [{'ImmutableMatrix': np.array}, "numpy"]

    def __init__(self, model, params, with_jacobian=False):
        """
        Create an instance of the Simulator class.

        """
        self.model = model
        self.params = params
        self.with_jacobian = with_jacobian

    @property
    def _change_of_variables(self):
        """Change of variables (necessary in order to use quantecon.ivp)."""
        new_variables = ([(models.males[i], X[i]) for i in range(4)] +
                         [(models.females[i], X[i+4]) for i in range(4)] +
                         [(models.U[i], X[i+8]) for i in range(64)])
        return new_variables

    @property
    def _numeric_jacobian(self):
        """
        Vectorized, numpy-aware function defining the Jacobian matrix of
        partial derivatives for the system of ODEs.

        :getter: Return vectorized symbolic Jacobian matrix.
        :type: function

        """
        if self.__numeric_jacobian is None:
            self.__numeric_jacobian = sym.lambdify(self._symbolic_args,
                                                   self._symbolic_jacobian,
                                                   self._modules)
        return self.__numeric_jacobian

    @property
    def _numeric_system(self):
        """
        Vectorized, numpy-aware function defining the system of ODEs.

        :getter: Return vectorized symbolic system of ODEs.
        :type: function

        """
        if self.__numeric_system is None:
            self.__numeric_system = sym.lambdify(self._symbolic_args,
                                                 self._symbolic_system,
                                                 self._modules)
        return self.__numeric_system

    @property
    def _symbolic_args(self):
        """
        List of symbolic arguments.

        :getter: Return list of symbolic arguments.
        :type: list

        """
        args = [t, X] + sym.symbols(list(self.params.keys()))
        return args

    @property
    def _symbolic_jacobian(self):
        """
        Symbolic Jacobian matrix for the system of ODEs.

        :getter: Return the symbolic Jacobian matrix.
        :type: sympy.Matrix

        """
        N = self._symbolic_system.shape[0]
        return self._symbolic_system.jacobian([X[i] for i in range(N)])

    @property
    def _symbolic_system(self):
        """
        Symbolic matrix defining the system of ODEs.

        :getter: Return the matrix defining the system of ODEs.
        :type: sympy.Matrix

        """
        change_of_vars = self._change_of_variables
        return sym.Matrix([self.model.symbolic_equations]).subs(change_of_vars)

    @property
    def ivp(self):
        r"""
        An instance of the `quantecon.ivp.IVP` class representing the model as
        an initial value problem (IVP).

        :getter: Return an instance of the `ivp.IVP` class.
        :type: ivp.IVP

        """
        if self.with_jacobian:
            tmp_ivp = ivp.IVP(self._numeric_system, self._numeric_jacobian)
        else:
            tmp_ivp = ivp.IVP(self._numeric_system)
        tmp_ivp.f_params = tuple(self.params.values())
        tmp_ivp.jac_params = tuple(self.params.values())
        return tmp_ivp

    @property
    def params(self):
        r"""
        Dictionary of model parameters.

        :getter: Return the current dictionary of model parameters.
        :setter: Set a new dictionary of model parameters.
        :type: dict

        Notes
        -----
        The following parameters are required:
        c : float
            Fecundity scaling factor (converts abstract payoffs to numbers of
            offsping).
        PiAA : float
            Payoff for cooperating when opponent cooperates.
        PiAa : float
            Payoff for cooperating when opponent defects.
        PiaA : float
            Payoff for defecting when opponent cooperates.
        Piaa : float
            Payoff for defecting when opponent defects.

        The model assumes that the payoff structure satisfies the standard
        Prisoner's dilemma conditions which require that

        .. math::
            \Pi_{aA} > \Pi_{AA} > \Pi_{aa} > \Pi_{Aa}

        The user must also specify any additional model parameters specific to
        the chosen functional forms for SGA and Sga.

        """
        return self._params

    @params.setter
    def params(self, value):
        """Set a new parameter dictionary."""
        valid_params = self._validate_params(value)
        self._params = self._order_params(valid_params)

    @staticmethod
    def _order_params(params):
        """Cast a dictionary to an order dictionary."""
        return collections.OrderedDict(sorted(params.items()))

    @staticmethod
    def _validate_params(params):
        """Validate the model parameters."""
        required_params = ['c', 'PiAA', 'PiAa', 'PiaA', 'Piaa']
        if not isinstance(params, dict):
            mesg = "The params attribute must have type dict, not a {}."
            raise AttributeError(mesg.format(params.__class__))
        elif not (set(required_params) <= set(params.keys())):
            mesg = "The params attribute must define the parameters {}."
            raise AttributeError(mesg.format(required_params))
        elif not (params['PiaA'] > params['PiAA'] > params['Piaa'] > params['PiAa']):
            mesg = "Prisoner's dilemma payoff structure not satisfied."
            raise AttributeError(mesg)
        else:
            return params
