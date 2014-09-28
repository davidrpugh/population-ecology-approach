import sympy as sym

# number of female children of particular genotype
girls = sym.DeferredVector('f')

# number of male adults of particular genotype
men = sym.DeferredVector('M')

# Payoff parameters (from a Prisoner's dilemma)
prisoners_dilemma_payoffs = sym.var('PiaA, PiAA, Piaa, PiAa')
PiaA, PiAA, Piaa, PiAa = prisoners_dilemma_payoffs

# Female fecundity scaling factor
fecundity_factor = sym.var('c')


class Equations(object):
    """Class representing the model equations."""

    @property
    def _altruistic_girls(self):
        """
        Number of female children carrying the `A` allele of the alpha gene.

        """
        return girls[0] + girls[2]

    @property
    def _selfish_girls(self):
        """
        Number of female children carrying the a allele of the alpha gene.

        """
        return girls[1] + girls[3]

    @property
    def SGA(self):
        """
        Conditional probability that a male carrying the `G` allele of the
        gamma gene mates with a female carrying the `A` allele of the alpha
        gene.

        :getter: Return symbolic expression for the conditional probability.
        :setter: Set a new symbolic expression for the conditional probability.
        :type: sym.basic

        """
        return self._SGA

    @SGA.setter
    def SGA(self, value):
        """Set a new symbolic expression for the conditional probability."""
        self._SGA = self._validate_conditional_prob(value)

    @property
    def SGa(self):
        """
        Conditional probability that a male carrying the `G` allele of the
        gamma gene mates with a female carrying the `a` allele of the alpha
        gene.

        :getter: Return symbolic expression for the conditional probability.
        :type: sym.basic

        """
        return 1 - self._SGA

    @property
    def Sga(self):
        """
        Conditional probability that a male carrying the `g` allele of the
        gamma gene mates with a female carrying the `a` allele of the alpha
        gene.

        :getter: Return symbolic expression for the conditional probability.
        :setter: Set a new symbolic expression for the conditional probability.
        :type: sym.basic

        """
        return self._Sga

    @Sga.setter
    def Sga(self, value):
        """Set a new symbolic expression for the conditional probability."""
        self._Sga = self._validate_conditional_prob(value)

    @property
    def SgA(self):
        """
        Conditional probability that a male carrying the `g` allele of the
        gamma gene mates with a female carrying the `A` allele of the alpha
        gene.

        :getter: Return symbolic expression for the conditional probability.
        :type: sym.basic

        """
        return 1 - self._Sga

    def _validate_conditional_prob(self, value):
        """Validate the expression for the conditional matching probability."""
        if not isinstance(value, sym.basic):
            raise AttributeError
        else:
            return value
