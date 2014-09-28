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
        return self._SGA

    @SGA.setter
    def SGA(self, value):
        self._SGA = value

    @property
    def SGa(self):
        return 1 - self._SGA

    @property
    def Sga(self):
        return self._Sga

    @Sga.setter
    def Sga(self, value):
        self._Sga = value

    @property
    def SgA(self):
        return 1 - self._Sga
