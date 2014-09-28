import sympy as sym

# number of female children of particular genotype
girls = sym.DeferredVector('f')

# number of male adults of particular genotype
men = sym.DeferredVector('M')

# Female signaling probabilities
female_signaling_probs = sym.var('dA, da')
dA, da = female_signaling_probs

# Male screening probabilities
male_screening_probs = sym.var('eG, eg')
eG, eg = male_screening_probs

# Payoff parameters (from a Prisoner's dilemma)
prisoners_dilemma_payoffs = sym.var('PiaA, PiAA, Piaa, PiAa')
PiaA, PiAA, Piaa, PiAa = prisoners_dilemma_payoffs

# Female fecundity scaling factor
fecundity_factor = sym.var('c')


class Equations(object):
    """Class representing the model equations."""

    @property
    def _altruist_adoption_pool(self):
        """Number of female children in the "altruistic" adoption pool."""
        return self._true_altruistic_girls + self._mistaken_for_altruistic_girls

    @property
    def _altruistic_girls(self):
        """
        Number of female children carrying the `A` allele of the alpha gene.

        """
        return girls[0] + girls[2]

    @property
    def _false_altruistic_girls(self):
        """
        Number of female children carrying the `a` allele of the alpha gene who
        mimic females carrying the `A` allele of the alpha gene.

        """
        return (1 - da) * self._selfish_girls

    @property
    def _false_selfish_girls(self):
        """
        Number of female children carrying the `A` allele of the alpha gene who
        mimic females carrying the `a` allele of the alpha gene.

        """
        return (1 - dA) * self._altruistic_girls

    @property
    def _mistaken_for_altruistic_girls(self):
        """
        Number of female children mistakenly thought to be carrying the `A`
        allele of the alpha gene by males carrying the `G` allele of the gamma
        gene.

        """
        return (1 - eG) * self._false_altruistic_girls

    @property
    def _mistaken_for_selfish_girls(self):
        """
        Number of female children mistakenly thought to be carrying the `a`
        allele of the alpha gene by males carrying the `g` allele of the gamma
        gene.

        """
        return (1 - eg) * self._false_selfish_girls

    @property
    def _selfish_girls(self):
        """
        Number of female children carrying the a allele of the alpha gene.

        """
        return girls[1] + girls[3]

    @property
    def _selfish_adoption_pool(self):
        """Number of female children in the "selfish" adoption pool."""
        return self._true_selfish_girls + self._mistaken_for_selfish_girls

    @property
    def _true_altruistic_girls(self):
        """
        Number of female children carrying the `A` allele of the alpha gene who
        accurately signal that they are carrying the `A` allele.

        """
        return dA * self._altruistic_girls

    @property
    def _true_selfish_girls(self):
        """
        Number of female children carrying the a allele of the alpha gene who
        accurately signal that they are carrying the a allele.

        """
        return da * self._selfish_girls
