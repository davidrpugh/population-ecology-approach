"""

Notes:

In order to follow DRY, consider...

1. passing model as only parameter to the FamilyUnit constructor.
2. making i, j, k settable properties of the FamilyUnit class.

This would allow a composition relation between Model and FamilyUnit, whilst
still allowing for easy switching between different configurations.

"""
import sympy as sym

# number of male adults of particular genotype
men = sym.DeferredVector('M')


class Family(object):
    """Class representing a family in the 1M2F model."""

    def __init__(self, model):
        """
        Creates an instance of the Family class.

        Parameters
        ----------
        model : model.Model
            Instance of the model.Model class representing the 1M2F model.

        """
        self.model = model

    @property
    def male_genotype(self):
        return self._male_genotype

    @male_genotype.setter
    def male_genotype(self, genotype):
        self._male_genotype = self._validate(genotype)

    @property
    def female_genotype_1(self):
        return self._female_genotype_1

    @female_genotype_1.setter
    def female_genotype_1(self, genotype):
        self._female_genotype_1 = self._validate(genotype)

    @property
    def female_genotype_2(self):
        return self._female_genotype_2

    @female_genotype_2.setter
    def female_genotype_2(self, genotype):
        self._female_genotype_2 = self._validate(genotype)

    @property
    def unit(self):
        """
        Family unit in the 1M2F model is comprised of male and two females.

        :getter: Return a symbolic expression for the family unit.
        :type: sym.Basic

        """
        i = self.male_genotype
        j = self.female_genotype_1
        k = self.female_genotype_2

        # size of unit depends on number of males and matching probs
        U_ijk = (men[i] * self._genotype_matching_prob(i, j) *
                 self._genotype_matching_prob(i, k))

        return U_ijk

    @staticmethod
    def _validate(genotype):
        """Validate the genotype."""
        valid_genotypes = range(4)
        if not isinstance(genotype, int):
            mesg = "FamilyUnit genotypes must be an int, not a {}."
            raise AttributeError(mesg.format(genotype.__class__))
        if genotype not in valid_genotypes:
            mesg = "FamilyUnit genotypes must be in {}."
            raise AttributeError(mesg.format(valid_genotypes))
        else:
            return genotype
