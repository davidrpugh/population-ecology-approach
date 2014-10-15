"""

Notes:

In order to follow DRY, consider...

1. passing model as only parameter to the FamilyUnit constructor.
2. making i, j, k settable properties of the FamilyUnit class.

This would allow a composition relation between Model and FamilyUnit, whilst
still allowing for easy switching between different configurations.

"""
import sympy as sym

# number of female children of particular genotype
girls = sym.DeferredVector('f')

# number of male adults of particular genotype
men = sym.DeferredVector('M')


class Family(object):
    """Class representing a family unit."""

    @property
    def female_genotypes(self):
        """

        Integer indices for valid female genotypes.

        :getter: Retun indices for the females' genotypes.
        :setter: Set a new indices for the females' genotypes.
        :type: tuple

        Notes
        -----
        We index genotypes by integers 0, 1, 2, 3 as follows:

            0 = `GA`, 1 = `Ga`, 2 = `gA`, 3 = `ga`.

        """
        return self._female_genotypes

    @female_genotypes.setter
    def female_genotypes(self, genotypes):
        """Set new indices for female genotypes."""
        self._female_genotypes = self._validate_female_genotypes(genotypes)

    @property
    def male_genotype(self):
        """
        Integer index for a valid male genotype.

        :getter: Return the index of the male's genotype.
        :setter: Set a new index for the male genotype.
        :type: int

        Notes
        -----
        We index genotypes by integers 0, 1, 2, 3 as follows:

            0 = `GA`, 1 = `Ga`, 2 = `gA`, 3 = `ga`.

        """
        return self._male_genotype

    @male_genotype.setter
    def male_genotype(self, genotype):
        """Set a new index for the male genotype."""
        self._male_genotype = self._validate_genotype(genotype)

    @classmethod
    def _validate_female_genotypes(cls, genotypes):
        """Validates the females_genotypes attribute."""
        return tuple(cls._validate_genotype(genotype) for genotype in genotypes)

    @staticmethod
    def _validate_genotype(genotype):
        """Validates a genotype."""
        valid_genotypes = range(4)
        if not isinstance(genotype, int):
            mesg = "Genotype indices must have type int, not {}."
            raise AttributeError(mesg.format(genotype.__class__))
        if genotype not in valid_genotypes:
            mesg = "Valid genotype indices are {}."
            raise AttributeError(mesg.format(valid_genotypes))
        else:
            return genotype
