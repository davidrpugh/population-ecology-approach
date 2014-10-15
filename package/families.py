"""

Notes:

In order to follow DRY, consider...

1. passing model as only parameter to the FamilyUnit constructor.
2. making i, j, k settable properties of the FamilyUnit class.

This would allow a composition relation between Model and FamilyUnit, whilst
still allowing for easy switching between different configurations.

"""
import numpy as np
import sympy as sym

# number of female children of particular genotype
girls = sym.DeferredVector('f')

# number of male adults of particular genotype
men = sym.DeferredVector('M')


class Family(object):
    """Class representing a family unit."""

    __numeric_size = None

    modules = [{'ImmutableMatrix': np.array}, "numpy"]

    @property
    def _numeric_size(self):
        """
        Vectorized function for numerically evaluating family size.

        :getter: Return the current function.
        :type: function.

        """
        if self.__numeric_size is None:
            tmp_args = [men, girls] + list(self.params.keys())
            self.__numeric_size = sym.lambdify(tmp_args,
                                               self._symbolic_size,
                                               self.modules)
        return self.__numeric_size

    @property
    def _symbolic_size(self):
        """
        Symbolic representation of the recurrence relation for family size.

        :getter: Return the symbolic recurrence relation for family size.
        :type: sym.Basic

        """
        return self._family_unit(self.male_genotype, self.female_genotypes)

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

    def family_unit(self, male_genotype, *female_genotypes):
        raise NotImplementedError

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

    def compute_size(self, X):
        """
        Recurrence relation for family size.

        Parameters
        ----------
        X : numpy.ndarray (shape=(8,))
            Array of values for adult males in period t+1 and female children
            in period t.

        Returns
        -------
        size = numpy.ndarray (shape=(1,))
            Size of the family unit period t+1.

        """
        size = self._numeric_size(X[:4], X[4:], **self.params)
        return size.ravel()


class OneMaleTwoFemales(Family):

    def family_unit(self, male_genotype, *female_genotypes):
        """
        A family unit in the 1M2F model is comprised of a single adult male and
        two adult females.

        Parameters
        ----------
        male_genotype : int
            Integer index of a valid genotype.
        female_genotypes : tuple
            Integer indices of valid genotypes.

        Returns
        -------
        U_ijk : sympy.Basic
            Symbolic expression for a family unit in a 1M2F model.

        """
        i = male_genotype
        j, k = female_genotypes

        # size of unit depends on number of males and matching probs
        U_ijk = (men[i] * self._genotype_matching_prob(i, j) *
                 self._genotype_matching_prob(i, k))

        return U_ijk
