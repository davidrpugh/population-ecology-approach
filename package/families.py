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

    def __init__(self, params, SGA, Sga):
        """
        Create an instance of the Model class.

        Parameters
        ----------
        params : dict
            Dictionary of model parameters.
        SGA : sym.Basic
            Symbolic expression for the conditional phenotype matching
            probability for a male carrying the `G` allele of the gamma gene
            and a female carrying the `A` allele of the alpha gene.
        Sga : sym.Basic
            Symbolic expression for the conditional phenotype matching
            probability for a male carrying the `g` allele of the gamma gene
            and a female carrying the `a` allele of the alpha gene.

        """
        self.params = params
        self.SGA = SGA
        self.Sga = Sga

    @property
    def _altruistic_girls(self):
        """
        Number of female children carrying the `A` allele of the alpha gene.

        :getter: Return number of female children carrying the `A` allele
        :type: sym.Symbol

        """
        return girls[0] + girls[2]

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
    def _selfish_girls(self):
        """
        Number of female children carrying the `a` allele of the alpha gene.

        :getter: Return number of female children carrying the `a` allele
        :type: sym.Symbol

        """
        return girls[1] + girls[3]

    @property
    def _symbolic_size(self):
        """
        Symbolic representation of the recurrence relation for family size.

        :getter: Return the symbolic recurrence relation for family size.
        :type: sym.Basic

        """
        return self._family_unit(self.male_genotype, *self.female_genotypes)

    @property
    def female_genotypes(self):
        """

        Integer indices for valid female genotypes.

        :getter: Retun indices for the females' genotypes.
        :setter: Set a new indices for the females' genotypes.
        :type: tuple

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

        """
        return self._male_genotype

    @male_genotype.setter
    def male_genotype(self, genotype):
        """Set a new index for the male genotype."""
        self._male_genotype = self._validate_genotype(genotype)

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
        self._SGA = self._validate_matching_prob(value)
        self._clear_cache()

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
        self._Sga = self._validate_matching_prob(value)
        self._clear_cache()

    @property
    def SgA(self):
        """
        Conditional probability that a male carrying the `g` allele of the
        gamma gene mates with a female carrying the `A` allele of the alpha
        gene.

        :getter: Return symbolic expression for the conditional probability.
        :type: sym.basic

        """
        return 1 - self.Sga

    @property
    def SGa(self):
        """
        Conditional probability that a male carrying the `G` allele of the
        gamma gene mates with a female carrying the `a` allele of the alpha
        gene.

        :getter: Return symbolic expression for the conditional probability.
        :type: sym.basic

        """
        return 1 - self.SGA

    def _clear_cache(self):
        """Clear all cached values."""
        self.__numeric_size = None

    def _family_unit(self, male_genotype, *female_genotypes):
        raise NotImplementedError

    def _genotype_matching_prob(self, male_genotype, female_genotype):
        """
        Conditional probability that an adult male with a given genotype is
        matched with a girl of some other genotype depends on the underlying
        phenotype matching probabilities as well as the population share of
        girls with that genotype.

        Parameters
        ----------
        male_genotype : int
            Integer index of a valid genotype.
        female_genotype : int
            Integer index of a valid genotype.

        Returns
        -------
        genotype_matching_prob: sym.Basic
            Symbolic expression for the conditional genotype matching
            probability.

        """
        i, j = male_genotype, female_genotype
        genotype_matching_prob = (self._phenotype_matching_prob(i, j) *
                                  self._share_girls_with_common_allele(j))
        return genotype_matching_prob

    def _girls_with_common_allele(self, genotype):
        """
        Number of girls who share the same allele of the alpha gene with a
        given genotype.

        Parameters
        ----------
        genotype : int
            Integer index of a valid genotype.

        Returns
        -------
        number_girls : sym.Basic
            Symbolic expression for the number of girls sharing a common allele
            with a given genotype.

        """
        number_girls = (self._iscarrier_A(genotype) * self._altruistic_girls +
                        self._iscarrier_a(genotype) * self._selfish_girls)
        return number_girls

    @classmethod
    def _iscarrier_a(cls, genotype):
        """
        Indicates whether or not adult with a given genotype carries the `a`
        allele.

        Parameters
        ----------
        genotype : int
            Integer index of a valid genotype.

        Returns
        -------
        1 if adult carries the `a` allele, 0 otherwise.

        """
        return 1 - cls._iscarrier_A(genotype)

    @staticmethod
    def _iscarrier_A(genotype):
        """
        Indicates whether or not adult with a given genotype carries the `A`
        allele.

        Parameters
        ----------
        genotype : int
            Integer index of a valid genotype.

        Returns
        -------
        1 if adult carries the `A` allele, 0 otherwise.

        """
        if genotype in [0, 2]:
            return 1
        else:
            return 0

    @classmethod
    def _iscarrier_g(cls, genotype):
        """
        Indicates whether or not adult with a genotype carries the `g` allele.

        Parameters
        ----------
        genotype : int
            Integer index of a valid genotype.

        Returns
        -------
        1 if adult carries the `g` allele, 0 otherwise.

        """
        return 1 - cls._iscarrier_G(genotype)

    @staticmethod
    def _iscarrier_G(genotype):
        """
        Indicates whether or not adult with a genotype carries the `G` allele.

        Parameters
        ----------
        i : int
            Integer index of a valid genotype.

        Returns
        -------
        1 if adult carries the `G` allele, 0 otherwise.

        """
        if genotype in [0, 1]:
            return 1
        else:
            return 0

    def _phenotype_matching_prob(self, male_genotype, female_genotype):
        """
        Conditional probabilities that an adult male expressing the particular
        phenotype associated with its genotype is matched with a girl
        expressing the phenotype associated with some other genotype are
        exogenous.

        Parameters
        ----------
        male_genotype : int
            Integer index of a valid genotype.
        female_genotype : int
            Integer index of a valid genotype.

        Returns
        -------
        prob : sym.Basic
            Symbolic expression for the conditional phenotype matching
            probability.

        """
        i, j = male_genotype, female_genotype
        prob = (self._iscarrier_G(i) * self._iscarrier_A(j) * self.SGA +
                self._iscarrier_G(i) * self._iscarrier_a(j) * self.SGa +
                self._iscarrier_g(i) * self._iscarrier_A(j) * self.SgA +
                self._iscarrier_g(i) * self._iscarrier_a(j) * self.Sga)
        return prob

    def _share_girls_with_common_allele(self, genotype):
        """
        Ratio of the number of girls with a given genotype to the total number
        of girls sharing a common allele of the alpha gene with that genotype.

        Parameters
        ----------
        genotype : int
            Integer index of a valid genotype.

        Returns
        -------
        share : sym.Basic
            Symbolic expression for the share of girls with a common allele for
            a given genotype.

        """
        share = girls[genotype] / self._girls_with_common_allele(genotype)
        return share

    @staticmethod
    def _validate_matching_prob(probability):
        """Validate a phenotype matching probability."""
        if not isinstance(probability, sym.Basic):
            mesg = ("Family matching probabilities must have type " +
                    "sympy.Basic, not a {}.")
            raise AttributeError(mesg.format(probability.__class__))
        else:
            return probability

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

    def _family_unit(self, male_genotype, *female_genotypes):
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
