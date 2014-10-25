"""

@author: David R. Pugh
@date: 2014-10-20

"""
import numpy as np
import pandas as pd
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


class Family(object):
    """Class representing a family unit."""

    _modules = [{'ImmutableMatrix': np.array}, "numpy"]

    def __init__(self, params, SGA, Sga):
        """
        Create an instance of the Family class.

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
    def _numeric_jacobian(self):
        """
        Vectorized function for numerically evaluating the Jacobian matrix of
        partial derivatives.

        :getter: Return the current function.
        :type: function.

        """
        if self.__numeric_jacobian is None:
            self.__numeric_jacobian = sym.lambdify(self._symbolic_args,
                                                   self._symbolic_jacobian,
                                                   self._modules)
        return self.__numeric_jacobian

    @property
    def _numeric_size(self):
        """
        Vectorized function for numerically evaluating family size.

        :getter: Return the current function.
        :type: function.

        """
        if self.__numeric_size is None:
            self.__numeric_size = sym.lambdify(self._symbolic_args,
                                               self._symbolic_size,
                                               self._modules)
        return self.__numeric_size

    @property
    def _numeric_system(self):
        """
        Vectorized function for numerically evaluating the system of recurrence
        relations.

        :getter: Return the current function.
        :type: function.

        """
        if self.__numeric_system is None:
            self.__numeric_system = sym.lambdify(self._symbolic_args,
                                                 self._symbolic_system,
                                                 self._modules)
        return self.__numeric_system

    @property
    def _selfish_girls(self):
        """
        Number of female children carrying the `a` allele of the alpha gene.

        :getter: Return number of female children carrying the `a` allele
        :type: sympy.Symbol

        """
        return girls[1] + girls[3]

    @property
    def _symbolic_args(self):
        """
        List of symbolic endogenous variables and parameters..

        :getter: Return list of symbolic endogenous variables
        :type: list

        """
        return [men, girls] + sym.var(list(self.params.keys()))

    @property
    def _symbolic_jacobian(self):
        """
        Symbolic representation of the Jacobian matrix of partial derivatives
        for the system of recurrence relations.

        :getter: Return Jacobian matrix of partial derivatives.
        :type: sympy.Matrix

        """
        return self._symbolic_system.jacobian(self._symbolic_vars)

    @property
    def _symbolic_size(self):
        """
        Symbolic representation of the recurrence relation for family size.

        :getter: Return the symbolic recurrence relation for family size.
        :type: sym.Basic

        """
        return self._family_unit(self.male_genotype, *self.female_genotypes)

    @property
    def _symbolic_system(self):
        """
        Symbolic representation of the system of recurrence relations for male
        adults and female children of various genotypes.

        :getter: Return the symbolic system of recurrence relations.
        :type: sympy.Matrix

        """
        male_eqns = [self._recurrence_relation_men(x) for x in range(4)]
        female_eqns = [self._recurrence_relation_girls(x) for x in range(4)]
        return sym.Matrix(male_eqns + female_eqns)

    @property
    def _symbolic_vars(self):
        """
        List of symbolic endogenous variables.

        :getter: Return list of symbolic endogenous variables
        :type: list

        """
        adult_males = [men[i] for i in range(4)]
        female_children = [girls[j] for j in range(4)]
        return adult_males + female_children

    @property
    def configurations(self):
        """
        List of valid genotype configurations for a Family.

        :getter: Return the current list of valid configurations.
        :type: pandas.MultiIndex

        """
        raise NotImplementedError

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
        self._clear_cache()

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
        self._clear_cache()

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
        """Set a new dictionary of model parameters."""
        self._params = self._validate_params(value)

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
        self.__numeric_jacobian = None
        self.__numeric_size = None
        self.__numeric_system = None

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

    @staticmethod
    def _genotype_to_allele_pair(genotype):
        """
        Return allele pair for a given genotype.

        Parameters
        ----------
        i : int
            Integer index of a valid genotype.

        Returns
        -------
        allele_pair : tuple (size=2)
            Tuple of the form `(q, r)` where `q` indexes the gamma gene and `r`
            indexes the alpha gene.

        Notes
        -----
        Our allele index `(q, r)` where `q` indexes the gamma gene and `r`
        indexes the alpha gene uses the following mapping:

            `q=0=G, q=1=g, r=0=A, r=1=a`.

        For examples, an allele index of (0, 1) indicates that the host carrys
        the `G` allele of the gamma gene and the `a` allele of the alpha gene.

        """
        if genotype == 0:
            allele_pair = (0, 0)
        elif genotype == 1:
            allele_pair = (0, 1)
        elif genotype == 2:
            allele_pair = (1, 0)
        else:
            allele_pair = (1, 1)

        return allele_pair

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

    @staticmethod
    def _has_common_allele(allele_pair1, allele_pair2):
        """
        Check if two allele pairs have a common allele.

        Parameters
        ----------
        allele_pair1 : tuple (size=2)
            Tuple of the form `(q, r)` where `q` indexes the gamma gene and `r`
            indexes the alpha gene.
        allele_pair2 : tuple (size=2)
            Tuple of the form `(q, r)` where `q` indexes the gamma gene and `r`
            indexes the alpha gene.

        Returns
        -------
        True if two genotypes share a common allele; false otherwise.

        """
        for allele1, allele2 in zip(allele_pair1, allele_pair2):
            if allele1 == allele2:
                return True

        else:
            return False

    @staticmethod
    def _has_same_genotype(allele_pair1, allele_pair2):
        """
        Return true if two genotypes are a perfect match.

        Parameters
        ----------
        allele_pair1 : tuple (size=2)
            Tuple of the form `(q, r)` where `q` indexes the gamma gene and `r`
            indexes the alpha gene.
        allele_pair2 : tuple (size=2)
            Tuple of the form `(q, r)` where `q` indexes the gamma gene and `r`
            indexes the alpha gene.

        Returns
        -------
        True if two genotypes are a perfect match; false otherwise.

        """
        if allele_pair1 == allele_pair2:
            return True
        else:
            return False

    @classmethod
    def _individual_offspring(cls, female_genotype_1, female_genotype_2):
        """
        Number of offspring produced by a woman with genotype1 when matched in
        family unit with another woman with genotype2.

        Parameters
        ----------
        female_genotype_1 : int
            Integer index of a valid genotype.
        female_genotype_2 : int
            Integer index of a valid genotype.

        Returns
        -------
        individual_offspring : sym.Basic
            Symbolic expression for the number of offspring produced by female
            with genotype1.

        """
        j, k = female_genotype_1, female_genotype_2
        payoff = (cls._iscarrier_a(j) * cls._iscarrier_A(k) * PiaA +
                  cls._iscarrier_A(j) * cls._iscarrier_A(k) * PiAA +
                  cls._iscarrier_a(j) * cls._iscarrier_a(k) * Piaa +
                  cls._iscarrier_A(j) * cls._iscarrier_a(k) * PiAa)
        individual_offspring = fecundity_factor * payoff
        return individual_offspring

    @classmethod
    def _inheritance_prob(cls, child, parent1, parent2):
        """
        Conditional probability of child's allele pair given parents' allele
        pairs.

        Parameters
        ----------
        child : tuple (size=2)
            Tuple of the form `(q, r)` where `q` indexes the gamma gene and `r`
            indexes the alpha gene.
        parent1 : tuple (size=2)
            Tuple of the form `(q, r)` where `q` indexes the gamma gene and `r`
            indexes the alpha gene.
        parent2 : tuple (size=2)
            Tuple of the form `(q, r)` where `q` indexes the gamma gene and `r`
            indexes the alpha gene.

        Returns
        -------
        inheritance_prob : float
            Probability that the child inherits a certain pair of alleles from
            its parents.

        """
        if cls._has_same_genotype(parent1, parent2):
            if cls._has_same_genotype(child, parent1):
                inheritance_prob = 1.0
            else:
                inheritance_prob = 0.0

        elif cls._has_common_allele(parent1, parent2):
            if cls._has_same_genotype(child, parent1):
                inheritance_prob = 0.5
            elif cls._has_same_genotype(child, parent2):
                inheritance_prob = 0.5
            else:
                inheritance_prob = 0.0

        else:
            inheritance_prob = 0.25

        return inheritance_prob

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
        genotype : int
            Integer index of a valid genotype.

        Returns
        -------
        1 if adult carries the `G` allele, 0 otherwise.

        """
        if genotype in [0, 1]:
            return 1
        else:
            return 0

    @classmethod
    def _offspring_share(cls, female_genotype_1, female_genotype_2):
        """
        Share of total offspring produced by woman with genotype_1 when matched
        in a family unit with a woman with genotype_2.

        Parameters
        ----------
        female_genotype_1 : int
            Integer index of a valid genotype.
        female_genotype_2 : int
            Integer index of a valid genotype.

        Returns
        -------
        offspring_share : sym.Basic
            Symbolic expression for the share of total offspring in a family
            unit produced by female with genotype i.

        """
        j, k = female_genotype_1, female_genotype_2
        offspring_share = (cls._individual_offspring(j, k) /
                           cls._total_offspring(j, k))
        return offspring_share

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

    def _recurrence_relation_girls(self, genotype):
        """
        Recurrence relation for number of female childen with a given genotype.

        Parameters
        ----------
        genotype : int
            Integer index of a valid genotype.

        Returns
        -------
        reccurence_relation : sympy.basic
            Symbolic expression for the recurrence relation.

        """
        raise NotImplementedError

    def _recurrence_relation_men(self, genotype):
        """
        Recurrence relation for number of male adults with a given genotype.

        Parameters
        ----------
        genotype : int
            Integer index of a valid genotype.

        Returns
        -------
        relation : sympy.basic
            Symbolic expression for the recurrence relation.

        """
        raise NotImplementedError

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

    @classmethod
    def _total_offspring(cls, female_genotype_1, female_genotype_2):
        """
        Total number of children produced when a woman with genotype_1 is
        matched in a family unit with another woman with genotype_2.

        Parameters
        ----------
        Parameters
        ----------
        female_genotype_1 : int
            Integer index of a valid genotype.
        female_genotype_2 : int
            Integer index of a valid genotype.

        Returns
        -------
        total_offspring : sym.Basic
            Symbolic expression for the total number of children produced.

        """
        j, k = female_genotype_1, female_genotype_2
        total_offspring = (cls._individual_offspring(j, k) +
                           cls._individual_offspring(k, j))
        return total_offspring

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

    def _recurrence_relation_girls(self, genotype):
        """
        Recurrence relation for number of female childen with a given genotype.

        Parameters
        ----------
        genotype : int
            Integer index of a valid genotype.

        Returns
        -------
        reccurence_relation : sympy.basic
            Symbolic expression for the recurrence relation.

        """
        terms = []
        for i in range(4):
            for j in range(4):
                for k in range(4):

                    # expected number of offspring of i and j
                    tmp_child = self._genotype_to_allele_pair(genotype)
                    tmp_father = self._genotype_to_allele_pair(i)
                    tmp_mother = self._genotype_to_allele_pair(j)
                    tmp_daughters_ij = 0.5 * self._individual_offspring(j, k)
                    tmp_ij = (self._inheritance_prob(tmp_child,
                                                     tmp_father,
                                                     tmp_mother) *
                              tmp_daughters_ij)

                    # expected genotype of offspring of i and k
                    tmp_mother = self._genotype_to_allele_pair(k)
                    tmp_daughters_ik = 0.5 * self._individual_offspring(k, j)
                    tmp_ik = (self._inheritance_prob(tmp_child,
                                                     tmp_father,
                                                     tmp_mother) *
                              tmp_daughters_ik)

                    # expected genotype of offspring of family unit
                    tmp_term = self._family_unit(i, j, k) * (tmp_ij + tmp_ik)

                    terms.append(tmp_term)

        recurrence_relation = sum(terms)

        return recurrence_relation

    def _recurrence_relation_men(self, genotype):
        """
        Recurrence relation for number of male adults with a given genotype.

        Parameters
        ----------
        genotype : int
            Integer index of a valid genotype.

        Returns
        -------
        relation : sympy.basic
            Symbolic expression for the recurrence relation.

        """
        terms = []
        for i in range(4):
            for j in range(4):
                for k in range(4):

                    # expected genotype of offspring of i and j
                    tmp_child = self._genotype_to_allele_pair(genotype)
                    tmp_father = self._genotype_to_allele_pair(i)
                    tmp_mother = self._genotype_to_allele_pair(j)
                    tmp_ij = (self._inheritance_prob(tmp_child,
                                                     tmp_father,
                                                     tmp_mother) *
                              self._offspring_share(j, k))

                    # expected genotype of offspring of i and k
                    tmp_mother = self._genotype_to_allele_pair(k)
                    tmp_ik = (self._inheritance_prob(tmp_child,
                                                     tmp_father,
                                                     tmp_mother) *
                              self._offspring_share(k, j))

                    tmp_term = self._family_unit(i, j, k) * (tmp_ij + tmp_ik)

                    terms.append(tmp_term)

        recurrence_relation = sum(terms)

        return recurrence_relation

    @property
    def configurations(self):
        """
        List of valid genotype configurations for a Family.

        :getter: Return the current list of valid configurations.
        :type: pandas.MultiIndex

        """
        headers = ['male_genotype', 'female1_genotype', 'female2_genotype']
        configs = pd.MultiIndex.from_product([range(4), range(4), range(4)],
                                             names=headers)
        return configs
