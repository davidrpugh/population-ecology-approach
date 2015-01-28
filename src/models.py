import numpy as np
import sympy as sym

# females carrying a particular genotype
females = sym.DeferredVector('f')

# males carrying a particular genotype
males = sym.DeferredVector('M')

# families configurations are represented by a symbolic tensor
U = sym.DeferredVector('U')

# Payoff parameters (from a Prisoner's dilemma)
prisoners_dilemma_payoffs = sym.var('PiaA, PiAA, Piaa, PiAa')
PiaA, PiAA, Piaa, PiAa = prisoners_dilemma_payoffs

# Female fecundity scaling factor
fecundity_factor = sym.var('c')


class Model(object):
    """Base class for all Model objects."""

    def __init__(self, SGA, Sga):
        """
        Create an instance of the Family class.

        Parameters
        ----------
        SGA : sym.Basic
            Symbolic expression for the conditional phenotype matching
            probability for a male carrying the `G` allele of the gamma gene
            and a female carrying the `A` allele of the alpha gene.
        Sga : sym.Basic
            Symbolic expression for the conditional phenotype matching
            probability for a male carrying the `g` allele of the gamma gene
            and a female carrying the `a` allele of the alpha gene.

        """
        self.SGA = self._validate_matching_probability(SGA)
        self.Sga = self._validate_matching_probability(Sga)

    @property
    def altruistic_females(self):
        """
        Number of females carrying the `A` allele of the alpha gene.

        :getter: Return number of femalescarrying the `A` allele
        :type: sym.Symbol

        """
        return females[0] + females[2]

    @property
    def family_genotype_configurations_equations(self):
        """List of equations for the family genotype configurations."""
        raise NotImplementedError

    @property
    def family_unit_symbolic_vars(self):
        """List of symbolic variables for family unit genotype configurations."""
        raise NotImplementedError

    @property
    def female_genotype_equations(self):
        """List of equations for the number of females by genotype."""
        raise NotImplementedError

    @property
    def female_symbolic_vars(self):
        """List of symbolic variables for male genotypes."""
        return [females[i] for i in range(4)]

    @property
    def male_genotype_equations(self):
        """List of equations for the female genotype numbers."""
        raise NotImplementedError

    @property
    def male_symbolic_vars(self):
        """List of symbolic variables for male genotypes."""
        return [males[i] for i in range(4)]

    @property
    def selfish_females(self):
        """
        Number of females carrying the `a` allele of the alpha gene.

        :getter: Return number of females carrying the `a` allele
        :type: sympy.Symbol

        """
        return females[1] + females[3]

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
    def SGA(self, expression):
        """Set new expression for thr conditional probability."""
        self._SGA = expression

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
    def Sga(self, expression):
        """Set new expression for thr conditional probability."""
        self._Sga = expression

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

    @property
    def symbolic_equations(self):
        """
        List of symbolic equations describing the RHS of the system of ODEs.

        :getter: Return the current list of symbolic equations
        :type: list

        """
        equations = (self.male_genotype_equations +
                     self.female_genotype_equations +
                     self.family_genotype_configurations_equations)
        return equations

    @property
    def symbolic_vars(self):
        """
        List of symbolic variables for the model.

        :getter: Return the current list of symbolic variables
        :type: list

        """
        variables = (self.male_symbolic_vars +
                     self.female_symbolic_vars +
                     self.family_unit_symbolic_vars)
        return variables

    @staticmethod
    def _validate_matching_probability(expression):
        """Validate a phenotype matching probability."""
        if not isinstance(expression, sym.Basic):
            mesg = ("Family matching probabilities must have type " +
                    "sympy.Basic, not a {}.")
            raise AttributeError(mesg.format(expression.__class__))
        else:
            return expression

    def females_with_common_allele(self, genotype):
        """
        Number of females who share the same allele of the alpha gene with a
        given genotype.

        Parameters
        ----------
        genotype : int
            Integer index of a valid genotype.

        Returns
        -------
        number_females : sym.Basic
            Symbolic expression for the number of females sharing a common
            allele with a given genotype.

        """
        num_females = (self.iscarrier_A(genotype) * self.altruistic_females +
                       self.iscarrier_a(genotype) * self.selfish_females)
        return num_females

    def genotype_matching_prob(self, male_genotype, female_genotype):
        """
        Probability that a male with a given genotype is matched with a female
        of some other genotype.

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

        Notes
        -----
        Genotype matching probabilities depend on the underlying phenotype
        matching probabilities as well as the population share of females with
        the given genotype.

        """
        i, j = male_genotype, female_genotype
        genotype_matching_prob = (self.phenotype_matching_prob(i, j) *
                                  self.share_females_with_common_allele(j))
        return genotype_matching_prob

    @staticmethod
    def genotype_to_allele_pair(genotype):
        """
        Return the allele pair for a given genotype.

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

    @staticmethod
    def has_common_allele(allele_pair1, allele_pair2):
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
    def has_same_genotype(allele_pair1, allele_pair2):
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
    def inheritance_probability(cls, child_genotype, parent_genotype1, parent_genotype2):
        """
        Probability of a child allele pair given parents' allele pairs.

        Parameters
        ----------
        child_genotype : int
            Integer index of a valid genotype.
        parent_genotype1 : int
            Integer index of a valid genotype.
        parent_genotype2 : int
            Integer index of a valid genotype.

        Returns
        -------
        inheritance_prob : float
            Probability that the child inherits a certain pair of alleles from
            its parents.

        """
        child = cls.genotype_to_allele_pair(child_genotype)
        parent1 = cls.genotype_to_allele_pair(parent_genotype1)
        parent2 = cls.genotype_to_allele_pair(parent_genotype2)

        if cls.has_same_genotype(parent1, parent2):
            if cls.has_same_genotype(child, parent1):
                inheritance_prob = 1.0
            else:
                inheritance_prob = 0.0

        elif cls.has_common_allele(parent1, parent2):
            if cls.has_same_genotype(child, parent1):
                inheritance_prob = 0.5
            elif cls.has_same_genotype(child, parent2):
                inheritance_prob = 0.5
            else:
                inheritance_prob = 0.0

        else:
            inheritance_prob = 0.25

        return inheritance_prob

    @classmethod
    def individual_offspring(cls, genotype1, genotype2):
        """
        Number of offspring produced by a match between individuals carrying
        different genotypes.

        Parameters
        ----------
        genotype1 : int
            Integer index of a valid genotype.
        genotype2 : int
            Integer index of a valid genotype.

        Returns
        -------
        individual_offspring : sym.Basic
            Symbolic expression for the number of offspring produced by female
            with genotype1.

        """
        i, j = genotype1, genotype2
        payoff = (cls.iscarrier_a(i) * cls.iscarrier_A(j) * PiaA +
                  cls.iscarrier_A(i) * cls.iscarrier_A(j) * PiAA +
                  cls.iscarrier_a(i) * cls.iscarrier_a(j) * Piaa +
                  cls.iscarrier_A(i) * cls.iscarrier_a(j) * PiAa)
        individual_offspring = fecundity_factor * payoff
        return individual_offspring

    @classmethod
    def iscarrier_a(cls, genotype):
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
        return 1 - cls.iscarrier_A(genotype)

    @staticmethod
    def iscarrier_A(genotype):
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
    def iscarrier_g(cls, genotype):
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
        return 1 - cls.iscarrier_G(genotype)

    @staticmethod
    def iscarrier_G(genotype):
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

    def phenotype_matching_prob(self, male_genotype, female_genotype):
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
        prob = (self.iscarrier_G(i) * self.iscarrier_A(j) * self.SGA +
                self.iscarrier_G(i) * self.iscarrier_a(j) * self.SGa +
                self.iscarrier_g(i) * self.iscarrier_A(j) * self.SgA +
                self.iscarrier_g(i) * self.iscarrier_a(j) * self.Sga)
        return prob


class OneMaleTwoFemalesModel(Model):
    """Class representing a OneMaleTwoFemales model."""

    _indices = np.arange(0, 64, 1).reshape((4, 4, 4))

    @property
    def family_genotype_configurations_equations(self):
        """
        List of equations for the family genotype configurations.

        :getter: Return the current list of symbolic expressions.
        :type: list

        """
        eqns = []
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    idx = self._indices[i, j, k]
                    U_dot = (males[i] * self.genotype_matching_prob(i, j) *
                             self.genotype_matching_prob(i, k) - U[idx])
                    eqns.append(U_dot)
        return eqns

    @property
    def family_unit_symbolic_vars(self):
        """List of symbolic variables for family unit genotype configurations."""
        return [U[i] for i in range(64)]

    @property
    def female_genotype_equations(self):
        """
        List of equations for the number of females by genotype.

        :getter: Return the current list of equations.
        :type: List

        """
        eqns = []
        for x in range(4):
            terms = []  # container for summation terms
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        idx = self._indices[i, j, k]
                        term = U[idx] * self.expected_number_offspring(x, i, j, k)
                        terms.append(term)

            f_dot = 0.5 * sum(terms) - females[x]
            eqns.append(f_dot)

        return eqns

    @property
    def male_genotype_equations(self):
        """
        List of equations for the male genotype shares.

        :getter: Return the current list of equations.
        :type: List

        """
        eqns = []
        for x in range(4):
            terms = []  # container for summation terms
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        idx = self._indices[i, j, k]
                        term = U[idx] * self.male_survival_probability(x, i, j, k)
                        terms.append(term)

            M_dot = sum(terms) - males[x]
            eqns.append(M_dot)

        return eqns

    def expected_number_offspring(self, genotype, male_genotype,
                                  female_genotype1, female_genotype2):
        """
        Expected number of offspring with a particular genotype given the
        genotype configuration of the family unit.

        Parameters
        ----------
        genotype : int
            Integer index of a valid genotype.
        male_genotype : int
            Integer index of a valid genotype.
        female_genotype1 : int
            Integer index of a valid genotype.
        female_genotype2 : int
            Integer index of a valid genotype.

        Returns:
        --------
        expected_offsping : float
            Expected number of offspring with a particular genotype conditional
            on the genotype configuration of the family unit.

        """
        female_offspring_1 = (self.inheritance_probability(genotype,
                                                           male_genotype,
                                                           female_genotype1) *
                              self.individual_offspring(female_genotype1,
                                                        female_genotype2))

        female_offspring_2 = (self.inheritance_probability(genotype,
                                                           male_genotype,
                                                           female_genotype2) *
                              self.individual_offspring(female_genotype2,
                                                        female_genotype1))
        expected_offspring = female_offspring_1 + female_offspring_2

        return expected_offspring

    def share_females_with_common_allele(self, genotype):
        """
        Ratio of the number of females with a given genotype to the total
        number of females sharing a common allele of the alpha gene with that
        genotype.

        Parameters
        ----------
        genotype : int
            Integer index of a valid genotype.

        Returns
        -------
        share : sym.Basic
            Symbolic expression for the share of females with a common allele
            for a given genotype.

        """
        share = females[genotype] / self.females_with_common_allele(genotype)
        return share

    def male_survival_probability(self, genotype, male_genotype,
                                  female_genotype1, female_genotype2):
        """
        Probabiliy that a male with a given genotype survives given the
        genotype configuration of his family unit.

        Parameters
        ----------
        genotype : int
            Integer index of a valid genotype.
        male_genotype : int
            Integer index of a valid genotype.
        female_genotype1 : int
            Integer index of a valid genotype.
        female_genotype2 : int
            Integer index of a valid genotype.

        Returns
        -------
        survival_prob : float
            Survival probability of a male with a particular genotype.

        """
        survival_prob = (self.expected_number_offspring(genotype,
                                                        male_genotype,
                                                        female_genotype1,
                                                        female_genotype2) /
                         self.total_offspring(female_genotype1,
                                              female_genotype2))
        return survival_prob

    @classmethod
    def total_offspring(cls, female_genotype1, female_genotype2):
        """
        Total number of children produced when a female with genotype1 is
        matched in a family unit with another female with genotype2.

        Parameters
        ----------
        female_genotype1 : int
            Integer index of a valid genotype.
        female_genotype2 : int
            Integer index of a valid genotype.

        Returns
        -------
        total_offspring : sym.Basic
            Symbolic expression for the total number of children produced.

        """
        j, k = female_genotype1, female_genotype2
        total_offspring = (cls.individual_offspring(j, k) +
                           cls.individual_offspring(k, j))
        return total_offspring
