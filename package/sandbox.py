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

        :getter: Return number of female children carrying the `A` allele
        :type: sym.Symbol

        """
        return girls[0] + girls[2]

    @property
    def _selfish_girls(self):
        """
        Number of female children carrying the `a` allele of the alpha gene.

        :getter: Return number of female children carrying the `a` allele
        :type: sym.Symbol

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

    def _genotype_matching_prob(self, i, j):
        """
        Conditional probability that man with genotype i is matched to girl
        with genotype j.

        Parameters
        ----------
        i : int
            Integer index of a valid genotype.
        j : int
            Integer index of a valid genotype.

        Returns
        -------
        probability : sym.Basic
            Symbolic expression for the conditional genotype matching
            probability.

        Notes
        -----
        We index genotypes by integers 0, 1, 2, 3 as follows:

            0 = `GA`, 1 = `Ga`, 2 = `gA`, 3 = `ga`.

        """
        phenotype_matching_prob = self._compute_phenotype_matching_prob(i, j)
        girl_population_share = girls[j] / self._girls_with_common_allele(j)
        probability = phenotype_matching_prob * girl_population_share
        return probability

    def _individual_offspring(self, i, j):
        """
        Number of offspring produced by a woman with genotype i when matched in
        family unit with another woman with genotype j.

        Parameters
        ----------
        i : int
            Integer index of a valid genotype.
        j : int
            Integer index of a valid genotype.

        Returns
        -------
        individual_offspring : sym.Basic
            Symbolic expression for the number of offspring produced by female
            with genotype i.

        Notes
        -----
        We index genotypes by integers 0, 1, 2, 3 as follows:

            0 = `GA`, 1 = `Ga`, 2 = `gA`, 3 = `ga`.

        """
        payoff = (self.iscarrier_a(i) * self.iscarrier_A(j) * PiaA +
                  self.iscarrier_A(i) * self.iscarrier_A(j) * PiAA +
                  self.iscarrier_a(i) * self.iscarrier_a(j) * Piaa +
                  self.iscarrier_A(i) * self.iscarrier_a(j) * PiAa)
        individual_offspring = fecundity_factor * payoff
        return individual_offspring

    def _girls_with_common_allele(self, i):
        """
        Number of girls who share common allele with genotype i.

        Parameters
        ----------
        i : int
            Integer index of a valid genotype.

        Returns
        -------
        number_girls : sym.Basic
            Symbolic expression for the number of girls sharing a common allele
            with genotype i.

        Notes
        -----
        We index genotypes by integers 0, 1, 2, 3 as follows:

            0 = `GA`, 1 = `Ga`, 2 = `gA`, 3 = `ga`.

        """
        number_girls = (self._iscarrier_A(i) * self._altruistic_girls +
                        self._iscarrier_a(i) * self._selfish_girls)
        return number_girls

    def _iscarrier_G(self, i):
        """
        Indicates whether or not adult with genotype i carries the `G` allele.

        Parameters
        ----------
        i : int
            Integer index of a valid genotype.

        Returns
        -------
        1 if adult carries the `G` allele, 0 otherwise.

        Notes
        -----
        We index genotypes by integers 0, 1, 2, 3 as follows:

            0 = `GA`, 1 = `Ga`, 2 = `gA`, 3 = `ga`.

        """
        if i in [0, 1]:
            return 1
        else:
            return 0

    def _iscarrier_g(self, i):
        """
        Indicates whether or not adult with genotype i carries the `g` allele.

        Parameters
        ----------
        i : int
            Integer index of a valid genotype.

        Returns
        -------
        1 if adult carries the `g` allele, 0 otherwise.

        Notes
        -----
        We index genotypes by integers 0, 1, 2, 3 as follows:

            0 = `GA`, 1 = `Ga`, 2 = `gA`, 3 = `ga`.

        """
        return 1 - self._iscarrier_G(i)

    def _iscarrier_A(self, i):
        """
        Indicates whether or not adult with genotype i carries the `A` allele.

        Parameters
        ----------
        i : int
            Integer index of a valid genotype. Must take values 0,1,2,3.

        Returns
        -------
        1 if adult carries the `A` allele, 0 otherwise.

        Notes
        -----
        We index genotypes by integers 0, 1, 2, 3 as follows:

            0 = `GA`, 1 = `Ga`, 2 = `gA`, 3 = `ga`.

        """
        if i in [0, 2]:
            return 1
        else:
            return 0

    def _iscarrier_a(self, i):
        """
        Indicates whether or not adult with genotype i carries the `a` allele.

        Parameters
        ----------
        i : int
            Integer index of a valid genotype.

        Returns
        -------
        1 if adult carries the `a` allele, 0 otherwise.

        Notes
        -----
        We index genotypes by integers 0, 1, 2, 3 as follows:

            0 = `GA`, 1 = `Ga`, 2 = `gA`, 3 = `ga`.

        """
        return 1 - self.iscarrier_A(i)

    def _phenotype_matching_prob(self, i, j):
        """
        Conditional probability that male with phenotype i is matched to girl
        with phenotype j.

        Parameters
        ----------
        i : int
            Integer index of a valid genotype.
        j : int
            Integer index of a valid genotype.

        Returns
        -------
        probability : sym.Basic
            Symbolic expression for the conditional phenotype matching
            probability.

        Notes
        -----
        We index genotypes by integers 0, 1, 2, 3 as follows:

            0 = `GA`, 1 = `Ga`, 2 = `gA`, 3 = `ga`.

        """
        probability = (self._iscarrier_G(i) * self._iscarrier_A(j) * self.SGA +
                       self._iscarrier_G(i) * self._iscarrier_a(j) * self.SGa +
                       self._iscarrier_g(i) * self._iscarrier_A(j) * self.SgA +
                       self._iscarrier_g(i) * self._iscarrier_a(j) * self.Sga)
        return probability

    def _offspring_share(self, i, j):
        """
        Share of total offspring produced by woman with genotype i when matched
        in a family unit with a woman with genotype j.

        Parameters
        ----------
        i : int
            Integer index of a valid genotype.
        j : int
            Integer index of a valid genotype.

        Returns
        -------
        offspring_share : sym.Basic
            Symbolic expression for the share of total offspring in a family
            unit produced by female with genotype i.

        Notes
        -----
        We index genotypes by integers 0, 1, 2, 3 as follows:

            0 = `GA`, 1 = `Ga`, 2 = `gA`, 3 = `ga`.

        """
        offspring_share = (self._compute_individual_offspring(i, j) /
                           self._compute_total_offspring(i, j))
        return offspring_share

    def _total_offspring(self, i, j):
        """
        Total number of children produced when a woman with genotype i is
        matched in a family unit with another woman with genotype j.

        Parameters
        ----------
        i : int
            Integer index of a valid genotype.
        j : int
            Integer index of a valid genotype.

        Returns
        -------
        total_offspring : sym.Basic
            Symbolic expression for the total number of children produced.

        Notes
        -----
        We index genotypes by integers 0, 1, 2, 3 as follows:

            0 = `GA`, 1 = `Ga`, 2 = `gA`, 3 = `ga`.

        """
        total_offspring = (self._compute_individual_offspring(i, j) +
                           self._compute_individual_offspring(j, i))
        return total_offspring

    def _validate_conditional_prob(self, value):
        """Validate the expression for the conditional matching probability."""
        if not isinstance(value, sym.Basic):
            raise AttributeError
        else:
            return value
