import numpy as np
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


class Model(object):
    """Base class representing the model of Pugh-Schaffer-Seabright."""

    def __init__(self, params, SGA, Sga):
        """
        Create and instance of the Model class.

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
        self.__numeric_system = None
        self.__numeric_jacobian = None

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
    def _selfish_girls(self):
        """
        Number of female children carrying the `a` allele of the alpha gene.

        :getter: Return number of female children carrying the `a` allele
        :type: sym.Symbol

        """
        return girls[1] + girls[3]

    @property
    def _numeric_jacobian(self):
        if self.__numeric_jacobian is None:
            tmp_args = [men, girls] + list(self.params.keys())
            self.__numeric_jacobian = sym.lambdify(tmp_args,
                                                   self._symbolic_jacobian,
                                                   [{'ImmutableMatrix': np.array}, "numpy"])
        return self.__numeric_system

    @property
    def _numeric_system(self):
        if self.__numeric_system is None:
            tmp_args = [men, girls] + list(self.params.keys())
            self.__numeric_system = sym.lambdify(tmp_args,
                                                 self._symbolic_system,
                                                 [{'ImmutableMatrix': np.array}, "numpy"])
        return self.__numeric_system

    @property
    def _symbolic_jacobian(self):
        """
        Jacobian matrix of partial derivatives for the system of equations
        defining the evolution of the numbers of male adults and female
        children of various genotypes.

        :getter: Return Jacobian matrix of partial derivatives.
        :type: sym.Matrix

        """
        adult_males = [men[i] for i in range(4)]
        female_children = [girls[j] for j in range(4)]
        endog_vars = adult_males + female_children
        return self._symbolic_system.jacobian(endog_vars)

    @property
    def _symbolic_system(self):
        """
        Symbolic system of equations defining the evolution of the numbers of
        male adults and female children of various genotypes.

        :getter: Return the symbolic system of recurrence relations.
        :type: sym.Matrix

        """
        male_eqns = [self._recurrence_relations_males(x) for x in range(4)]
        female_eqns = [self._recurrence_relations_females(x) for x in range(4)]
        return sym.Matrix(male_eqns + female_eqns)

    @property
    def initial_condition(self):
        """
        Initial condition for a simulation.

        :getter: Return an array defining the initial condition.
        :setter: Set the initial number of males with genotype `GA`.
        :type: float

        """
        mGA0 = self._initial_condition
        mga0 = 1 - mGA0

        initial_males = np.array([mGA0, 0, 0, mga0])

        # f_GA(0)=mGA0*Pi_AA and f_ga(0)=mga0*Pi_aa.
        fGA0 = self.params['c'] * self.params['PiAA'] * mGA0
        fga0 = self.params['c'] * self.params['Piaa'] * mga0
        initial_females = np.array([fGA0, 0.0, 0.0, fga0])

        return np.hstack((initial_males, initial_females))

    @initial_condition.setter
    def initial_condition(self, value):
        self._initial_condition = value

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
        self._SGA = self._validate_conditional_prob(value)

        # clear the cache
        self.__numeric_jacobian = None
        self.__numeric_system = None

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

        # clear the cache
        self.__numeric_jacobian = None
        self.__numeric_system = None

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

    def _family_unit(self, i, j, k):
        """
        Family unit comprised of male with genoytpe i, and females with
        genotypes j and k.

        """
        raise NotImplementedError

    @classmethod
    def _genotype_matching_prob(cls, i, j):
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
        phenotype_matching_prob = cls._phenotype_matching_prob(i, j)
        girl_population_share = girls[j] / cls._girls_with_common_allele(j)
        probability = phenotype_matching_prob * girl_population_share
        return probability

    @staticmethod
    def _genotype_to_allele_pair(i):
        """
        Return allele pair for a given genotype i.

        Parameters
        ----------
        i : int
            Integer index of a valid genotype. Must take values 0,1,2,3.

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
        if i == 0:
            allele_pair = (0, 0)
        elif i == 1:
            allele_pair = (0, 1)
        elif i == 2:
            allele_pair = (1, 0)
        else:
            allele_pair = (1, 1)

        return allele_pair

    @classmethod
    def _girls_with_common_allele(cls, i):
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
        number_girls = (cls._iscarrier_A(i) * cls._altruistic_girls +
                        cls._iscarrier_a(i) * cls._selfish_girls)
        return number_girls

    @staticmethod
    def _has_common_allele(allele_pair1, allele_pair2):
        """
        Check if two allele pairs have a common allele.

        Parameters
        ----------
        allele_pair1 : tuple (size=2)
        allele_pair2 : tuple (size=2)

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
        allele_pair1 : tuple (size=2)

        Returns
        -------
        True if two genotypes are a perfect match; false otherwise.

        """
        if allele_pair1 == allele_pair2:
            return True
        else:
            return False

    @classmethod
    def _individual_offspring(cls, i, j):
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
        payoff = (cls._iscarrier_a(i) * cls._iscarrier_A(j) * PiaA +
                  cls._iscarrier_A(i) * cls._iscarrier_A(j) * PiAA +
                  cls._iscarrier_a(i) * cls._iscarrier_a(j) * Piaa +
                  cls._iscarrier_A(i) * cls._iscarrier_a(j) * PiAa)
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
        parent1 : tuple (size=2)
        parent2 : tuple (size=2)

        Returns
        -------
        inheritance_prob : float

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

    @staticmethod
    def _iscarrier_G(i):
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

    @classmethod
    def _iscarrier_g(cls, i):
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
        return 1 - cls._iscarrier_G(i)

    @staticmethod
    def _iscarrier_A(i):
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

    @classmethod
    def _iscarrier_a(cls, i):
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
        return 1 - cls._iscarrier_A(i)

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

    @classmethod
    def _offspring_share(cls, i, j):
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
        offspring_share = (cls._individual_offspring(i, j) /
                           cls._total_offspring(i, j))
        return offspring_share

    def _recurrence_relations_females(self, l):
        """Recurrence relation for the number of female offspring with genotype l."""
        raise NotImplementedError

    def _recurrence_relations_males(self, l):
        """Recurrence relation for the number of male adults with genotype l."""
        raise NotImplementedError

    def _simulate_fixed_trajectory(self, initial_condition, T):
        """Simulates a trajectory of fixed length."""
        # set up the trajectory array
        traj = np.empty((8, T))
        traj[:, 0] = initial_condition

        # run the simulation
        for t in range(1, T):
            current_shares = traj[:, t-1]
            new_shares = self.F(current_shares)
            traj[:, t] = new_shares

        return traj

    def _simulate_variable_trajectory(self, initial_condition, rtol):
        """Simulates a trajectory of variable length."""
        # set up the trajectory array
        traj = initial_condition.reshape((8, 1))

        # initialize delta
        delta = np.ones(8)

        # run the simulation
        while np.any(np.greater(delta, rtol)):
            current_shares = traj[:, -1]
            new_shares = self.F(current_shares)
            delta = np.abs(new_shares - current_shares)

            # update the trajectory
            traj = np.hstack((traj, new_shares[:, np.newaxis]))

        return traj

    @classmethod
    def _total_offspring(cls, i, j):
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
        total_offspring = (cls._individual_offspring(i, j) +
                           cls._individual_offspring(j, i))
        return total_offspring

    @staticmethod
    def _validate_conditional_prob(value):
        """Validate the expression for the conditional matching probability."""
        if not isinstance(value, sym.Basic):
            raise AttributeError
        else:
            return value

    @staticmethod
    def _validate_params(params):
        """Validate the model parameters."""
        required_params = ['c', 'PiAA', 'PiAa', 'PiaA', 'Piaa']
        if not isinstance(params, dict):
            mesg = "Model.params must be a dict, not a {}."
            raise AttributeError(mesg.format(params.__class__))
        if not set(required_params) < set(params.keys()):
            mesg = "Model.params must contain the required parameters {}."
            raise AttributeError(mesg.format(required_params))
        if not params['PiaA'] > params['PiAA'] > params['Piaa'] > params['PiAa']:
            mesg = "Prisoner's dilemma payoff structure not satisfied."
            raise AttributeError(mesg)
        else:
            return params

    def F(self, X):
        """Equation of motion for population allele shares."""
        out = self._numeric_system(X[:4], X[4:], **self.params)
        return out.ravel()

    def F_jacobian(self, X):
        """Jacobian for equation of motion."""
        jac = self._numeric_jacobian(X[:4], X[4:], **self.params)
        return jac

    def simulate(self, T=None, rtol=None):
        """Simulates a run of the model given some initial_condition."""
        if T is not None:
            traj = self._simulate_fixed_trajectory(self.initial_condition, T)
        elif rtol is not None:
            traj = self._simulate_variable_trajectory(self.initial_condition, rtol)
        else:
            raise ValueError("One of 'T' or 'rtol' must be specified.")
        return traj


class OneMaleTwoFemalesModel(Model):
    """Class representing the 1M2F model of Pugh-Schaffer-Seabright."""

    def _family_unit(self, i, j, k):
        """
        Family unit comprised of male with genoytpe i, and females with
        genotypes j and k.

        Parameters
        ----------
        i : int
            Integer index of a valid genotype.
        j : int
            Integer index of a valid genotype.
        k : int
            Integer index of a valid genotype.

        Returns
        -------
        U_ijk : sym.Basic
            Symbolic expression for a family unit comprised of a male with
            genoytpe i, and two females with genotypes j and k.

        """
        U_ijk = (men[i] * self._genotype_matching_prob(i, j) *
                 self._genotype_matching_prob(i, k))
        return U_ijk

    def _recurrence_relations_females(self, l):
        """
        Recurrence relation for the number of female offspring with genotype l.

        Parameters
        ----------
        l : int
            Integer index of a valid genotype.

        Returns
        -------
        recurrence_relation : sym.Basic
            Symbolic expression for the recurrence relation describing the
            evolution of the number of female offspring with genotype i.

        Notes
        -----
        We index genotypes by integers 0, 1, 2, 3 as follows:

            0 = `GA`, 1 = `Ga`, 2 = `gA`, 3 = `ga`.

        """
        terms = []
        for i in range(4):
            for j in range(4):
                for k in range(4):

                    # configuration of family unit
                    tmp_family_unit = self._family_unit(i, j, k)
                    tmp_child_allele_pair = self._genotype_to_allele_pair(l)
                    tmp_father_allele_pair = self._genotype_to_allele_pair(i)

                    # expected genotype of offspring of i and j
                    tmp_mother_allele_pair = self._genotype_to_allele_pair(j)
                    tmp_daughters_ij = 0.5 * self._individual_offspring(j, k)
                    tmp_ij = (self._inheritance_prob(tmp_child_allele_pair,
                                                     tmp_father_allele_pair,
                                                     tmp_mother_allele_pair) *
                              tmp_daughters_ij)

                    # expected genotype of offspring of i and k
                    tmp_mother_allele_pair = self._genotype_to_allele_pair(k)
                    tmp_daughters_ik = 0.5 * self._individual_offspring(k, j)
                    tmp_ik = (self._inheritance_prob(tmp_child_allele_pair,
                                                     tmp_father_allele_pair,
                                                     tmp_mother_allele_pair) *
                              tmp_daughters_ik)

                    # expected genotype of offspring of family unit
                    tmp_term = tmp_family_unit * (tmp_ij + tmp_ik)

                    terms.append(tmp_term)

        recurrence_relation = sum(terms)

        return recurrence_relation

    def _recurrence_relations_males(self, l):
        """
        Recurrence relation for the number of male adults with genotype l.

        Parameters
        ----------
        l : int
            Integer index of a valid genotype.

        Returns
        -------
        recurrence_relation : sym.Basic
            Symbolic expression for the recurrence relation describing the
            evolution of the number of adult males with genotype i.

        Notes
        -----
        We index genotypes by integers 0, 1, 2, 3 as follows:

            0 = `GA`, 1 = `Ga`, 2 = `gA`, 3 = `ga`.

        """
        terms = []
        for i in range(4):
            for j in range(4):
                for k in range(4):

                    # configuration of family unit
                    tmp_family_unit = self._family_unit(i, j, k)
                    tmp_child_allele_pair = self._genotype_to_allele_pair(l)
                    tmp_father_allele_pair = self._genotype_to_allele_pair(i)

                    # expected genotype of offspring of i and j
                    tmp_mother_allele_pair = self._genotype_to_allele_pair(j)
                    tmp_ij = (self._inheritance_prob(tmp_child_allele_pair,
                                                     tmp_father_allele_pair,
                                                     tmp_mother_allele_pair) *
                              self._offspring_share(j, k))

                    # expected genotype of offspring of i and k
                    tmp_mother_allele_pair = self._genotype_to_allele_pair(k)
                    tmp_ik = (self._inheritance_prob(tmp_child_allele_pair,
                                                     tmp_father_allele_pair,
                                                     tmp_mother_allele_pair) *
                              self._offspring_share(k, j))

                    tmp_term = tmp_family_unit * (tmp_ij + tmp_ik)

                    terms.append(tmp_term)

        recurrence_relation = sum(terms)

        return recurrence_relation
