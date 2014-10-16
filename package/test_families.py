import nose
import unittest

import numpy as np
import sympy as sym

import families

# number of female children of particular genotype
girls = sym.DeferredVector('f')

# number of male adults of particular genotype
men = sym.DeferredVector('M')

# Male screening probabilities
e = sym.var('e')


class BaseCase(unittest.TestCase):

    def setUp(self):
        """Set up code for test fixtures."""
        # conditional phenotype matching probabilities
        SGA = e
        Sga = e

        # specify some valid parameters
        params = {'c': 5.0, 'e': 1.0, 'PiaA': 9.0, 'PiAA': 5.0, 'Piaa': 3.0,
                  'PiAa': 2.0}

        # create an instance of the Family class
        self.family = families.Family(params=params,
                                      SGA=SGA,
                                      Sga=Sga)

    def test_family_unit(self):
        """Testing family_unit method for base class."""
        with nose.tools.assert_raises(NotImplementedError):
            self.family._family_unit(*np.random.randint(0, 4, 3))

    def test_share_girls_with_common_allele(self):
        """Testing the share of girls with a common allele."""
        # expected share of girls sharing a common allele with some genotype...
        genotype = np.random.randint(0, 4)
        if genotype in [0, 2]:
            expected_share = girls[genotype] / (girls[0] + girls[2])
        else:
            expected_share = girls[genotype] / (girls[1] + girls[3])

        # ...actual share from code
        actual_share = self.family._share_girls_with_common_allele(genotype)

        nose.tools.assert_equals(expected_share, actual_share)

    def test_validate_female_genotypes(self):
        """Testing validation of female_genotype attribute."""
        # ...each element of the tuple must be in 0,1,2,3
        invalid_genotype = (0, 4)
        with nose.tools.assert_raises(AttributeError):
            self.family.female_genotypes = invalid_genotype

        # test valid female_genotypes
        valid_female_genotypes = tuple(np.random.randint(0, 4, 2))
        self.family.female_genotypes = valid_female_genotypes
        nose.tools.assert_equals(valid_female_genotypes,
                                 self.family.female_genotypes)

    def test_validate_male_genotypes(self):
        """Testing validation of male_genotype attribute."""
        # genotype must have type int...
        invalid_genotype = 1.0
        with nose.tools.assert_raises(AttributeError):
            self.family.male_genotype = invalid_genotype

        # ...and be in 0,1,2,3
        invalid_genotype = 4
        with nose.tools.assert_raises(AttributeError):
            self.family.male_genotype = invalid_genotype

        # test valid male_genotype
        valid_male_genotype = np.random.randint(0, 4)
        self.family.male_genotype = valid_male_genotype
        nose.tools.assert_equals(valid_male_genotype, self.family.male_genotype)

    def test_validate_matching_probabilities(self):
        """Testing validation of the SGA and Sga attributes."""

        def invalid_SGA(fa, fA, e):
            """Matching probabilities should be sympy.Basic expressions."""
            return e + (1 - e) * fA / (fA + fa)

        with nose.tools.assert_raises(AttributeError):
            self.family.SGA = invalid_SGA

        def invalid_Sga(fa, fA, e):
            """Matching probabilities should be sympy.Basic expressions."""
            return e + (1 - e) * fa / (fA + fa)

        with nose.tools.assert_raises(AttributeError):
            self.family.Sga = invalid_Sga

    def test_validate_params(self):
        """Testing validation of the params attribute."""
        pass


class WrightBergstromCase(unittest.TestCase):

    def setUp(self):
        """Set up code for Wright-Bergstrom test case."""
        # Female population by phenotype.
        altruistic_girls = girls[0] + girls[2]
        selfish_girls = girls[1] + girls[3]

        # conditional phenotype matching probabilities (a la Wright/Bergstrom)
        SGA = e + (1 - e) * altruistic_girls / (altruistic_girls + selfish_girls)
        Sga = e + (1 - e) * selfish_girls / (altruistic_girls + selfish_girls)

        # females send precise signals, but males screen almost randomly
        eps = 0.5
        params = {'c': 5.0, 'e': eps,
                  'PiaA': 9.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}

        self.family = families.OneMaleTwoFemales(params=params,
                                                 SGA=SGA,
                                                 Sga=Sga)

    def test_compute_size(self):
        """Testing the computation of family size."""

        # test case for perfect signaling
        self.family.params['e'] = 1.0

        for i in range(4):
            for j in range(4):
                for k in range(4):

                    # specify the genotypes
                    self.family.male_genotype = i
                    self.family.female_genotypes = j, k

                    # pick a random vector for endogenous variables
                    men = np.random.dirichlet((1, 1, 1, 1))
                    girls = np.array([5.0, 4.0, 3.2, 2.5])
                    tmp_X = np.hstack((men, girls))

                    if (i in [0, 1]) and (j in [0, 2]) and (k in [0, 2]):
                        # wright-bergstrom SGA (with e = 1)
                        SGA = 1.0
                        altruistic_girls = girls[0] + girls[2]
                        share_girls_1 = girls[j] / altruistic_girls
                        share_girls_2 = girls[k] / altruistic_girls
                        genotype_match_probs = (SGA * share_girls_1 *
                                                SGA * share_girls_2)

                    elif (i in [2, 3]) and (j in [1, 3]) and (k in [1, 3]):
                        # wright-bergstrom Sga (with e = 1)
                        Sga = 1.0
                        selfish_girls = girls[1] + girls[3]
                        share_girls_1 = girls[j] / selfish_girls
                        share_girls_2 = girls[k] / selfish_girls
                        genotype_match_probs = (Sga * share_girls_1 *
                                                Sga * share_girls_2)
                    else:
                        genotype_match_probs = np.zeros(1)

                    expected_size = men[i] * genotype_match_probs
                    actual_size = self.family.compute_size(tmp_X)[0]
                    np.testing.assert_almost_equal(expected_size, actual_size)
