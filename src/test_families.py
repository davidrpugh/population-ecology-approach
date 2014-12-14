"""
Testing suite for the families.py module.

@author : David R. Pugh
@date : 2014-11-08

"""
import unittest

import numpy as np

import families
import simulators

import pugh_schaffer_seabright
import wright_bergstrom


class FamilyCase(unittest.TestCase):

    def setUp(self):
        """Set up code for test fixtures."""
        params = {'c': 1.0, 'e_G': 0.5, 'e_g': 0.5, 'd_A': 0.5, 'd_a': 0.5,
                  'PiaA': 7.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}
        self.family = families.Family(params=params,
                                      SGA=pugh_schaffer_seabright.SGA,
                                      Sga=pugh_schaffer_seabright.Sga)

    def test_not_implemented_methods(self):
        """Test that certain methods are not implemented."""
        with self.assertRaises(NotImplementedError):
            self.family.configurations

        with self.assertRaises(NotImplementedError):
            genotype = np.random.randint(0, 4)
            self.family._recurrence_relation_girls(genotype)

        with self.assertRaises(NotImplementedError):
            genotype = np.random.randint(0, 4)
            self.family._recurrence_relation_men(genotype)

        with self.assertRaises(NotImplementedError):
            genotypes = np.random.randint(0, 4, 3)
            self.family._family_unit(*genotypes)

    def test_validate_female_genotypes(self):
        """Testing validation of female_genotype attribute."""
        # ...each element of the tuple must be in 0,1,2,3
        invalid_genotype = (0, 4)
        with self.assertRaises(AttributeError):
            self.family.female_genotypes = invalid_genotype

        # test valid female_genotypes
        valid_female_genotypes = tuple(np.random.randint(0, 4, 2))
        self.family.female_genotypes = valid_female_genotypes
        self.assertEquals(valid_female_genotypes,
                          self.family.female_genotypes)

    def test_validate_male_genotypes(self):
        """Testing validation of male_genotype attribute."""
        # genotype must have type int...
        invalid_genotype = 1.0
        with self.assertRaises(AttributeError):
            self.family.male_genotype = invalid_genotype

        # ...and be in 0,1,2,3
        invalid_genotype = 4
        with self.assertRaises(AttributeError):
            self.family.male_genotype = invalid_genotype

        # test valid male_genotype
        expected_genotype = np.random.randint(0, 4)
        self.family.male_genotype = expected_genotype
        actual_genotype = self.family.male_genotype
        self.assertEquals(expected_genotype, actual_genotype)

    def test_validate_matching_probabilities(self):
        """Testing validation of the SGA and Sga attributes."""

        def invalid_SGA(fa, fA, e):
            """Matching probabilities should be sympy.Basic expressions."""
            return e + (1 - e) * fA / (fA + fa)

        with self.assertRaises(AttributeError):
            self.family.SGA = invalid_SGA

        def invalid_Sga(fa, fA, e):
            """Matching probabilities should be sympy.Basic expressions."""
            return e + (1 - e) * fa / (fA + fa)

        with self.assertRaises(AttributeError):
            self.family.Sga = invalid_Sga

    def test_validate_params(self):
        """Testing validation of the params attribute."""
        # parameters must be a dict
        invalid_params = (5.0, 1.0, 2.0, 3.0, 5.0, 9.0)

        with self.assertRaises(AttributeError):
            self.family.params = invalid_params

        # parameters must define all required params
        invalid_params = {'e': 1.0, 'PiaA': 2.0, 'PiAA': 3.0, 'Piaa': 5.0,
                          'PiAa': 9.0}

        with self.assertRaises(AttributeError):
            self.family.params = invalid_params

        # parameters fail prisoner's dilemma
        invalid_params = {'c': 5.0, 'e': 1.0, 'PiaA': 2.0, 'PiAA': 3.0,
                          'Piaa': 5.0, 'PiAa': 9.0}

        with self.assertRaises(AttributeError):
            self.family.params = invalid_params


class PughSchafferSeabrightCase(unittest.TestCase):

    def setUp(self):
        """Set up code for test fixtures."""
        params = {'c': 1.0, 'e_G': 0.5, 'e_g': 0.5, 'd_A': 0.5, 'd_a': 0.5,
                  'PiaA': 7.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}
        self.family = families.OneMaleTwoFemales(params=params,
                                                 SGA=pugh_schaffer_seabright.SGA,
                                                 Sga=pugh_schaffer_seabright.Sga)

    def test_numeric_jacobian_shape(self):
        """Validate the shape of the numeric Jacobian matrix."""
        X = np.repeat(0.25, 8)
        tmp_jacobian = self.family._numeric_jacobian(X[:4], X[4:],
                                                     *self.family.params.values())
        actual_shape = tmp_jacobian.shape
        expected_shape = (8, 8)
        self.assertEquals(actual_shape, expected_shape)

    def test_symbolic_jacobian_shape(self):
        """Validate the shape of the symbolic Jacobian matrix."""
        expected_shape = (8, 8)
        actual_shape = self.family._symbolic_jacobian.shape
        self.assertEquals(actual_shape, expected_shape)

    def test_selfish_equilibrium_female_children(self):
        """Testing number of female children in a selfish equilibrium."""
        # need ipd condition to hold for this to pass
        ipd_params = {'c': 3.5, 'e_G': 0.5, 'e_g': 0.5, 'd_A': 0.5, 'd_a': 0.5,
                      'PiaA': 7.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}
        self.family.params = ipd_params

        # simulate the trajectory of the model
        simulation = simulators.Simulator(self.family)
        simulation.initial_condition = 0.5
        sim = simulation.simulate(rtol=1e-12)

        # equilibrium number of female children is propto payoff
        selfish_females = sim['Female Offspring Genotypes'][[1, 3]]
        actual_number_females = selfish_females.sum(axis=1).iloc[-1]
        expected_number_females = self.family.params['c'] * self.family.params['Piaa']

        self.assertAlmostEqual(actual_number_females, expected_number_females)

    def test_altruistic_equilibrium_female_children(self):
        """Testing number of female children in an altruistic equilibrium."""
        # need ipd condition to hold for this to pass
        ipd_params = {'c': 1.5, 'e_G': 0.5, 'e_g': 0.5, 'd_A': 0.75, 'd_a': 0.5,
                      'PiaA': 7.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}
        self.family.params = ipd_params

        # simulate the trajectory of the model
        simulation = simulators.Simulator(self.family)
        simulation.initial_condition = 0.5
        sim = simulation.simulate(rtol=1e-12)

        # equilibrium number of female children is propto payoff
        altruistic_females = sim['Female Offspring Genotypes'][[0, 2]]
        actual_number_females = altruistic_females.sum(axis=1).iloc[-1]
        expected_number_females = self.family.params['c'] * self.family.params['PiAA']

        self.assertAlmostEqual(actual_number_females, expected_number_females)


class WrightBergstromCase(unittest.TestCase):

    def numeric_SGA(self, girls, e):
        altruistic_girls = girls[0] + girls[2]
        return e + (1 - e) * (altruistic_girls / girls.sum())

    def numeric_SGa(self, girls, e):
        return 1 - self.SGA(girls, e)

    def numeric_Sga(self, girls, e):
        selfish_girls = girls[1] + girls[3]
        return e + (1 - e) * (selfish_girls / girls.sum())

    def numeric_SgA(self, girls, e):
        return 1 - self.Sga(girls, e)

    def setUp(self):
        """Set up code for Wright-Bergstrom test case."""
        # females send precise signals, but males screen almost randomly
        eps = 0.5
        params = {'c': 5.0, 'e': eps,
                  'PiaA': 9.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}

        self.family = families.OneMaleTwoFemales(params=params,
                                                 SGA=wright_bergstrom.SGA,
                                                 Sga=wright_bergstrom.Sga)

    def test_perfect_signaling(self):
        """Testing the computation of family size with perfect signaling."""
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
                        SGA = self.numeric_SGA(girls, 1.0)
                        altruistic_girls = girls[0] + girls[2]
                        share_girls_1 = girls[j] / altruistic_girls
                        share_girls_2 = girls[k] / altruistic_girls
                        genotype_match_probs = (SGA * share_girls_1 *
                                                SGA * share_girls_2)

                    elif (i in [2, 3]) and (j in [1, 3]) and (k in [1, 3]):
                        # wright-bergstrom Sga (with e = 1)
                        Sga = self.numeric_Sga(girls, 1.0)
                        selfish_girls = girls[1] + girls[3]
                        share_girls_1 = girls[j] / selfish_girls
                        share_girls_2 = girls[k] / selfish_girls
                        genotype_match_probs = (Sga * share_girls_1 *
                                                Sga * share_girls_2)
                    else:
                        genotype_match_probs = np.zeros(1)

                    expected_size = men[i] * genotype_match_probs
                    actual_size = self.family.compute_size(tmp_X)
                    np.testing.assert_almost_equal(expected_size, actual_size)
