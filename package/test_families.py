import unittest

import numpy as np

import families
import simulator

import model
import wright_bergstrom


class FamilyCase(unittest.TestCase):

    def setUp(self):
        """Set up code for test fixtures."""
        params = {'c': 1.0, 'e_G': 0.5, 'e_g': 0.5, 'd_A': 0.5, 'd_a': 0.5,
                  'PiaA': 7.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}
        self.family = families.Family(params=params,
                                      SGA=model.SGA,
                                      Sga=model.Sga)

    def test_not_implemented_methods(self):
        """Test that certain methods are not implemented."""
        with self.assertRaises(NotImplementedError):
            genotype = np.random.randint(0, 4)
            self.family._recurrence_relation_girls(genotype)

        with self.assertRaises(NotImplementedError):
            genotype = np.random.randint(0, 4)
            self.family._recurrence_relation_men(genotype)

        with self.assertRaises(NotImplementedError):
            genotypes = np.random.randint(0, 4, 3)
            self.family._family_unit(*genotypes)


class PughSchaeferSeabrightCase(unittest.TestCase):

    def setUp(self):
        """Set up code for test fixtures."""
        params = {'c': 1.0, 'e_G': 0.5, 'e_g': 0.5, 'd_A': 0.5, 'd_a': 0.5,
                  'PiaA': 7.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}
        self.family = families.OneMaleTwoFemales(params=params,
                                                 SGA=model.SGA,
                                                 Sga=model.Sga)

    def test_numeric_jacobian_shape(self):
        """Validate the shape of the numeric Jacobian matrix."""
        X = np.repeat(0.25, 8)
        tmp_jacobian = self.family._numeric_jacobian(X[:4], X[4:],
                                                     **self.family.params)
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
        simulation = simulator.Simulator(self.family)
        simulation.initial_condition = 0.5
        traj = simulation.simulate(rtol=1e-12)

        # equilibrium number of female children is propto payoff
        actual_females = traj[5::2, -1].sum()
        expected_females = self.family.params['c'] * self.family.params['Piaa']

        self.assertAlmostEqual(actual_females, expected_females)

    def test_altruistic_equilibrium_female_children(self):
        """Testing number of female children in an altruistic equilibrium."""
        # need ipd condition to hold for this to pass
        ipd_params = {'c': 1.5, 'e_G': 0.5, 'e_g': 0.5, 'd_A': 0.75, 'd_a': 0.5,
                      'PiaA': 7.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}
        self.family.params = ipd_params

        # simulate the trajectory of the model
        simulation = simulator.Simulator(self.family)
        simulation.initial_condition = 0.5
        traj = simulation.simulate(rtol=1e-12)

        # equilibrium number of female children is propto payoff
        actual_females = traj[4::2, -1].sum()
        expected_females = self.family.params['c'] * self.family.params['PiAA']

        self.assertAlmostEqual(actual_females, expected_females)


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
                    actual_size = self.family.compute_size(tmp_X)[0]
                    np.testing.assert_almost_equal(expected_size, actual_size)
