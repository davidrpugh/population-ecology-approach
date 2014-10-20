import unittest

import numpy as np

import families
import simulator

import model


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


class OneMaleTwoFemalesCase(unittest.TestCase):

    def setUp(self):
        """Set up code for test fixtures."""
        params = {'c': 1.0, 'e_G': 0.5, 'e_g': 0.5, 'd_A': 0.5, 'd_a': 0.5,
                  'PiaA': 7.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}
        self.family = families.OneMaleTwoFemales(params=params,
                                                 SGA=model.SGA,
                                                 Sga=model.Sga)

    def test_symbolic_jacobian_shape(self):
        """Validate the shape of the Jacobian matrix."""
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
