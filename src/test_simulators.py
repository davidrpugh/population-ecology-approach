"""
Testing suite for simulators.py

Notes
-----
The testing functions for the plotting methods only test return types as the
notebooks and example scripts include many functional tests of the plotting
methods.

@author : David R. Pugh
@date : 2014-11-08

"""
import nose

import families
import simulators
import wright_bergstrom

# create an instance of the Family class
params = {'c': 5.0, 'e': 0.5, 'PiaA': 7.0, 'PiAA': 5.0, 'Piaa': 3.0,
          'PiAa': 2.0}
family = families.OneMaleTwoFemales(params,
                                    SGA=wright_bergstrom.SGA,
                                    Sga=wright_bergstrom.Sga)
simulator = simulators.Simulator(family)


def test_simulate():
    """test the simulate method of the Simulator class."""
    # one of either rtol or T must be specified!
    simulator.initial_condition = 0.5
    with nose.tools.assert_raises(ValueError):
        simulator.simulate()


def test_plot_isolated_subpopulations_simulation():
    """Test return type for isolated subpopulations simulation plot."""
    valid_kinds = ['genotypes', 'alpha_allele', 'gamma_allele']

    # tests for male subplots
    for male_kind in valid_kinds:
        out = simulators.plot_isolated_subpopulations_simulation(simulator,
                                                                 mGA0=0.5,
                                                                 T=250,
                                                                 males=male_kind,
                                                                 females='genotypes',
                                                                 **params)
        nose.tools.assert_is_instance(out, list)

    with nose.tools.assert_raises(ValueError):
        simulators.plot_isolated_subpopulations_simulation(simulator,
                                                           mGA0=0.5,
                                                           rtol=1e-12,
                                                           males='invalid_kind',
                                                           females='genotypes',
                                                           share=True,
                                                           **params)

    # tests for female subplots
    for female_kind in valid_kinds:
        for tmp_share in [True, False]:
            out = simulators.plot_isolated_subpopulations_simulation(simulator,
                                                                     mGA0=0.5,
                                                                     rtol=1e-12,
                                                                     males='genotypes',
                                                                     females=female_kind,
                                                                     share=tmp_share,
                                                                     **params)
            nose.tools.assert_is_instance(out, list)

    with nose.tools.assert_raises(ValueError):
        simulators.plot_isolated_subpopulations_simulation(simulator,
                                                           mGA0=0.5,
                                                           rtol=1e-12,
                                                           males='gamma_allele',
                                                           females='invalid_kind',
                                                           share=True,
                                                           **params)


def test_plot_selection_pressure():
    """Test return type for selection pressure plot."""
    out = simulators.plot_selection_pressure(simulator, mGA0=0.5, rtol=1e-12,
                                             **params)
    nose.tools.assert_is_instance(out, list)
