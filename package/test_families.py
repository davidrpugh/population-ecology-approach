import nose

import numpy as np
import sympy as sym

import families

# number of female children of particular genotype
girls = sym.DeferredVector('f')

# number of male adults of particular genotype
men = sym.DeferredVector('M')

# Male screening probabilities
e = sym.var('e')

# Female population by phenotype.
altruistic_girls = girls[0] + girls[2]
selfish_girls = girls[1] + girls[3]

# conditional phenotype matching probabilities (a la Wright/Bergstrom)
SGA = e + (1 - e) * altruistic_girls / (altruistic_girls + selfish_girls)
SGa = 1 - SGA
Sga = e + (1 - e) * selfish_girls / (altruistic_girls + selfish_girls)
SgA = 1 - Sga

# females send precise signals, but males screen almost randomly
eps = 0.5
params = {'c': 5.0, 'e': eps,
          'PiaA': 9.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}

# create an instance of the Family class
family = families.Family(params=params,
                         SGA=SGA,
                         Sga=Sga)

valid_female_genotypes = tuple(np.random.randint(0, 4, 2))
valid_male_genotype = np.random.randint(0, 4)

family.female_genotypes = valid_female_genotypes
family.male_genotype = valid_male_genotype


def test_validate_male_genotypes():
    """Testing validation of male_genotype attribute."""
    # genotype must have type int...
    invalid_genotype = 1.0
    with nose.tools.assert_raises(AttributeError):
        family.male_genotype = invalid_genotype

    # ...and be in 0,1,2,3
    invalid_genotype = 4
    with nose.tools.assert_raises(AttributeError):
        family.male_genotype = invalid_genotype

    # test valid male_genotype
    nose.tools.assert_equals(valid_male_genotype, family.male_genotype)


def test_validate_female_genotypes():
    """Testing validation of female_genotype attribute."""
    # ...each element of the tuple must be in 0,1,2,3
    invalid_genotype = (0, 4)
    with nose.tools.assert_raises(AttributeError):
        family.female_genotypes = invalid_genotype

    # test valid female_genotypes
    nose.tools.assert_equals(valid_female_genotypes,
                             family.female_genotypes)


def test_validate_matching_probabilities():
    """Testing validation of the SGA and Sga attributes."""

    def invalid_SGA(fa, fA, e):
        """Matching probabilities should be sympy.Basic expressions."""
        return e + (1 - e) * fA / (fA + fa)

    with nose.tools.assert_raises(AttributeError):
        family.SGA = invalid_SGA

    def invalid_Sga(fa, fA, e):
        """Matching probabilities should be sympy.Basic expressions."""
        return e + (1 - e) * fa / (fA + fa)

    with nose.tools.assert_raises(AttributeError):
        family.Sga = invalid_Sga


def test_family_unit():
    """Testing family_unit method."""
    with nose.tools.assert_raises(NotImplementedError):
        family._family_unit(*np.random.randint(0, 4, 3))


def test_share_girls_with_common_allele():
    """Testing the share of girls with a common allele."""
    # expected share of girls sharing a common allele with some genotype...
    genotype = np.random.randint(0, 4)
    if genotype in [0, 2]:
        expected_share = girls[genotype] / (girls[0] + girls[2])
    else:
        expected_share = girls[genotype] / (girls[1] + girls[3])

    # ...actual share from code
    actual_share = family._share_girls_with_common_allele(genotype)

    nose.tools.assert_equals(expected_share, actual_share)


def test_compute_size():
    """Testing the computation of family size."""
    params = {'c': 5.0, 'e': 1.0,
              'PiaA': 9.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}

    family = families.OneMaleTwoFemales(params=params,
                                        SGA=SGA,
                                        Sga=Sga)

    family.male_genotype = 0
    family.female_genotypes = 0, 0

    men = np.repeat(0.25, 4)
    girls = np.array([5.0, 4.0, 3.2, 2.5])
    initial_condition = np.hstack((men, girls))

    actual_size = family.compute_size(initial_condition)
    expected_size = np.array([men[0] * (girls[0] / (girls[0] + girls[2]))**2])

    np.testing.assert_almost_equal(expected_size, actual_size)
