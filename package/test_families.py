import nose

import numpy as np
import sympy as sym

import families

# number of female children of particular genotype
girls = sym.DeferredVector('f')

# create an instance of the Family class
valid_female_genotypes = tuple(np.random.randint(0, 4, 2))
valid_male_genotype = np.random.randint(0, 4)
valid_family = families.Family()
valid_family.female_genotypes = valid_female_genotypes
valid_family.male_genotype = valid_male_genotype


def test_validate_male_genotypes():
    """Testing validation of male_genotype attribute."""
    # genotype must have type int...
    invalid_genotype = 1.0
    with nose.tools.assert_raises(AttributeError):
        tmp_family = families.Family()
        tmp_family.male_genotype = invalid_genotype

    # ...and be in 0,1,2,3
    invalid_genotype = 4
    with nose.tools.assert_raises(AttributeError):
        tmp_family = families.Family()
        tmp_family.male_genotype = invalid_genotype

    # test valid male_genotype
    nose.tools.assert_equals(valid_male_genotype, valid_family.male_genotype)


def test_validate_female_genotypes():
    """Testing validation of female_genotype attribute."""
    # ...each element of the tuple must be in 0,1,2,3
    invalid_genotype = (0, 4)
    with nose.tools.assert_raises(AttributeError):
        tmp_family = families.Family()
        tmp_family.female_genotypes = invalid_genotype

    # test valid female_genotypes
    nose.tools.assert_equals(valid_female_genotypes,
                             valid_family.female_genotypes)


def test_family_unit():
    """Testing family_unit method."""
    with nose.tools.assert_raises(NotImplementedError):
        valid_family.family_unit(*np.random.randint(0, 4, 3))


def test_share_girls_with_common_allele():
    """Testing the share of girls with a common allele."""
    # expected share of girls sharing a common allele with some genotype...
    genotype = np.random.randint(0, 4)
    if genotype in [0, 2]:
        expected_share = girls[genotype] / (girls[0] + girls[2])
    else:
        expected_share = girls[genotype] / (girls[1] + girls[3])

    # ...actual share from code
    actual_share = valid_family._share_girls_with_common_allele(genotype)

    nose.tools.assert_equals(expected_share, actual_share)
