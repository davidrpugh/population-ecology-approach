import nose

import numpy as np

import families


def test_validate_male_genotypes():
    """Testing validation of male_genotype attribute."""
    # genotype must have type int...
    invalid_genotype = 1.0
    with nose.tools.assert_raises(AttributeError):
        family = families.Family()
        family.male_genotype = invalid_genotype

    # ...and be in 0,1,2,3
    invalid_genotype = 4
    with nose.tools.assert_raises(AttributeError):
        family = families.Family()
        family.male_genotype = invalid_genotype

    # test valid male_genotype
    valid_genotype = np.random.randint(0, 4)
    family = families.Family()
    family.male_genotype = valid_genotype
    nose.tools.assert_equals(valid_genotype, family.male_genotype)


def test_validate_female_genotypes():
    """Testing validation of female_genotype attribute."""
    # ...each element of the tuple must be in 0,1,2,3
    invalid_genotype = (0, 4)
    with nose.tools.assert_raises(AttributeError):
        family = families.Family()
        family.female_genotypes = invalid_genotype

    # test valid female_genotypes
    valid_genotypes = tuple(np.random.randint(0, 4, 2))
    family = families.Family()
    family.female_genotypes = valid_genotypes
    nose.tools.assert_equals(valid_genotypes, family.female_genotypes)
