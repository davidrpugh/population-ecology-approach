import nose

import sandbox


def test_b():
    """Testing mapping from genotype index to allele pairs."""
    for i in range(4):
        actual_genotype = sandbox.b(i)

        if i == 0:
            expected_genotype = (0, 0)
        elif i == 1:
            expected_genotype = (0, 1)
        elif i == 2:
            expected_genotype = (1, 0)
        else:
            expected_genotype = (1, 1)

        nose.tools.assert_equals(actual_genotype, expected_genotype)

    else:
        with nose.tools.assert_raises(ValueError):
            sandbox.b(5)


def test_get_number_children():
    """
    Testing function for computing number of children produced by individual
    female.

    """
    for i in range(4):
        for j in range(4):

            # compute actual individual payoff
            actual_children = sandbox.get_number_children(i, j)

            # expected matching probability depends on genotypes
            if (i in [0, 2]) and (j in [0, 2]):
                expected_children = sandbox.fecundity_factor * sandbox.PiAA
            elif (i in [0, 2]) and (j in [1, 3]):
                expected_children = sandbox.fecundity_factor * sandbox.PiAa
            elif (i in [1, 3]) and (j in [0, 2]):
                expected_children = sandbox.fecundity_factor * sandbox.PiaA
            else:
                expected_children = sandbox.fecundity_factor * sandbox.Piaa

            # conduct the test
            nose.tools.assert_equals(actual_children, expected_children)


def test_get_total_offspring():
    """
    Testing function for computing the total number of offsping produced by a
    given family unit.

    """
    for i in range(4):
        for j in range(4):

            # compute actual family payoff
            actual_offspring = sandbox.get_total_offspring(i, j)

            # expected matching probability depends on genotypes
            if (i in [0, 2]) and (j in [0, 2]):
                expected_offspring = 2 * sandbox.fecundity_factor * sandbox.PiAA
            elif (i in [0, 2]) and (j in [1, 3]):
                expected_offspring = (sandbox.fecundity_factor * sandbox.PiAa +
                                      sandbox.fecundity_factor * sandbox.PiaA)
            elif (i in [1, 3]) and (j in [0, 2]):
                expected_offspring = (sandbox.fecundity_factor * sandbox.PiaA +
                                      sandbox.fecundity_factor * sandbox.PiAa)
            else:
                expected_offspring = 2 * sandbox.fecundity_factor * sandbox.Piaa

            # conduct the test
            nose.tools.assert_equals(actual_offspring, expected_offspring)


def test_get_inheritance_prob():
    """Testing function for computing conditional inheritance probabilities."""
    valid_genotypes = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for parent1 in valid_genotypes:
        for parent2 in valid_genotypes:
            for child in valid_genotypes:

                # compute the inheritance probability
                tmp_prob = sandbox.get_inheritance_prob(child,
                                                        parent1,
                                                        parent2)

                if sandbox.has_same_genotype(parent1, parent2):
                    if sandbox.has_same_genotype(child, parent1):
                        nose.tools.assert_almost_equals(tmp_prob, 1.0)
                    else:
                        nose.tools.assert_almost_equals(tmp_prob, 0.0)

                elif sandbox.has_common_allele(parent1, parent2):
                    if sandbox.has_same_genotype(child, parent1):
                        nose.tools.assert_almost_equals(tmp_prob, 0.5)
                    elif sandbox.has_same_genotype(child, parent2):
                        nose.tools.assert_almost_equals(tmp_prob, 0.5)
                    else:
                        nose.tools.assert_almost_equals(tmp_prob, 0.0)

                elif not sandbox.has_common_allele(parent1, parent2):
                    nose.tools.assert_almost_equals(tmp_prob, 0.25)

                else:
                    pass


def test_get_phenotype_matching_prob():
    """Testing function for computing phenotype matching probabilities."""
    for i in range(4):
        for j in range(4):

            # compute actual matching probability
            actual_matching_prob = sandbox.get_phenotype_matching_prob(i, j)

            # expected matching probability depends on genotypes
            if (i in [0, 1]) and (j in [0, 2]):
                expected_matching_prob = sandbox.SGA
            elif (i in [0, 1]) and (j in [1, 3]):
                expected_matching_prob = sandbox.SGa
            elif (i in [2, 3]) and (j in [0, 2]):
                expected_matching_prob = sandbox.SgA
            else:
                expected_matching_prob = sandbox.Sga

            # conduct the test
            nose.tools.assert_equals(actual_matching_prob,
                                     expected_matching_prob)


def test_girls_with_common_allele():
    """
    Testing function for computing the number girls who share common allele
    with a given genotype.

    """
    for i in range(4):

        actual_count = sandbox.girls_with_common_allele(i)

        if i in [0, 2]:
            expected_count = sandbox.altruistic_girls
        else:
            expected_count = sandbox.selfish_girls

        nose.tools.assert_equals(actual_count, expected_count)
