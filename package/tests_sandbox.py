import nose

import sandbox


def test_get_inheritance_prob():
    """Test conditional probabilities."""
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
