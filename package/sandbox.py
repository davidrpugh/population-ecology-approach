import sympy as sym

# number of female children of particular genotype
girls = sym.DeferredVector('f')

# number of male adults of particular genotype
men = sym.DeferredVector('M')

# Female signaling probabilities
dA, da = sym.var('dA, da')

# Male screening probabilities
eA, ea = sym.var('eA, ea')

# Payoff parameters (from a Prisoner's dilemma)
PiaA, PiAA, Piaa, PiAa = sym.var('PiaA, PiAA, Piaa, PiAa')

# Female population by phenotype.
altruistic_girls = girls[0] + girls[2]
selfish_girls = girls[1] + girls[3]

# Probability of man with gene gamma matching with girl carrying gene alpha.
SGA = (dA * altruistic_girls) / (dA * altruistic_girls + (1 - eA) * (1 - da) * selfish_girls)
SGa = 1 - SGA
Sga = (da * selfish_girls) / (da * selfish_girls + (1 - ea) * (1 - dA) * altruistic_girls)
SgA = 1 - Sga


def b(i):
    """Return binary representation of the integer."""
    if i == 0:
        return (0, 0)
    elif i == 1:
        return (0, 1)
    elif i == 2:
        return (1, 0)
    elif i == 3:
        return (1, 1)
    else:
        raise ValueError('Valid genotype indices are 0,1,2,3')


def iscarrier_G(i):
    """Return True if adult carries the G allele."""
    if i in [0, 1]:
        return 1
    else:
        return 0


def iscarrier_g(i):
    """Return True if adult carries the g allele."""
    return 1 - iscarrier_G(i)


def iscarrier_A(i):
    """Return True if adult carries the A allele."""
    if i in [0, 2]:
        return 1
    else:
        return 0


def iscarrier_a(i):
    """Return True if adult carries the A allele."""
    return 1 - iscarrier_A(i)


def get_individual_payoff(i, j):
    """Number of children produced by woman with genotype i when to woman with genotype j"""
    female_i_children = (iscarrier_a(i) * iscarrier_A(j) * PiaA +
                         iscarrier_A(i) * iscarrier_A(j) * PiAA +
                         iscarrier_a(i) * iscarrier_a(j) * Piaa +
                         iscarrier_A(i) * iscarrier_a(j) * PiAa)
    return female_i_children


def get_family_payoff(i, j):
    """Number of children from families where women have genotypes i and j."""
    total_children = get_individual_payoff(i, j) + get_individual_payoff(j, i)
    return total_children


def get_relative_payoff(i, j):
    """Share of children produced by woman with genotype i when to woman with genotype j"""
    share_female_i_children = get_individual_payoff(i, j) / get_family_payoff(i, j)
    return share_female_i_children


def get_matching_probability(i, j):
    """Probability that man with genotype i is matched to girl with genotype j."""
    matching_prob = (iscarrier_G(i) * iscarrier_A(j) * SGA +
                     iscarrier_G(i) * iscarrier_a(j) * SGa +
                     iscarrier_g(i) * iscarrier_A(j) * SgA +
                     iscarrier_g(i) * iscarrier_a(j) * Sga)
    girl_population_share = girls[j] / girls_with_common_allele(j)

    return matching_prob * girl_population_share


def girls_with_common_allele(j):
    """Number of girls who share common allele with genotype j."""
    count = (iscarrier_A(j) * (altruistic_girls) +
             iscarrier_a(j) * (selfish_girls))
    return count


def get_inheritance_probability(child, parent1, parent2):
    """Conditional probability of child's genotype given parents' genotypes."""
    if has_same_genotype(parent1, parent2):
        if has_same_genotype(child, parent1):
            return 1.0
        else:
            return 0.0

    elif has_common_allele(parent1, parent2):
        if has_same_genotype(child, parent1):
            return 0.5
        elif has_same_genotype(child, parent2):
            return 0.5
        else:
            return 0.0

    elif not has_common_allele(parent1, parent2):
        return 0.25

    else:
        assert False, "You should not be reading this!"


def has_common_allele(genotype1, genotype2):
    """Return True if two genotypes share a common allele."""
    for allele1, allele2 in zip(genotype1, genotype2):
        if allele1 == allele2:
            return True
        else:
            pass
    else:
        return False


def has_same_genotype(genotype1, genotype2):
    """Return True if two genotypes are a perfect match."""
    if genotype1 == genotype2:
        return True
    else:
        return False
