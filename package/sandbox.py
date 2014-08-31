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

# Conditional phenotype matching probabilities
true_altruistic_girls = dA * altruistic_girls
false_altruistic_girls = (1 - da) * selfish_girls
mistaken_for_altruistic_girls = (1 - eA) * false_altruistic_girls
altruist_adoption_pool = true_altruistic_girls + mistaken_for_altruistic_girls

true_selfish_girls = da * selfish_girls
false_selfish_girls = (1 - dA) * altruistic_girls
mistaken_for_selfish_girls = (1 - ea) * false_selfish_girls
selfish_adpotion_pool = true_selfish_girls + mistaken_for_selfish_girls

SGA = true_altruistic_girls / altruist_adoption_pool
SGa = 1 - SGA
Sga = true_selfish_girls / selfish_adpotion_pool
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
    """Payoff produced by woman with genotype i when to woman with genotype j"""
    payoff_i = (iscarrier_a(i) * iscarrier_A(j) * PiaA +
                iscarrier_A(i) * iscarrier_A(j) * PiAA +
                iscarrier_a(i) * iscarrier_a(j) * Piaa +
                iscarrier_A(i) * iscarrier_a(j) * PiAa)
    return payoff_i


def get_family_payoff(i, j):
    """Payoff from families where women have genotypes i and j."""
    total_payoff = get_individual_payoff(i, j) + get_individual_payoff(j, i)
    return total_payoff


def get_payoff_share(i, j):
    """Share of family payoff produced by woman with genotype i when matched to woman with genotype j"""
    payoff_share = get_individual_payoff(i, j) / get_family_payoff(i, j)
    return payoff_share


def get_genotype_matching_prob(i, j):
    """Conditional probability that man with genotype i is matched to girl with genotype j."""
    phenotype_matching_prob = get_phenotype_matching_prob(i, j)
    girl_population_share = girls[j] / girls_with_common_allele(j)
    return phenotype_matching_prob * girl_population_share


def get_phenotype_matching_prob(i, j):
    """Conditional probability that man with phenotype i is matched to girl with phenotype j."""
    phenotype_matching_prob = (iscarrier_G(i) * iscarrier_A(j) * SGA +
                               iscarrier_G(i) * iscarrier_a(j) * SGa +
                               iscarrier_g(i) * iscarrier_A(j) * SgA +
                               iscarrier_g(i) * iscarrier_a(j) * Sga)
    return phenotype_matching_prob


def girls_with_common_allele(j):
    """Number of girls who share common allele with genotype j."""
    count = (iscarrier_A(j) * (altruistic_girls) +
             iscarrier_a(j) * (selfish_girls))
    return count


def get_inheritance_prob(child, parent1, parent2):
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


def get_family_unit(i, j, k):
    """Family unit comprised of male with genoytpe i, and females with genotypes j and k."""
    U_ijk = (men[i] * get_genotype_matching_prob(i, j) *
             get_genotype_matching_prob(i, k))
    return U_ijk


def get_female_recurrence_relation(x):
    """Return recurrence relation for female children with genotype x."""
    terms = []
    for i in range(4):
        for j in range(4):
            for k in range(4):

                tmp_family_unit = get_family_unit(i, j, k)
                tmp_daughters_ij = get_inheritance_prob(b(x), b(i), b(j)) * get_individual_payoff(j, k)
                tmp_daughters_ik = get_inheritance_prob(b(x), b(i), b(k)) * get_individual_payoff(k, j)
                tmp_term = tmp_family_unit * (tmp_daughters_ij + tmp_daughters_ik)

                terms.append(tmp_term)

    return 0.5 * sum(terms)


def get_male_recurrence_relation(x):
    """Return recurrence relation for male adults with genotype x."""
    terms = []
    for i in range(4):
        for j in range(4):
            for k in range(4):

                tmp_family_unit = get_family_unit(i, j, k)
                tmp_daughters_ij = get_inheritance_prob(b(x), b(i), b(j)) * get_payoff_share(j, k)
                tmp_daughters_ik = get_inheritance_prob(b(x), b(i), b(k)) * get_payoff_share(k, j)
                tmp_term = tmp_family_unit * (tmp_daughters_ij + tmp_daughters_ik)

                terms.append(tmp_term)

    return sum(terms)
