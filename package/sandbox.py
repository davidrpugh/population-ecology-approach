import sympy as sym

# number of female children of particular genotype
girls = sym.DeferredVector('f')

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


def get_payoff(i, j):
    """Payoff to female with genotype i, when matched to female with genotype j."""
    payoff = (iscarrier_a(i) * iscarrier_A(j) * PiaA +
              iscarrier_A(i) * iscarrier_A(j) * PiAA +
              iscarrier_a(i) * iscarrier_a(j) * Piaa +
              iscarrier_A(i) * iscarrier_a(j) * PiAa)
    return payoff


def get_matching_probability(i, j):
    """Probability that male with genotype i is matched to female with genotype j."""
    matching_prob = (iscarrier_G(i) * iscarrier_A(j) * SGA +
                     iscarrier_G(i) * iscarrier_a(j) * SGa +
                     iscarrier_g(i) * iscarrier_A(j) * SgA +
                     iscarrier_g(i) * iscarrier_a(j) * Sga)
    population_share = girls[j] / girls_with_same_allele(j)

    return matching_prob * population_share


def girls_with_same_allele(j):
    """Number of girls who share same alpha with genotype j."""
    return (iscarrier_A(j) * (altruistic_girls) + iscarrier_a(j) * (selfish_girls))
