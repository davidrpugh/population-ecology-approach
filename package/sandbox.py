import sympy as sym

# Payoff parameters (from a Prisoner's dilemma)
PiaA, PiAA, Piaa, PiAa = matching_payoffs = sym.var('PiaA, PiAA, Piaa, PiAa')


def iscarrier_G(i):
    """Return True if adult carries the G allele."""
    if i in [0, 1]:
        return True
    else:
        return False


def iscarrier_g(i):
    """Return True if adult carries the g allele."""
    return 1 - iscarrier_G(i)


def iscarrier_A(i):
    """Return True if adult carries the A allele."""
    if i in [0, 2]:
        return True
    else:
        return False


def iscarrier_a(i):
    """Return True if adult carries the A allele."""
    return 1 - iscarrier_A(i)


def get_payoff(i, j):
    payoff = (iscarrier_a(i) * iscarrier_A(j) * PiaA +
              iscarrier_A(i) * iscarrier_A(j) * PiAA +
              iscarrier_a(i) * iscarrier_a(j) * Piaa +
              iscarrier_A(i) * iscarrier_a(j) * PiAa)
    return payoff