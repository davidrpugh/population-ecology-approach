# define the allele indicator functions
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
