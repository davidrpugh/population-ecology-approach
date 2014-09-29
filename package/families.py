class FamilyUnit(object):
    """Class for a family unit in the 1M2F model."""

    def __init__(self, male_genotype, female_genotype_1, female_genotype_2):
        """
        Creates an instance of the FamilyUnit class.

        """
        self.i = male_genotype
        self.j = female_genotype_1
        self.k = female_genotype_2