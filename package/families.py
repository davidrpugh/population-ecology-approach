class FamilyUnit(object):
    """Class representing a family unit in the 1M2F model."""

    def __init__(self, male_genotype, female_genotype_1, female_genotype_2,
                 GA_matching_probability, ga_matching_probability):
        """
        Creates an instance of the FamilyUnit class.

        """
        self.i = male_genotype
        self.j = female_genotype_1
        self.k = female_genotype_2

        self.SGA = GA_matching_probability
        self.Sgs = ga_matching_probability