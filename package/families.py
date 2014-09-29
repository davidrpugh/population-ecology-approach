import sympy as sym


class FamilyUnit(object):
    """Class representing a family unit in the 1M2F model."""

    def __init__(self, GA_matching_probability, ga_matching_probability,
                 female_genotype_1, female_genotype_2, male_genotype, params):
        """
        Creates an instance of the FamilyUnit class.

        """
        self.i = self._validate_genotype(male_genotype)
        self.j = self._validate_genotype(female_genotype_1)
        self.k = self._validate_genotype(female_genotype_2)

        self.SGA = self._validate_matching_prob(GA_matching_probability)
        self.Sgs = self._validate_matching_prob(ga_matching_probability)

        self.params = self._validate_params(params)

    @staticmethod
    def _validate_genotype(genotype):
        """Validate the genotype."""
        valid_genotypes = range(4)
        if not isinstance(genotype, int):
            mesg = "FamilyUnit genotypes must be an int, not a {}."
            raise AttributeError(mesg.format(genotype.__class__))
        if genotype not in valid_genotypes:
            mesg = "FamilyUnit genotypes must be in {}."
            raise AttributeError(mesg.format(valid_genotypes))
        else:
            return genotype

    @staticmethod
    def _validate_matching_prob(expr):
        """Validate the phenotype matching probability."""
        if not isinstance(expr, sym.Basic):
            mesg = ("FamilyUnit matching probabilities must be an instance of "
                    "sym.Basic, not a {}.")
            raise AttributeError(mesg.format(expr.__class__))
        else:
            return expr

    @staticmethod
    def _validate_params(params):
        """Validate the model parameters."""
        if not isinstance(params, dict):
            mesg = "FamilyUnit params must be a dict, not a {}."
            raise AttributeError(mesg.format(params.__class__))
        else:
            return params