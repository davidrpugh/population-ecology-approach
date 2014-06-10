import sympy as sp
from traits.api import Dict, Float, HasPrivateTraits, Property, Str

# population shares by phenotype
#mG = mGA + mGa
#mg = mgA + mga
#fA = fGA + fgA
#fa = fGa + fga

class Model(HasPrivateTraits):
    """Base class representing the model of Pugh-Schaefer-Seabright."""

    _equation_1 = Property

    _equation_2 = Property

    _equation_3 = Property

    _equation_4 = Property

    _equation_5 = Property

    _equation_6 = Property

    _equation_7 = Property

    _equation_8 = Property

    _female_allele_shares = Property

    _female_signaling_probs = Property

    _male_allele_shares = Property

    _male_screening_probs = Property

    _phenotype_shares = Property

    params = Dict(Str, Float)

    def _get__equation_1(self):
        raise NotImplementedError

    def _get__equation_2(self):
        raise NotImplementedError

    def _get__equation_3(self):
        raise NotImplementedError

    def _get__equation_4(self):
        raise NotImplementedError

    def _get__equation_5(self):
        raise NotImplementedError

    def _get__equation_6(self):
        raise NotImplementedError

    def _get__equation_7(self):
        raise NotImplementedError

    def _get__equation_8(self):
        raise NotImplementedError

    def _get__female_allele_shares(self):
        """Population shares of adult females carrying alleles (Gamma, Theta)."""
        return sp.var('fGA, fGa, fgA, fga')

    def _get__female_signaling_probs(self):
        """Female signaling probabilities."""
        return sp.var('dA, da') 

    def _get__male_allele_shares(self):
        """Population shares of adult males carrying alleles (Gamma, Theta)."""
        return sp.var('mGA, mGa, mgA, mga')

    def _get__male_screening_probs(self):
        """Male screening probabilities."""
        return sp.var('eA, ea') 

    def _get__phenotype_shares(self):
        """Population shares by phenotype."""
        mGA, mGa, mgA, mga = self._male_allele_shares
        fGA, fGa, fgA, fga = self._female_allele_shares

        # compute the share by phenotype
        mG = mGA + mGa
        mg = mgA + mga
        fA = fGA + fgA
        fa = fGa + fga

        return mG, mg, fA, fa

if __name__ == '__main__':
    
    params = {'dA':0.25, 'da':0.75, 'eA':0.25, 'ea':0.5, 'PiaA':6.0, 'PiAA':5.0, 
              'Piaa':4.0, 'PiAa':3.0}

    model = Model(params=params)