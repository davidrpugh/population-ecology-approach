from traits.api import Dict, Float, HasPrivateTraits, Str

class Model(HasPrivateTraits):
    """Base class representing the model of Pugh-Schaefer-Seabright."""

    params = Dict(Str, Float)

    _equation_1 = Property

    _equation_2 = Property

    _equation_3 = Property

    _equation_4 = Property

    _equation_5 = Property

    _equation_6 = Property

    _equation_7 = Property

    _equation_8 = Property

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



if __name__ == '__main__':
    
    params = {'dA':0.25, 'da':0.75, 'eA':0.25, 'ea':0.5, 'PiaA':6.0, 'PiAA':5.0, 
              'Piaa':4.0, 'PiAa':3.0}

    model = Model(params=params)