from traits.api import Float, HasPrivateTraits, Str

class Model(HasPrivateTraits):
    """Base class representing the model of Pugh-Schaefer-Seabright."""

    params = Dict(Str, Float)