from __future__ import division

import numpy as np
import sympy as sp

from traits.api import (cached_property, Dict, Float, HasPrivateTraits, 
                        Property, Str)

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

    _matching_probs = Property

    _number_female_children = Property

    _numeric_simulation_jacobian = Property(depends_on='_symbolic_simulation_jacobian')

    _numeric_simulation_system = Property(depends_on='_symbolic_simulation_system')

    _payoffs = Property

    _phenotype_shares = Property

    _symbolic_simulation_jacobian = Property(depends_on='_symbolic_simulation_system')

    _symbolic_simulation_system = Property

    _symbolic_steady_state_jacobian = Property(depends_on='_symbolic_steady_state_system')

    _symbolic_steady_state_system = Property

    params = Dict(Str, Float)

    def _get__equation_1(self):
        """Recurrence relation for mGA."""
        # extract the variables
        mGA, mGa, mgA, mga = self._male_allele_shares
        fGA, fGa, fgA, fga = self._female_allele_shares
        PiaA, PiAA, Piaa, PiAa = self._payoffs
        mG, mg, fA, fa = self._phenotype_shares
        SGA, SGa, SgA, Sga = self._matching_probs

        eqn1 = (mGA * SGA**2 * ((fGA + 0.5 * fgA) / fA) + 
                mGA * SGa**2 * ((0.5 * fGa + 0.25 * fga) / fa) + 
                2 * mGA * SGA * SGa * (((fGA + 0.5 * fgA) / fA) * (PiAa / (PiAa + PiaA)) + ((0.5 * fGa + 0.25 * fga) / fa) * (PiaA / (PiAa + PiaA))) + 
                mGa * SGA**2 * ((0.5 * fGA + 0.25 * fgA) / fA) +
                2 * mGa * SGA * SGa * (((0.5 * fGA + 0.25 * fgA) / fA) * (PiAa / (PiAa + PiaA))) + 
                mgA * SgA**2 * (0.5 * fGA / fA) + 
                mgA * Sga**2 * (0.25 * fGa / fa) + 
                2 * mgA * SgA * Sga * ((0.5 * fGA / fA) * (PiAa / (PiAa + PiaA)) + (0.25 * fGa / fa) * (PiaA / (PiAa + PiaA))) + 
                mga * SgA**2 * (0.25 * fGA / fA) +
                2 * mga * SgA * Sga * ((0.25 * fGA / fA) * (PiAa / (PiAa + PiaA))))

        return eqn1

    def _get__equation_2(self):
        """Recurrence relation for mGa."""
        # extract the variables
        mGA, mGa, mgA, mga = self._male_allele_shares
        fGA, fGa, fgA, fga = self._female_allele_shares
        PiaA, PiAA, Piaa, PiAa = self._payoffs
        mG, mg, fA, fa = self._phenotype_shares
        SGA, SGa, SgA, Sga = self._matching_probs

        eqn2 = (mGA * SGa**2 * ((0.5 * fGa + 0.25 * fga) / fa) + 
                2 * mGA * SGA * SGa * (((0.5 * fGa + 0.25 * fga) / fa) * (PiaA / (PiAa + PiaA))) + 
                mGa * SGA**2 * ((0.5 * fGA + 0.25 * fgA) / fA) +
                mGa * SGa**2 * ((fGa + 0.5 * fga) / fa) + 
                2 * mGa * SGA * SGa * (((0.5 * fGA + 0.25 * fgA) / fA) * (PiAa / (PiAa + PiaA)) + ((fGa + 0.5 * fga) / fa) * (PiaA / (PiAa + PiaA))) + 
                mgA * Sga**2 * (0.25 * fGa / fa) + 
                2 * mgA * SgA * Sga * ((0.25 * fGa / fa) * (PiaA / (PiAa + PiaA))) +
                mga * SgA**2 * (0.25 * fGA / fA) + 
                mga * Sga**2 * (0.5 * fGa / fa) +
                2 * mga * SgA * Sga * ((0.25 * fGA / fA) * (PiAa / (PiAa + PiaA)) + (0.5 * fGa / fa) * (PiaA / (PiAa + PiaA))))

        return eqn2

    def _get__equation_3(self):
        """Recurrence relation for mgA.""" 
        # extract the variables
        mGA, mGa, mgA, mga = self._male_allele_shares
        fGA, fGa, fgA, fga = self._female_allele_shares
        PiaA, PiAA, Piaa, PiAa = self._payoffs
        mG, mg, fA, fa = self._phenotype_shares
        SGA, SGa, SgA, Sga = self._matching_probs

        eqn3 = (mGA * SGA**2 * (0.5 * fgA / fA) + 
                mGA * SGa**2 * (0.25 * fga / fa) + 
                2 * mGA * SGA * SGa * ((0.5 * fgA / fA) * (PiAa / (PiAa + PiaA)) + (0.25 * fga / fa) * (PiaA / (PiAa + PiaA))) +
                mGa * SGA**2 * (0.25 * fgA / fA) + 
                2 * mGa * SGA * SGa * ((0.25 * fgA / fA) * (PiAa / (PiAa + PiaA))) + 
                mgA * SgA**2 * ((0.5 * fGA + fgA) / fA) + 
                mgA * Sga**2 * ((0.25 * fGa + 0.5 * fga) / fa) + 
                2 * mgA * SgA * Sga * (((0.5 * fGA + fgA) / fA) * (PiAa / (PiAa + PiaA)) + ((0.25 * fGa + 0.5 * fga) / fa) * (PiaA / (PiAa + PiaA))) + 
                mga * SgA**2 * ((0.25 * fGA + 0.5 * fgA) / fA) + 
                2 * mga * SgA * Sga * (((0.25 * fGA + 0.5 * fgA) / fA) * (PiAa / (PiAa + PiaA))))

        return eqn3

    def _get__equation_4(self):
        """Recurrence relation for mga."""
        # extract the variables
        mGA, mGa, mgA, mga = self._male_allele_shares
        fGA, fGa, fgA, fga = self._female_allele_shares
        PiaA, PiAA, Piaa, PiAa = self._payoffs
        mG, mg, fA, fa = self._phenotype_shares
        SGA, SGa, SgA, Sga = self._matching_probs

        eqn4 = (mGA * SGa**2 * (0.25 * fga / fa) + 
                2 * mGA * SGA * SGa * ((0.25 * fga / fa) * (PiaA / (PiAa + PiaA))) + 
                mGa * SGA**2 * (0.25 * fgA / fA) + 
                mGa * SGa**2 * (0.5 * fga / fa) + 
                2 * mGa * SGA * SGa * ((0.25 * fgA / fA) * (PiAa / (PiAa + PiaA)) + (0.5 * fga / fa) * (PiaA / (PiAa + PiaA))) + 
                mgA * Sga**2 * ((0.25 * fGa + 0.5 * fga) / fa) + 
                2 * mgA * SgA * Sga * (((0.25 * fGa + 0.5 * fga) / fa) * (PiaA / (PiAa + PiaA))) + 
                mga * SgA**2 * ((0.25 * fGA + 0.5 * fgA) / fA) + 
                mga * Sga**2 * ((0.5 * fGa + fga) / fa) + 
                2 * mga * SgA * Sga * (((0.25 * fGA + 0.5 * fgA) / fA) * (PiAa / (PiAa + PiaA)) + ((0.5 * fGa + fga) / fa) * (PiaA / (PiAa + PiaA))))

        return eqn4

    def _get__equation_5(self):
        """Recurrence relation for fGA."""
        # extract the variables
        mGA, mGa, mgA, mga = self._male_allele_shares
        fGA, fGa, fgA, fga = self._female_allele_shares
        PiaA, PiAA, Piaa, PiAa = self._payoffs
        mG, mg, fA, fa = self._phenotype_shares
        SGA, SGa, SgA, Sga = self._matching_probs
        Nprime = self._number_female_children

        eqn5 = (mGA * SGA**2 * (((fGA + 0.5 * fgA) / fA) * (2 * PiAA / Nprime)) + 
                mGA * SGa**2 * (((0.5 * fGa + 0.25 * fga) / fa) * (2 * Piaa / Nprime)) + 
                2 * mGA * SGA * SGa * (((fGA + 0.5 * fgA) / fA) * (PiAa / Nprime) + ((0.5 * fGa + 0.25 * fga) / fa) * (PiaA / Nprime)) + 
                mGa * SGA**2 * (((0.5 * fGA + 0.25 * fgA) / fA) * (2 * PiAA / Nprime)) + 
                2 * mGa * SGA * SGa * (((0.5 * fGA + 0.25 * fgA) / fA) * (PiAa / Nprime)) + 
                mgA * SgA**2 * ((0.5 * fGA / fA) * (2 * PiAA / Nprime)) + 
                mgA * Sga**2 * ((0.25 * fGa / fa) * (2 * Piaa / Nprime)) +
                2 * mgA * SgA * Sga * ((0.5 * fGA / fA) * (PiAa / Nprime) + (0.25 * fGa / fa) * (PiaA / Nprime)) + 
                mga * SgA**2 * ((0.25 * fGA / fA) * (2 * PiAA / Nprime)) + 
                2 * mga * SgA * Sga * ((0.25 * fGA / fA) * (PiAa / Nprime)))

        return eqn5

    def _get__equation_6(self):
        """Recurrence relation for fGa.""" 
        # extract the variables
        mGA, mGa, mgA, mga = self._male_allele_shares
        fGA, fGa, fgA, fga = self._female_allele_shares
        PiaA, PiAA, Piaa, PiAa = self._payoffs
        mG, mg, fA, fa = self._phenotype_shares
        SGA, SGa, SgA, Sga = self._matching_probs
        Nprime = self._number_female_children

        eqn6 = (mGA * SGa**2 * ((0.5 * fGa + 0.25 * fga) / fa) * (2 * Piaa / Nprime) +
                2 * mGA * SGA * SGa * (((0.5 * fGa + 0.25 * fga) / fa) * (PiaA / Nprime)) +
                mGa * SGA**2 * ((0.5 * fGA + 0.25 * fgA) / fA) * (2 * PiAA / Nprime) +
                mGa * SGa**2 * ((fGa + 0.5 * fga) / fa) * (2 * Piaa / Nprime) +
                2 * mGa * SGA * SGa * ((0.5 * fGA + 0.25 * fgA) / fA * PiAa / Nprime + (fGa + 0.5 * fga) / fa * PiaA / Nprime) +
                mgA * Sga**2 * 0.25 * fGa / fa * (2 * Piaa / Nprime) +
                2 * mgA * SgA * Sga * (0.25 * fGa / fa) * (PiaA / Nprime)    +
                mga * SgA**2 * (0.25 * fGA / fA) * (2 * PiAA / Nprime) +
                mga * Sga**2 * (0.5 * fGa / fa) * (2 * Piaa / Nprime) +
                2 * mga * SgA * Sga * ((0.25 * fGA / fA) * (PiAa / Nprime) + (0.5 * fGa / fa) * (PiaA / Nprime)))

        return eqn6

    def _get__equation_7(self):
        """Recurrence relation for fgA."""
        # extract the variables
        mGA, mGa, mgA, mga = self._male_allele_shares
        fGA, fGa, fgA, fga = self._female_allele_shares
        PiaA, PiAA, Piaa, PiAa = self._payoffs
        mG, mg, fA, fa = self._phenotype_shares
        SGA, SGa, SgA, Sga = self._matching_probs
        Nprime = self._number_female_children

        eqn7 = (mGA * SGA**2 * ((0.5 * fgA / fA) * (2 * PiAA / Nprime)) +
                mGA * SGa**2 * ((0.25 * fga / fa) * (2 * Piaa / Nprime)) + 
                2 * mGA * SGA * SGa * ((0.5 * fgA / fA) * (PiAa / Nprime) + (0.25 * fga / fa) * (PiaA / Nprime)) +
                mGa * SGA**2 * ((0.25 * fgA / fA) * (2 * PiAA / Nprime)) + 
                2 * mGa * SGA * SGa * ((0.25 * fgA / fA) * (PiAa / Nprime)) + 
                mgA * SgA**2 * (((0.5 * fGA + fgA) / fA) * (2 * PiAA / Nprime)) + 
                mgA * Sga**2 * (((0.25 * fGa + 0.5 * fga) / fa) * (2 * Piaa / Nprime)) + 
                2 * mgA * SgA * Sga * (((0.5 * fGA + fgA) / fA) * (PiAa / Nprime) + ((0.25 * fGa + 0.5 * fga) / fa) * (PiaA / Nprime)) +
                mga * SgA**2 * (((0.25 * fGA + 0.5 * fgA) / fA) * (2 * PiAA / Nprime)) +
                2 * mga * SgA * Sga * (((0.25 * fGA + 0.5 * fgA) / fA) * (PiAa / Nprime)))

        return eqn7

    def _get__equation_8(self):
        """Recurrence relation for fga."""
        # extract the variables
        mGA, mGa, mgA, mga = self._male_allele_shares
        fGA, fGa, fgA, fga = self._female_allele_shares
        PiaA, PiAA, Piaa, PiAa = self._payoffs
        mG, mg, fA, fa = self._phenotype_shares
        SGA, SGa, SgA, Sga = self._matching_probs
        Nprime = self._number_female_children

        eqn8 = (mGA * SGa**2 * ((0.25 * fga / fa) * (2 * Piaa / Nprime)) + 
                2 * mGA * SGA * SGa * ((0.25 * fga / fa) * (PiaA / Nprime)) + 
                mGa * SGA**2 * ((0.25 * fgA / fA) * (2 * PiAA / Nprime)) +
                mGa * SGa**2 * ((0.5 * fga / fa) * (2 * Piaa / Nprime)) + 
                2 * mGa * SGA * SGa * ((0.25 * fgA / fA) * (PiAa / Nprime) + (0.5 * fga / fa) * (PiaA / Nprime)) +
                mgA * Sga**2 * (((0.25 * fGa + 0.5 * fga) / fa) * (2 * Piaa / Nprime)) +
                2 * mgA * SgA * Sga * (((0.25 * fGa + 0.5 * fga) / fa) * (PiaA / Nprime)) + 
                mga * SgA**2 * (((0.25 * fGA + 0.5 * fgA) / fA) * (2 * PiAA / Nprime)) +
                mga * Sga**2 * (((0.5 * fGa + fga) / fa) * (2 * Piaa / Nprime)) +
                2 * mga * SgA * Sga * (((0.25 * fGA + 0.5 * fgA) / fA) * (PiAa / Nprime) + ((0.5 * fGa + fga) / fa) * (PiaA / Nprime)))

        return eqn8

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

    def _get__matching_probs(self):
        """Probabilities that male matches with desired female."""
        mG, mg, fA, fa = self._phenotype_shares
        dA, da = self._female_signaling_probs
        eA, ea = self._male_screening_probs

        # compute the matching probabilities
        SGA = (dA * fA) / (dA * fA + (1 - eA) * (1 - da) * fa)
        SGa = 1 - SGA
        Sga = (da * fa) / (da * fa + (1 - ea) * (1 - dA) * fA)
        SgA = 1 - Sga

        return SGA, SGa, SgA, Sga 

    def _get__number_female_children(self):
        """Total female children in the next generation."""
        # extract the variables
        PiaA, PiAA, Piaa, PiAa = self._payoffs
        mG, mg, fA, fa = self._phenotype_shares
        SGA, SGa, SgA, Sga = self._matching_probs

        # compute the number of female children
        Nprime = ((mG * SGA**2 + mg * SgA**2) * 2 * PiAA +
                  (mG * SGa**2 + mg * Sga**2) * 2 * Piaa + 
                  2 * (mG * SGA * SGa + mg * SgA * Sga) * (PiAa + PiaA))

        return Nprime

    @cached_property
    def _get__numeric_simulation_jacobian(self):
        """Wraps the symbolic system of equations for use in simulation."""
        tmp_args = (self._male_allele_shares + self._female_allele_shares + 
                    self._female_signaling_probs + self._male_screening_probs +
                    self._payoffs)
        tmp_jacobian = sp.lambdify(tmp_args, self._symbolic_simulation_jacobian, 
                                 modules=['numpy'])
        return tmp_jacobian

    @cached_property
    def _get__numeric_simulation_system(self):
        """Wraps the symbolic system of equations for use in simulation."""
        tmp_args = (self._male_allele_shares + self._female_allele_shares + 
                    self._female_signaling_probs + self._male_screening_probs +
                    self._payoffs)
        tmp_system = sp.lambdify(tmp_args, self._symbolic_simulation_system, 
                                 modules=['numpy'])
        return tmp_system

    def _get__payoffs(self):
        """Payoff parameters (from a Prisoner's dilemma)."""
        return sp.var('PiaA, PiAA, Piaa, PiAa')

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

    @cached_property
    def _get__symbolic_simulation_jacobian(self):
        """Symbolic Jacobian matrix for stability analysis."""
        # extract variables
        mGA, mGa, mgA, mga = self._male_allele_shares
        fGA, fGa, fgA, fga = self._female_allele_shares
        endog_vars = [mGA, mGa, mgA, mga, fGA, fGa, fgA, fga]

        # compute the jacobian (this may take a while!)
        jacobian = self._symbolic_simulation_system.jacobian(endog_vars)

        return jacobian

    def _get__symbolic_simulation_system(self):
        """Symbolic system of equations for model simulation."""
        system = sp.Matrix([self._equation_1, self._equation_2, 
                            self._equation_3, self._equation_4, 
                            self._equation_5, self._equation_6, 
                            self._equation_7, self._equation_8])
        return system

    @cached_property
    def _get__symbolic_steady_state_jacobian(self):
        """Symbolic Jacobian matrix for steady state solver."""
        # extract variables
        mGA, mGa, mgA, mga = self._male_allele_shares
        fGA, fGa, fgA, fga = self._female_allele_shares
        endog_vars = [mGA, mGa, mgA, fGA, fGa, fgA]

        # compute the jacobian (this may take a while!)
        jacobian = self._symbolic_steady_state_system.jacobian(endog_vars)

        return jacobian

    def _get__symbolic_steady_state_system(self):
        """Symbolic system of equation for steady state solver."""
        system = sp.Matrix([self._equation_1, self._equation_2, 
                            self._equation_3, self._equation_5, 
                            self._equation_6, self._equation_7])
        return system

    def F(self, X):
        """Equation of motion for population allele shares."""
        out = self._numeric_simulation_system(*X, **self.params)
        return np.array(out).flatten()

    def F_jacobian(self, X):
        """Jacobian for equation of motion."""
        out = self._numeric_simulation_jacobian(*X, **self.params)
        return np.array(out)

    def simulate(self, initial_condition, T=10):
        """Simulates a run of the model given some initial_condition."""

        # sets up the trajectory array
        traj = np.empty((8, T))
        traj[:,0] = initial_condition

        # runs the simulation
        for t in range(1,T):
            traj[:,t] = self.F(traj[:,t-1])

        return traj



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    params = {'dA':0.25, 'da':0.75, 'eA':0.25, 'ea':0.5, 'PiaA':6.0, 'PiAA':5.0, 
              'Piaa':4.0, 'PiAa':3.0}

    model = Model(params=params)

    # simulate the model
    initial_condition = np.array([0.05, 0.05, 0.05, 0.85, 0.05, 0.05, 0.05, 0.85])
    traj = model.simulate(initial_condition, T=100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    axes[0].plot(traj[0], label=r'$m_{GA}$')
    axes[0].plot(traj[1], label=r'$m_{Ga}$')
    axes[0].plot(traj[2], label=r'$m_{gA}$')
    axes[0].plot(traj[3], label=r'$m_{ga}$')

    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel('Population shares', family='serif', fontsize=15)
    axes[0].set_title('Males', family='serif', fontsize=20)
    axes[0].legend(loc=0, frameon=False)

    axes[1].plot(traj[4], label=r'$f_{GA}$')
    axes[1].plot(traj[5], label=r'$f_{Ga}$')
    axes[1].plot(traj[6], label=r'$f_{gA}$')
    axes[1].plot(traj[7], label=r'$f_{ga}$')

    axes[1].set_ylim(0, 1)
    axes[1].set_title('Females', family='serif', fontsize=20)
    axes[1].legend(loc=0, frameon=False)

    plt.show()

