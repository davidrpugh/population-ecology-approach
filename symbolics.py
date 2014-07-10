"""
Defines the non-linear equations describing our model as SymPy expressions.

"""
import sympy as sp 

# Population shares of adult females carrying alleles (Gamma, Theta)
female_allele_shares = sp.var('fGA, fGa, fgA, fga')

# Female signaling probabilities
female_signaling_probs = sp.var('dA, da') 

# Population shares of adult males carrying alleles (Gamma, Theta)
male_allele_shares = sp.var('mGA, mGa, mgA, mga')

# Male screening probabilities
male_screening_probs = sp.var('eA, ea') 

# Payoff parameters (from a Prisoner's dilemma)
matching_payoffs = sp.var('PiaA, PiAA, Piaa, PiAa')

# extract the important variables
fGA, fGa, fgA, fga = female_allele_shares
mGA, mGa, mgA, mga = male_allele_shares
dA, da = female_signaling_probs
eA, ea = male_screening_probs
PiaA, PiAA, Piaa, PiAa = matching_payoffs

# Population shares by phenotype."""
mG = mGA + mGa
mg = mgA + mga
fA = fGA + fgA
fa = fGa + fga

# Probabilities that male matches with desired female.
SGA = (dA * fA) / (dA * fA + (1 - eA) * (1 - da) * fa)
SGa = 1 - SGA
Sga = (da * fa) / (da * fa + (1 - ea) * (1 - dA) * fA)
SgA = 1 - Sga

# Total female children in the next generation.
Nprime = ((mG * SGA**2 + mg * SgA**2) * 2 * PiAA + 
          (mG * SGa**2 + mg * Sga**2) * 2 * Piaa + 
          2 * (mG * SGA * SGa + mg * SgA * Sga) * (PiAa + PiaA))

# define the various recurrence relations
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

# symbolic system of equations for model simulation
endog_vars = [mGA, mGa, mgA, mga, fGA, fGa, fgA, fga]
model_system = sp.Matrix([eqn1, eqn2, eqn3, eqn4, eqn5, eqn6, eqn7, eqn8])

# symbolic model Jacobian for stability analysis
model_jacobian = model_system.jacobian(endog_vars)

# steady state of the model makes residual zero
residual = model_system - sp.Matrix(endog_vars)

# residual Jacobian is an input to steady state solver
residual_jacobian = residual.jacobian(endog_vars)

