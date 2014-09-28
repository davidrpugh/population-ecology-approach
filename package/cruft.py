import sympy as sym

import sandbox

# number of female children of particular genotype
girls = sym.DeferredVector('f')

# number of male adults of particular genotype
men = sym.DeferredVector('M')

# Female signaling probabilities
female_signaling_probs = sym.var('dA, da')
dA, da = female_signaling_probs

# Male screening probabilities
male_screening_probs = sym.var('eG, eg')
eG, eg = male_screening_probs

# Female population by phenotype.
altruistic_girls = girls[0] + girls[2]
selfish_girls = girls[1] + girls[3]

# Conditional phenotype matching probabilities
true_altruistic_girls = dA * altruistic_girls
false_altruistic_girls = (1 - da) * selfish_girls
mistaken_for_altruistic_girls = (1 - eG) * false_altruistic_girls
altruist_adoption_pool = true_altruistic_girls + mistaken_for_altruistic_girls

true_selfish_girls = da * selfish_girls
false_selfish_girls = (1 - dA) * altruistic_girls
mistaken_for_selfish_girls = (1 - eg) * false_selfish_girls
selfish_adoption_pool = true_selfish_girls + mistaken_for_selfish_girls

SGA = true_altruistic_girls / altruist_adoption_pool
SGa = 1 - SGA
Sga = true_selfish_girls / selfish_adoption_pool
SgA = 1 - Sga

params = {'dA': 0.5, 'da': 0.5, 'eG': 0.5, 'eg': 0.5, 'c': 1.0,
          'PiaA': 9.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}
model = sandbox.OneMaleTwoFemalesModel(params, SGA, Sga)
