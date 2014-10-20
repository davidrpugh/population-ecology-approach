import sympy as sym

import families

# number of female children of particular genotype
girls = sym.DeferredVector('f')

# number of male adults of particular genotype
men = sym.DeferredVector('M')

# Female signaling probabilities
female_signaling_probs = sym.var('d_A, d_a')
d_A, d_a = female_signaling_probs

# Male screening probabilities
male_screening_probs = sym.var('e_G, e_g')
e_G, e_g = male_screening_probs

# Payoff parameters (from a Prisoner's dilemma)
prisoners_dilemma_payoffs = sym.var('PiaA, PiAA, Piaa, PiAa')
PiaA, PiAA, Piaa, PiAa = prisoners_dilemma_payoffs

# Female fecundity scaling factor
fecundity_factor = sym.var('c')

# Female population by phenotype.
altruistic_girls = girls[0] + girls[2]
selfish_girls = girls[1] + girls[3]

# Conditional phenotype matching probabilities
true_altruistic_girls = d_A * altruistic_girls
false_altruistic_girls = (1 - d_a) * selfish_girls
mistaken_for_altruistic_girls = (1 - e_G) * false_altruistic_girls
altruist_adoption_pool = true_altruistic_girls + mistaken_for_altruistic_girls

true_selfish_girls = d_a * selfish_girls
false_selfish_girls = (1 - d_A) * altruistic_girls
mistaken_for_selfish_girls = (1 - e_g) * false_selfish_girls
selfish_adoption_pool = true_selfish_girls + mistaken_for_selfish_girls

SGA = true_altruistic_girls / altruist_adoption_pool
Sga = true_selfish_girls / selfish_adoption_pool

# define some parameters
hierarchy_params = {'c': 2.0, 'eG': 0.5, 'eg': 0.5, 'dA': 0.5, 'da': 0.5,
                    'PiaA': 9.0, 'PiAA': 5.0, 'Piaa': 3.0, 'PiAa': 2.0}

family = families.OneMaleTwoFemales(params=hierarchy_params, SGA=SGA, Sga=Sga)
