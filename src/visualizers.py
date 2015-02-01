import pandas as pd


class Visualizer(object):
    """Base class for visualizing model simulations."""

    def __init__(self, array):
        self.simulation, self.distribution = self._array_to_dataframes(array)

    @property
    def number_A_female_adults(self):
        r"""
        Number of female adults carrying the `A` allele of the :math:`\alpha`
        gene.

        :getter: Return the number of female adults carrying the `A` allele.
        :type: pandas.Series

        """
        return self.number_GA_female_adults + self.number_gA_female_adults

    @property
    def number_A_female_offspring(self):
        r"""
        Number of female offspring carrying the `A` allele of the
        :math:`\alpha` gene.

        :getter: Return the number of female offspring carrying the `A` allele.
        :type: pandas.Series

        """
        A_female_offspring = self.simulation['Female Offspring Genotypes'][[0, 2]]
        return A_female_offspring.sum(axis=1)

    @property
    def number_A_male_adults(self):
        r"""
        Number of male adults carrying the `A` allele of the :math:`\alpha`
        gene.

        :getter: Return the number of male adults carrying the `A` allele.
        :type: pandas.Series

        """
        A_male_adults = self.simulation['Adult Male Genotypes'][[0, 2]]
        return A_male_adults.sum(axis=1)

    @property
    def number_A_male_offspring(self):
        r"""
        Number of male offspring carrying the `A` allele of the :math:`\alpha`
        gene.

        :getter: Return the number of male offspring carrying the `A` allele.
        :type: pandas.Series

        Notes
        -----
        By construction, the sex ratio at birth for male and females is 1:1 and
        thus the number of male offspring carrying the `A` allele of the
        :math:`\alpha` gene is the same as the number of female offspring
        carrying that allele.

        """
        return self.number_A_female_offspring

    @property
    def number_a_female_adults(self):
        r"""
        Number of female adults carrying the `a` allele of the :math:`\alpha`
        gene.

        :getter: Return the number of female adults carrying the `a` allele.
        :type: pandas.Series

        """
        return self.number_Ga_female_adults + self.number_ga_female_adults

    @property
    def number_a_female_offspring(self):
        r"""
        Number of female offspring carrying the `a` allele of the :math:`\alpha`
        gene.

        :getter: Return the number of female offspring carrying the `a` allele.
        :type: pandas.Series

        """
        a_female_offspring = self.simulation['Female Offspring Genotypes'][[1, 3]]
        return a_female_offspring.sum(axis=1)

    @property
    def number_a_male_adults(self):
        r"""
        Number of male adults carrying the `a` allele of the :math:`\alpha`
        gene.

        :getter: Return the number of male adults carrying the `a` allele.
        :type: pandas.Series

        """
        a_male_adults = self.simulation['Adult Male Genotypes'][[1, 3]]
        return a_male_adults.sum(axis=1)

    @property
    def number_a_male_offspring(self):
        r"""
        Number of male offspring carrying the `a` allele of the :math:`\alpha`
        gene.

        :getter: Return the number of male offspring carrying the `a` allele.
        :type: pandas.Series

        Notes
        -----
        By construction, the sex ratio at birth for male and females is 1:1 and
        thus the number of male offspring carrying the `a` allele of the
        :math:`\alpha` gene is the same as the number of female offspring
        carrying that allele.

        """
        return self.number_a_female_offspring

    @property
    def number_female_offspring(self):
        """
        Total number of offspring produced in a generation.

        :getter: Return the toal number of offspring.
        :type: pandas.Series

        Notes
        -----
        By construction, the sex ratio at birth for male and females is 1:1 the
        total number of offspring produced in each generation is twice the total
        number of female offspring.

        """
        female_offspring = self.simulation['Female Offspring Genotypes'][[0, 1, 2, 3]]
        return female_offspring.sum(axis=1)

    @property
    def number_G_female_adults(self):
        r"""
        Number of female adults carrying the `G` allele of the :math:`\gamma`
        gene.

        :getter: Return the number of female adults carrying the `G` allele.
        :type: pandas.Series

        """
        return self.number_GA_female_adults + self.number_Ga_female_adults

    @property
    def number_G_female_offspring(self):
        r"""
        Number of female offspring carrying the `G` allele of the :math:`\gamma`
        gene.

        :getter: Return the number of female offspring carrying the `G` allele.
        :type: pandas.Series

        """
        G_female_offspring = self.simulation['Female Offspring Genotypes'][[0, 1]]
        return G_female_offspring.sum(axis=1)

    @property
    def number_G_male_adults(self):
        r"""
        Number of male adults carrying the `G` allele of the :math:`\gamma`
        gene.

        :getter: Return the number of male adults carrying the `G` allele.
        :type: pandas.Series

        """
        G_male_adults = self.simulation['Adult Male Genotypes'][[0, 1]]
        return G_male_adults.sum(axis=1)

    @property
    def number_G_male_offspring(self):
        r"""
        Number of male offspring carrying the `G` allele of the :math:`\gamma`
        gene.

        :getter: Return the number of male offspring carrying the `G` allele.
        :type: pandas.Series

        Notes
        -----
        By construction, the sex ratio at birth for male and females is 1:1 and
        thus the number of male offspring carrying the `G` allele of the
        :math:`\gamma` gene is the same as the number of female offspring
        carrying that allele.

        """
        return self.number_G_female_offspring

    @property
    def number_g_female_adults(self):
        r"""
        Number of female adults carrying the `g` allele of the :math:`\gamma`
        gene.

        :getter: Return the number of female adults carrying the `g` allele.
        :type: pandas.Series

        """
        return self.number_gA_female_adults + self.number_ga_female_adults

    @property
    def number_g_female_offspring(self):
        r"""
        Number of female offspring carrying the `g` allele of the :math:`\gamma`
        gene.

        :getter: Return the number of female offspring carrying the `g` allele.
        :type: pandas.Series

        """
        g_female_offspring = self.simulation['Female Offspring Genotypes'][[2, 3]]
        return g_female_offspring.sum(axis=1)

    @property
    def number_g_male_adults(self):
        r"""
        Number of male adults carrying the `g` allele of the :math:`\gamma`
        gene.

        :getter: Return the number of male adults carrying the `g` allele.
        :type: pandas.Series

        """
        g_male_adults = self.simulation['Adult Male Genotypes'][[2, 3]]
        return g_male_adults.sum(axis=1)

    @property
    def number_g_male_offspring(self):
        r"""
        Number of male offspring carrying the `g` allele of the :math:`\gamma`
        gene.

        :getter: Return the number of male offspring carrying the `g` allele.
        :type: pandas.Series

        Notes
        -----
        By construction, the sex ratio at birth for male and females is 1:1 and
        thus the number of male offspring carrying the `g` allele of the
        :math:`\gamma` gene is the same as the number of female offspring
        carrying that allele.

        """
        return self.number_g_female_offspring

    @property
    def number_GA_female_adults(self):
        r"""
        Number of female adults carrying the `GA` genotype.

        :getter: Return the number of female adults carrying the `GA` genotype.
        :type: pandas.Series

        """
        GA_female_adults = (self.distribution.xs(0, 1, 'female1_genotype') +
                            self.distribution.xs(0, 1, 'female2_genotype'))
        return GA_female_adults.sum(axis=1)

    @property
    def number_GA_female_offspring(self):
        r"""
        Number of female offspring carrying the `GA` genotype.

        :getter: Return the number of female offspring carrying the `GA` genotype.
        :type: pandas.Series

        """
        return self.simulation['Female Offspring Genotypes'][0]

    @property
    def number_Ga_female_adults(self):
        r"""
        Number of female adults carrying the `Ga` genotype.

        :getter: Return the number of female adults carrying the `Ga` genotype.
        :type: pandas.Series

        """
        Ga_female_adults = (self.distribution.xs(1, 1, 'female1_genotype') +
                            self.distribution.xs(1, 1, 'female2_genotype'))
        return Ga_female_adults.sum(axis=1)

    @property
    def number_Ga_female_offspring(self):
        r"""
        Number of female offspring carrying the `Ga` genotype.

        :getter: Return the number of female offspring carrying the `Ga` genotype.
        :type: pandas.Series

        """
        return self.simulation['Female Offspring Genotypes'][1]

    @property
    def number_gA_female_adults(self):
        r"""
        Number of female adults carrying the `gA` genotype.

        :getter: Return the number of female adults carrying the `gA` genotype.
        :type: pandas.Series

        """
        gA_female_adults = (self.distribution.xs(2, 1, 'female1_genotype') +
                            self.distribution.xs(2, 1, 'female2_genotype'))
        return gA_female_adults.sum(axis=1)

    @property
    def number_gA_female_offspring(self):
        r"""
        Number of female offspring carrying the `gA` genotype.

        :getter: Return the number of female offspring carrying the `gA` genotype.
        :type: pandas.Series

        """
        return self.simulation['Female Offspring Genotypes'][2]

    @property
    def number_ga_female_adults(self):
        r"""
        Number of female adults carrying the `ga` genotype.

        :getter: Return the number of female adults carrying the `ga` genotype.
        :type: pandas.Series

        """
        ga_female_adults = (self.distribution.xs(3, 1, 'female1_genotype') +
                            self.distribution.xs(3, 1, 'female2_genotype'))
        return ga_female_adults.sum(axis=1)

    @property
    def number_ga_female_offspring(self):
        r"""
        Number of female offspring carrying the `ga` genotype.

        :getter: Return the number of female offspring carrying the `ga` genotype.
        :type: pandas.Series

        """
        return self.simulation['Female Offspring Genotypes'][3]

    @property
    def number_GA_male_adults(self):
        r"""
        Number of male adults carrying the `GA` genotype.

        :getter: Return the number of male adults carrying the `GA` genotype.
        :type: pandas.Series

        """
        return self.simulation['Adult Male Genotypes'][0]

    @property
    def number_Ga_male_adults(self):
        r"""
        Number of male adults carrying the `Ga` genotype.

        :getter: Return the number of male adults carrying the `Ga` genotype.
        :type: pandas.Series

        """
        return self.simulation['Adult Male Genotypes'][1]

    @property
    def number_gA_male_adults(self):
        r"""
        Number of male adults carrying the `gA` genotype.

        :getter: Return the number of male adults carrying the `gA` genotype.
        :type: pandas.Series

        """
        return self.simulation['Adult Male Genotypes'][2]

    @property
    def number_ga_male_adults(self):
        r"""
        Number of male adults carrying the `ga` genotype.

        :getter: Return the number of male adults carrying the `ga` genotype.
        :type: pandas.Series

        """
        return self.simulation['Adult Male Genotypes'][3]

    @property
    def share_A_female_adults(self):
        r"""
        Share of female adults carrying the `A` allele of the :math:`\alpha`
        gene.

        :getter: Return the share of female offspring carrying the `A` allele.
        :type: pandas.Series

        Notes
        -----
        In the one male and two females family unit, the number of adult
        females is normalized to two. Thus to compute shares one needs only
        to multiply by 0.5.

        """
        return 0.5 * self.number_A_female_adults

    @property
    def share_A_female_offspring(self):
        r"""
        Share of female offspring carrying the `A` allele of the :math:`\alpha`
        gene.

        :getter: Return the share of female offspring carrying the `A` allele.
        :type: pandas.Series

        """
        return self.number_A_female_offspring / self.number_female_offspring

    @property
    def share_a_female_adults(self):
        r"""
        Share of female adults carrying the `a` allele of the :math:`\alpha`
        gene.

        :getter: Return the share of female offspring carrying the `a` allele.
        :type: pandas.Series

        Notes
        -----
        In the one male and two females family unit, the number of adult
        females is normalized to two. Thus to compute shares one needs only
        to multiply by 0.5.

        """
        return 0.5 * self.number_a_female_adults

    @property
    def share_a_female_offspring(self):
        r"""
        Share of female offspring carrying the `a` allele of the :math:`\alpha`
        gene.

        :getter: Return the share of female offspring carrying the `a` allele.
        :type: pandas.Series

        """
        return self.number_a_female_offspring / self.number_female_offspring

    @property
    def share_G_female_adults(self):
        r"""
        Share of female adults carrying the `G` allele of the :math:`\gamma`
        gene.

        :getter: Return the share of female offspring carrying the `G` allele.
        :type: pandas.Series

        Notes
        -----
        In the one male and two females family unit, the number of adult
        females is normalized to two. Thus to compute shares one needs only
        to multiply by 0.5.

        """
        return 0.5 * self.number_G_female_adults

    @property
    def share_G_female_offspring(self):
        r"""
        Share of female offspring carrying the `G` allele of the :math:`\gamma`
        gene.

        :getter: Return the share of female offspring carrying the `G` allele.
        :type: pandas.Series

        """
        return self.number_G_female_offspring / self.number_female_offspring

    @property
    def share_g_female_adults(self):
        r"""
        Share of female adults carrying the `g` allele of the :math:`\gamma`
        gene.

        :getter: Return the share of female offspring carrying the `g` allele.
        :type: pandas.Series

        Notes
        -----
        In the one male and two females family unit, the number of adult
        females is normalized to two. Thus to compute shares one needs only
        to multiply by 0.5.

        """
        return 0.5 * self.number_g_female_adults

    @property
    def share_g_female_offspring(self):
        r"""
        Share of female offspring carrying the `g` allele of the :math:`\gamma`
        gene.

        :getter: Return the share of female offspring carrying the `g` allele.
        :type: pandas.Series

        """
        return self.number_g_female_offspring / self.number_female_offspring

    @property
    def share_GA_female_adults(self):
        r"""
        Share of female adults carrying the `GA` genotype.

        :getter: Return the share of female adults carrying the `GA` genotype.
        :type: pandas.Series

        Notes
        -----
        In the one male and two females family unit, the number of adult
        females is normalized to two. Thus to compute shares one needs only
        to multiply by 0.5.

        """
        return 0.5 * self.number_GA_female_adults

    @property
    def share_GA_female_offspring(self):
        r"""
        Share of female offspring carrying the `GA` genotype.

        :getter: Return the share of female offspring carrying the `GA` genotype.
        :type: pandas.Series

        """
        return self.number_GA_female_offspring / self.number_female_offspring

    @property
    def share_Ga_female_adults(self):
        r"""
        Share of female adults carrying the `Ga` genotype.

        :getter: Return the share of female adults carrying the `Ga` genotype.
        :type: pandas.Series

        Notes
        -----
        In the one male and two females family unit, the number of adult
        females is normalized to two. Thus to compute shares one needs only
        to multiply by 0.5.

        """
        return 0.5 * self.number_Ga_female_adults

    @property
    def share_Ga_female_offspring(self):
        r"""
        Share of female offspring carrying the `Ga` genotype.

        :getter: Return the share of female offspring carrying the `Ga` genotype.
        :type: pandas.Series

        """
        return self.number_Ga_female_offspring / self.number_female_offspring

    @property
    def share_gA_female_adults(self):
        r"""
        Share of female adults carrying the `gA` genotype.

        :getter: Return the share of female adults carrying the `gA` genotype.
        :type: pandas.Series

        Notes
        -----
        In the one male and two females family unit, the number of adult
        females is normalized to two. Thus to compute shares one needs only
        to multiply by 0.5.

        """
        return 0.5 * self.number_gA_female_adults

    @property
    def share_gA_female_offspring(self):
        r"""
        Share of female offspring carrying the `gA` genotype.

        :getter: Return the share of female offspring carrying the `gA` genotype.
        :type: pandas.Series

        """
        return self.number_gA_female_offspring / self.number_female_offspring

    @property
    def share_ga_female_adults(self):
        r"""
        Share of female adults carrying the `ga` genotype.

        :getter: Return the share of female adults carrying the `ga` genotype.
        :type: pandas.Series

        Notes
        -----
        In the one male and two females family unit, the number of adult
        females is normalized to two. Thus to compute shares one needs only
        to multiply by 0.5.

        """
        return 0.5 * self.number_ga_female_adults

    @property
    def share_ga_female_offspring(self):
        r"""
        Share of female offspring carrying the `ga` genotype.

        :getter: Return the share of female offspring carrying the `ga` genotype.
        :type: pandas.Series

        """
        return self.number_ga_female_offspring / self.number_female_offspring

    def _array_to_dataframes(self, array):
        """Convert an array into suitably formated pandas.DataFrames."""
        genotypes_df = self._genotype_numbers_to_dataframe(array)
        genotype_configs_df = self._genotype_configs_to_dataframe(array)
        return genotypes_df, genotype_configs_df

    def _genotype_numbers_to_dataframe(self, array):
        """
        Convert array containing the numbers of adult males and female
        offspring by genotype to a suitably formated pandas.DataFrame.

        """
        idx = pd.Index(array[:, 0], name='Time')
        values = array[:, 1:9]
        labels = ['Adult Male Genotypes', 'Female Offspring Genotypes']
        categories = range(4)
        cols = pd.MultiIndex.from_product([labels, categories])
        return pd.DataFrame(values, index=idx, columns=cols)

    def _genotype_configs_to_dataframe(self, array):
        """
        Convert array containing the numbers of various family genotype
        configurations into to a suitably formated pandas.DataFrame.

        """
        idx = pd.Index(array[:, 0], name='Time')
        values = array[:, 9:]
        labels = ['male_genotype', 'female1_genotype', 'female2_genotype']
        categories = [range(4), range(4), range(4)]
        cols = pd.MultiIndex.from_product(categories, names=labels)
        return pd.DataFrame(values, index=idx, columns=cols)
