import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Simulator(object):
    """Class for simulating the Pugh-Schaffer-Seabright model."""

    def __init__(self, family):
        """
        Create an instance of the Simulator class.

        Parameters
        ----------
        family : families.family
            Instance of the families.Family class defining a family unit.

        """
        self.family = family

    @property
    def initial_condition(self):
        """
        Initial condition for a simulation.

        :getter: Return the current initial condition.
        :setter: Set a new initial condtion.
        :type: numpy.ndarray

        Notes
        -----
        Currently, we assume that the initial condition consists of two
        isolated sub-populations. One group consistst of males and females
        carrying the GA genotype; the other group consists of males and females
        carrying the ga genotype.

        The setter method requires the user to specify a new value for the
        initial share of males carrying the GA genotype in the combined (i.e.,
        mixed) population. The getter method returns an initial condition
        for all eight of endogenous variables.

        """
        return self._initial_condition

    @initial_condition.setter
    def initial_condition(self, mGA):
        """Set a new initial condition."""
        # initial condition for male shares
        mga = 1 - mGA
        initial_male_shares = np.array([mGA, 0.0, 0.0, mga])

        # initial number female children
        fGA0 = self.family.params['c'] * self.family.params['PiAA'] * mGA
        fga0 = self.family.params['c'] * self.family.params['Piaa'] * mga
        initial_number_females = np.array([fGA0, 0.0, 0.0, fga0])

        initial_condition = np.hstack((initial_male_shares,
                                       initial_number_females))

        self._initial_condition = initial_condition

    def _simulate_fixed_trajectory(self, initial_condition, T):
        """Simulates a trajectory of fixed length."""
        # set up the trajectory array
        traj = np.empty((8, T))
        traj[:, 0] = initial_condition

        # run the simulation
        for t in range(1, T):
            current_shares = traj[:, t-1]
            new_shares = self.F(current_shares)
            traj[:, t] = new_shares

        return traj

    def _simulate_variable_trajectory(self, initial_condition, rtol):
        """Simulates a trajectory of variable length."""
        # set up the trajectory array
        traj = initial_condition.reshape((8, 1))

        # initialize delta
        delta = np.ones(8)

        # run the simulation
        while np.any(np.greater(delta, rtol)):
            current_shares = traj[:, -1]
            new_shares = self.F(current_shares)
            delta = np.abs(new_shares - current_shares)

            # update the trajectory
            traj = np.hstack((traj, new_shares[:, np.newaxis]))

        return traj

    def _trajectory_to_dataframe(self, trajectory):
        """Converts a numpy array into a suitably formated pandas.DataFrame."""
        idx = pd.Index(range(trajectory.shape[1]), name='Time')
        headers = ["Male Adult Genotypes", "Female Children Genotypes"]
        genotypes = range(4)
        cols = pd.MultiIndex.from_product([headers, genotypes])
        df = pd.DataFrame(trajectory.T, index=idx, columns=cols)
        return df

    def F(self, X):
        """Equation of motion for population allele shares."""
        out = self.family._numeric_system(X[:4], X[4:], **self.family.params)
        return out.ravel()

    def F_jacobian(self, X):
        """Jacobian for equation of motion."""
        jac = self.family._numeric_jacobian(X[:4], X[4:], **self.family.params)
        return jac

    def simulate(self, rtol=None, T=None):
        """
        Simulates the model for either fixed of variable number of time steps.

        Parameters
        ----------
        T : int (default=None)
            The number of time steps to simulate.
        rtol : float (default=None)
            Simulate the model until the relative difference between timesteps
            is sufficiently small.

        Returns
        -------
        df : pandas.DataFrame
            Hierarchical dataframe representing a simulation of the model.

        """
        if T is not None:
            traj = self._simulate_fixed_trajectory(self.initial_condition, T)
        elif rtol is not None:
            traj = self._simulate_variable_trajectory(self.initial_condition, rtol)
        else:
            raise ValueError("One of 'T' or 'rtol' must be specified.")

        df = self._trajectory_to_dataframe(traj)

        return df


class Distribution(object):

    __distribution = None

    def __init__(self, family, simulation):
        self.family = family
        self.simulation = simulation

    @property
    def distribution(self):
        """
        Hierarchical DataFrame of time series for family sizes.

        :getter: Return the current DataFrame.
        :type: pandas.DataFrame

        """
        if self.__distribution is None:
            self.__distribution = self.compute_distribution(self.simulation)
        return self.__distribution

    @property
    def alpha_natural_selection_pressure(self):
        """
        Measure of natural selection pressure on the alpha gene.

        :getter: Return the current time series for natural selection pressure.
        :type: pandas.Series

        """
        pressure = (self.alpha_natural_selection_pressure_females +
                    self.alpha_natural_selection_pressure_males)
        return pressure

    @property
    def alpha_natural_selection_pressure_females(self):
        """
        Measure of the component of natural selection pressure on the alpha
        gene coming from the females.

        :getter: Return the current time series for natural selection pressure.
        :type: pandas.Series

        """
        avg_A_female_children = (self.number_A_female_children /
                                 self.number_A_female_adults)
        avg_a_female_children = (self.number_a_female_children /
                                 self.number_a_female_adults)
        return np.log(avg_A_female_children) - np.log(avg_a_female_children)

    @property
    def alpha_natural_selection_pressure_males(self):
        """
        Measure of the component of natural selection pressure on the alpha
        gene coming from the males.

        :getter: Return the current time series for natural selection pressure.
        :type: pandas.Series

        """
        A_ratio = (self.number_A_male_adults.shift(-1) /
                   self.number_A_male_children)
        a_ratio = (self.number_a_male_adults.shift(-1) /
                   self.number_a_male_children)
        return np.log(A_ratio) - np.log(a_ratio)

    @property
    def alpha_sexual_selection_pressure(self):
        """
        Measure of the sexual selection pressure on the alpha gene.

        :getter: Return the current time series for sexual selection pressure.
        :type: pandas.Series

        """
        pressure = (self.alpha_sexual_selection_pressure_females +
                    self.alpha_sexual_selection_pressure_males)
        return pressure

    @property
    def alpha_sexual_selection_pressure_females(self):
        """
        Measure of the component of sexual selection pressure on the alpha
        gene from females.

        :getter: Return the current time series for sexual selection pressure.
        :type: pandas.Series

        """
        A_ratio = (self.number_A_female_adults.shift(-1) /
                   self.number_A_female_children)
        a_ratio = (self.number_a_female_adults.shift(-1) /
                   self.number_a_female_children)
        return np.log(A_ratio) - np.log(a_ratio)

    @property
    def alpha_sexual_selection_pressure_males(self):
        """
        Measure of the component of sexual selection pressure on the alpha
        gene from males.

        :getter: Return the current time series for sexual selection pressure.
        :type: pandas.Series

        """
        avg_A_male_children = (self.number_A_male_children /
                               self.number_A_male_adults)
        avg_a_male_children = (self.number_a_male_children /
                               self.number_a_male_adults)
        return np.log(avg_A_male_children) - np.log(avg_a_male_children)

    @property
    def alpha_selection_pressure(self):
        """
        Measure of total selection pressure on the alpha gene.

        :getter: Return the current time series for total selection pressure.
        :type: pandas.Series

        """
        total_pressure = (self.alpha_natural_selection_pressure +
                          self.alpha_sexual_selection_pressure)
        return total_pressure

    @property
    def gamma_natural_selection_pressure(self):
        """
        Measure of natural selection pressure on the gamma gene.

        :getter: Return the current time series for natural selection pressure.
        :type: pandas.Series

        """
        pressure = (self.gamma_natural_selection_pressure_females +
                    self.gamma_natural_selection_pressure_males)
        return pressure

    @property
    def gamma_natural_selection_pressure_females(self):
        """
        Measure of the component of natural selection pressure on the gamma
        gene coming from the females.

        :getter: Return the current time series for natural selection pressure.
        :type: pandas.Series

        """
        avg_G_female_children = (self.number_G_female_children /
                                 self.number_G_female_adults)
        avg_g_female_children = (self.number_g_female_children /
                                 self.number_g_female_adults)
        return np.log(avg_G_female_children) - np.log(avg_g_female_children)

    @property
    def gamma_natural_selection_pressure_males(self):
        """
        Measure of the component of natural selection pressure on the gamma
        gene coming from the males.

        :getter: Return the current time series for natural selection pressure.
        :type: pandas.Series

        """
        G_ratio = (self.number_G_male_adults.shift(-1) /
                   self.number_G_male_children)
        g_ratio = (self.number_g_male_adults.shift(-1) /
                   self.number_g_male_children)
        return np.log(G_ratio) - np.log(g_ratio)

    @property
    def gamma_sexual_selection_pressure(self):
        """
        Measure of sexual selection pressure on the gamma gene.

        :getter: Return the current time series for sexual selection pressure.
        :type: pandas.Series

        """
        pressure = (self.gamma_sexual_selection_pressure_females +
                    self.gamma_sexual_selection_pressure_males)
        return pressure

    @property
    def gamma_sexual_selection_pressure_females(self):
        """
        Measure of the component of sexual selection pressure on the gamma
        gene from females.

        :getter: Return the current time series for sexual selection pressure.
        :type: pandas.Series

        """
        G_ratio = (self.number_G_female_adults.shift(-1) /
                   self.number_G_female_children)
        g_ratio = (self.number_g_female_adults.shift(-1) /
                   self.number_g_female_children)
        return np.log(G_ratio) - np.log(g_ratio)

    @property
    def gamma_sexual_selection_pressure_males(self):
        """
        Measure of the component of sexual selection pressure on the gamma
        gene from males.

        :getter: Return the current time series for sexual selection pressure.
        :type: pandas.Series

        """
        avg_G_male_children = (self.number_G_male_children /
                               self.number_G_male_adults)
        avg_g_male_children = (self.number_g_male_children /
                               self.number_g_male_adults)
        return np.log(avg_G_male_children) - np.log(avg_g_male_children)

    @property
    def gamma_selection_pressure(self):
        """
        Measure of total selection pressure on the gamma gene.

        :getter: Return the current time series for total selection pressure.
        :type: pandas.Series

        """
        total_pressure = (self.gamma_sexual_selection_pressure +
                          self.gamma_natural_selection_pressure)
        return total_pressure

    @property
    def number_A_female_adults(self):
        r"""
        Number of female adults carrying the `A` allele of the :math:`\alpha`
        gene.

        :getter: Return the number of female adults carrying the `A` allele.
        :type: pandas.Series

        """
        A_female_adults = (self.distribution.xs(0, level='female1_genotype') +
                           self.distribution.xs(2, level='female1_genotype') +
                           self.distribution.xs(0, level='female2_genotype') +
                           self.distribution.xs(2, level='female2_genotype'))
        return A_female_adults.sum(axis=0)

    @property
    def number_A_female_children(self):
        r"""
        Number of female children carrying the `A` allele of the :math:`\alpha`
        gene.

        :getter: Return the number of female children carrying the `A` allele.
        :type: pandas.Series

        """
        A_female_children = self.simulation['Female Children Genotypes'][[0, 2]]
        return A_female_children.sum(axis=1)

    @property
    def number_A_male_adults(self):
        r"""
        Number of male adults carrying the `A` allele of the :math:`\alpha`
        gene.

        :getter: Return the number of male adults carrying the `A` allele.
        :type: pandas.Series

        """
        A_male_adults = self.simulation['Male Adult Genotypes'][[0, 2]]
        return A_male_adults.sum(axis=1)

    @property
    def number_A_male_children(self):
        r"""
        Number of male children carrying the `A` allele of the :math:`\alpha`
        gene.

        :getter: Return the number of male children carrying the `A` allele.
        :type: pandas.Series

        Notes
        -----
        By construction, the sex ratio at birth for male and females is 1:1 and
        thus the number of male children carrying the `A` allele of the
        :math:`\alpha` gene is the same as the number of female children
        carrying that allele.

        """
        return self.number_A_female_children

    @property
    def number_a_female_adults(self):
        r"""
        Number of female adults carrying the `a` allele of the :math:`\alpha`
        gene.

        :getter: Return the number of female adults carrying the `a` allele.
        :type: pandas.Series

        """
        a_female_adults = (self.distribution.xs(1, level='female1_genotype') +
                           self.distribution.xs(3, level='female1_genotype') +
                           self.distribution.xs(1, level='female2_genotype') +
                           self.distribution.xs(3, level='female2_genotype'))
        return a_female_adults.sum(axis=0)

    @property
    def number_a_female_children(self):
        r"""
        Number of female children carrying the `a` allele of the :math:`\alpha`
        gene.

        :getter: Return the number of female children carrying the `a` allele.
        :type: pandas.Series

        """
        a_female_children = self.simulation['Female Children Genotypes'][[1, 3]]
        return a_female_children.sum(axis=1)

    @property
    def number_a_male_adults(self):
        r"""
        Number of male adults carrying the `a` allele of the :math:`\alpha`
        gene.

        :getter: Return the number of male adults carrying the `a` allele.
        :type: pandas.Series

        """
        a_male_adults = self.simulation['Male Adult Genotypes'][[1, 3]]
        return a_male_adults.sum(axis=1)

    @property
    def number_a_male_children(self):
        r"""
        Number of male children carrying the `a` allele of the :math:`\alpha`
        gene.

        :getter: Return the number of male children carrying the `a` allele.
        :type: pandas.Series

        Notes
        -----
        By construction, the sex ratio at birth for male and females is 1:1 and
        thus the number of male children carrying the `a` allele of the
        :math:`\alpha` gene is the same as the number of female children
        carrying that allele.

        """
        return self.number_a_female_children

    @property
    def number_children(self):
        """
        Total number of children produced in a generation.

        :getter: Return the toal number of children.
        :type: pandas.Series

        Notes
        -----
        By construction, the sex ratio at birth for male and females is 1:1 the
        total number of children produced in each generation is twice the total
        number of female children.

        """
        return 2 * self.simulation['Female Children Genotypes'][[0, 1, 2, 3]]

    @property
    def number_G_female_adults(self):
        r"""
        Number of female adults carrying the `G` allele of the :math:`\gamma`
        gene.

        :getter: Return the number of female adults carrying the `G` allele.
        :type: pandas.Series

        """
        G_female_adults = (self.distribution.xs(0, level='female1_genotype') +
                           self.distribution.xs(1, level='female1_genotype') +
                           self.distribution.xs(0, level='female2_genotype') +
                           self.distribution.xs(1, level='female2_genotype'))
        return G_female_adults.sum(axis=0)

    @property
    def number_G_female_children(self):
        r"""
        Number of female children carrying the `G` allele of the :math:`\gamma`
        gene.

        :getter: Return the number of female children carrying the `G` allele.
        :type: pandas.Series

        """
        G_female_children = self.simulation['Female Children Genotypes'][[0, 1]]
        return G_female_children.sum(axis=1)

    @property
    def number_G_male_adults(self):
        r"""
        Number of male adults carrying the `G` allele of the :math:`\gamma`
        gene.

        :getter: Return the number of male adults carrying the `G` allele.
        :type: pandas.Series

        """
        G_male_adults = self.simulation['Male Adult Genotypes'][[0, 1]]
        return G_male_adults.sum(axis=1)

    @property
    def number_G_male_children(self):
        r"""
        Number of male children carrying the `G` allele of the :math:`\gamma`
        gene.

        :getter: Return the number of male children carrying the `G` allele.
        :type: pandas.Series

        Notes
        -----
        By construction, the sex ratio at birth for male and females is 1:1 and
        thus the number of male children carrying the `G` allele of the
        :math:`\gamma` gene is the same as the number of female children
        carrying that allele.

        """
        return self.number_G_female_children

    @property
    def number_g_female_adults(self):
        r"""
        Number of female adults carrying the `g` allele of the :math:`\gamma`
        gene.

        :getter: Return the number of female adults carrying the `g` allele.
        :type: pandas.Series

        """
        g_female_adults = (self.distribution.xs(2, level='female1_genotype') +
                           self.distribution.xs(3, level='female1_genotype') +
                           self.distribution.xs(2, level='female2_genotype') +
                           self.distribution.xs(3, level='female2_genotype'))
        return g_female_adults.sum(axis=0)

    @property
    def number_g_female_children(self):
        r"""
        Number of female children carrying the `g` allele of the :math:`\gamma`
        gene.

        :getter: Return the number of female children carrying the `g` allele.
        :type: pandas.Series

        """
        g_female_children = self.simulation['Female Children Genotypes'][[2, 3]]
        return g_female_children.sum(axis=1)

    @property
    def number_g_male_adults(self):
        r"""
        Number of male adults carrying the `g` allele of the :math:`\gamma`
        gene.

        :getter: Return the number of male adults carrying the `g` allele.
        :type: pandas.Series

        """
        g_male_adults = self.simulation['Male Adult Genotypes'][[2, 3]]
        return g_male_adults.sum(axis=1)

    @property
    def number_g_male_children(self):
        r"""
        Number of male children carrying the `g` allele of the :math:`\gamma`
        gene.

        :getter: Return the number of male children carrying the `g` allele.
        :type: pandas.Series

        Notes
        -----
        By construction, the sex ratio at birth for male and females is 1:1 and
        thus the number of male children carrying the `g` allele of the
        :math:`\gamma` gene is the same as the number of female children
        carrying that allele.

        """
        return self.number_g_female_children

    @property
    def number_GA_female_adults(self):
        r"""
        Number of female adults carrying the `GA` genotype.

        :getter: Return the number of female adults carrying the `GA` genotype.
        :type: pandas.Series

        """
        GA_female_adults = (self.distribution.xs(0, level='female1_genotype') +
                            self.distribution.xs(0, level='female2_genotype'))
        return GA_female_adults.sum(axis=0)

    @property
    def number_GA_female_children(self):
        r"""
        Number of female children carrying the `GA` genotype.

        :getter: Return the number of female children carrying the `GA` genotype.
        :type: pandas.Series

        """
        return self.simulation['Female Children Genotypes'][0]

    @property
    def number_Ga_female_adults(self):
        r"""
        Number of female adults carrying the `Ga` genotype.

        :getter: Return the number of female adults carrying the `Ga` genotype.
        :type: pandas.Series

        """
        Ga_female_adults = (self.distribution.xs(1, level='female1_genotype') +
                            self.distribution.xs(1, level='female2_genotype'))
        return Ga_female_adults.sum(axis=0)

    @property
    def number_Ga_female_children(self):
        r"""
        Number of female children carrying the `Ga` genotype.

        :getter: Return the number of female children carrying the `Ga` genotype.
        :type: pandas.Series

        """
        return self.simulation['Female Children Genotypes'][1]

    @property
    def number_gA_female_adults(self):
        r"""
        Number of female adults carrying the `gA` genotype.

        :getter: Return the number of female adults carrying the `gA` genotype.
        :type: pandas.Series

        """
        gA_female_adults = (self.distribution.xs(2, level='female1_genotype') +
                            self.distribution.xs(2, level='female2_genotype'))
        return gA_female_adults.sum(axis=0)

    @property
    def number_gA_female_children(self):
        r"""
        Number of female children carrying the `gA` genotype.

        :getter: Return the number of female children carrying the `gA` genotype.
        :type: pandas.Series

        """
        return self.simulation['Female Children Genotypes'][2]

    @property
    def number_ga_female_adults(self):
        r"""
        Number of female adults carrying the `ga` genotype.

        :getter: Return the number of female adults carrying the `ga` genotype.
        :type: pandas.Series

        """
        ga_female_adults = (self.distribution.xs(3, level='female1_genotype') +
                            self.distribution.xs(3, level='female2_genotype'))
        return ga_female_adults.sum(axis=0)

    @property
    def number_ga_female_children(self):
        r"""
        Number of female children carrying the `ga` genotype.

        :getter: Return the number of female children carrying the `ga` genotype.
        :type: pandas.Series

        """
        return self.simulation['Female Children Genotypes'][3]

    @property
    def number_GA_male_adults(self):
        r"""
        Number of male adults carrying the `GA` genotype.

        :getter: Return the number of male adults carrying the `GA` genotype.
        :type: pandas.Series

        """
        return self.simulation['Male Adult Genotypes'][0]

    @property
    def number_Ga_male_adults(self):
        r"""
        Number of male adults carrying the `Ga` genotype.

        :getter: Return the number of male adults carrying the `Ga` genotype.
        :type: pandas.Series

        """
        return self.simulation['Male Adult Genotypes'][1]

    @property
    def number_gA_male_adults(self):
        r"""
        Number of male adults carrying the `gA` genotype.

        :getter: Return the number of male adults carrying the `gA` genotype.
        :type: pandas.Series

        """
        return self.simulation['Male Adult Genotypes'][2]

    @property
    def number_ga_male_adults(self):
        r"""
        Number of male adults carrying the `ga` genotype.

        :getter: Return the number of male adults carrying the `ga` genotype.
        :type: pandas.Series

        """
        return self.simulation['Male Adult Genotypes'][3]

    @property
    def share_A_female_adults(self):
        r"""
        Share of female adults carrying the `A` allele of the :math:`\alpha`
        gene.

        :getter: Return the share of female children carrying the `A` allele.
        :type: pandas.Series

        Notes
        -----
        In the one male and two females family unit, the number of adult
        females is normalized to two. Thus to compute shares one needs only
        to multiply by 0.5.

        """
        return 0.5 * self.number_A_female_adults

    @property
    def share_A_female_children(self):
        r"""
        Share of female children carrying the `A` allele of the :math:`\alpha`
        gene.

        :getter: Return the share of female children carrying the `A` allele.
        :type: pandas.Series

        """
        return self.number_A_female_children / self.number_children

    @property
    def share_a_female_adults(self):
        r"""
        Share of female adults carrying the `a` allele of the :math:`\alpha`
        gene.

        :getter: Return the share of female children carrying the `a` allele.
        :type: pandas.Series

        Notes
        -----
        In the one male and two females family unit, the number of adult
        females is normalized to two. Thus to compute shares one needs only
        to multiply by 0.5.

        """
        return 0.5 * self.number_a_female_adults

    @property
    def share_a_female_children(self):
        r"""
        Share of female children carrying the `a` allele of the :math:`\alpha`
        gene.

        :getter: Return the share of female children carrying the `a` allele.
        :type: pandas.Series

        """
        return self.number_a_female_children / self.number_children

    @property
    def share_G_female_adults(self):
        r"""
        Share of female adults carrying the `G` allele of the :math:`\gamma`
        gene.

        :getter: Return the share of female children carrying the `G` allele.
        :type: pandas.Series

        Notes
        -----
        In the one male and two females family unit, the number of adult
        females is normalized to two. Thus to compute shares one needs only
        to multiply by 0.5.

        """
        return 0.5 * self.number_G_female_adults

    @property
    def share_G_female_children(self):
        r"""
        Share of female children carrying the `G` allele of the :math:`\gamma`
        gene.

        :getter: Return the share of female children carrying the `G` allele.
        :type: pandas.Series

        """
        return self.number_G_female_children / self.number_children

    @property
    def share_g_female_adults(self):
        r"""
        Share of female adults carrying the `g` allele of the :math:`\gamma`
        gene.

        :getter: Return the share of female children carrying the `g` allele.
        :type: pandas.Series

        Notes
        -----
        In the one male and two females family unit, the number of adult
        females is normalized to two. Thus to compute shares one needs only
        to multiply by 0.5.

        """
        return 0.5 * self.number_g_female_adults

    @property
    def share_g_female_children(self):
        r"""
        Share of female children carrying the `g` allele of the :math:`\gamma`
        gene.

        :getter: Return the share of female children carrying the `g` allele.
        :type: pandas.Series

        """
        return self.number_g_female_children / self.number_children

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
    def share_GA_female_children(self):
        r"""
        Share of female children carrying the `GA` genotype.

        :getter: Return the share of female children carrying the `GA` genotype.
        :type: pandas.Series

        """
        return self.number_GA_female_children / self.number_children

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
    def share_Ga_female_children(self):
        r"""
        Share of female children carrying the `Ga` genotype.

        :getter: Return the share of female children carrying the `Ga` genotype.
        :type: pandas.Series

        """
        return self.number_Ga_female_children / self.number_children

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
    def share_gA_female_children(self):
        r"""
        Share of female children carrying the `gA` genotype.

        :getter: Return the share of female children carrying the `gA` genotype.
        :type: pandas.Series

        """
        return self.number_gA_female_children / self.number_children

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
    def share_ga_female_children(self):
        r"""
        Share of female children carrying the `ga` genotype.

        :getter: Return the share of female children carrying the `ga` genotype.
        :type: pandas.Series

        """
        return self.number_ga_female_children / self.number_children

    def compute_distribution(self, dataframe):
        """Compute distributions of various family configurations."""
        family_distributions = []
        for config in self.family.configurations:
            self.family.male_genotype = config[0]
            self.family.female_genotypes = config[1:]

            tmp_dist = dataframe.apply(self.family.compute_size, axis=1,
                                       raw=True)
            family_distributions.append(tmp_dist)

        # want to return a properly formated pandas df
        df = pd.concat(family_distributions, axis=1)
        df.columns = self.family.configurations

        return df.T

    def plot_adult_female_genotypes(self, axis, share=False):
        """Plot the timepaths for individual adult female genotypes."""
        axis.set_title('Adult females', fontsize=20, family='serif')
        kwargs = {'marker': '.', 'linestyle': 'none', 'legend': False,
                  'ax': axis, 'alpha': 0.5}
        if not share:
            self.number_GA_female_adults.plot(label='$GA$', **kwargs)
            self.number_Ga_female_adults.plot(label='$Ga$', **kwargs)
            self.number_gA_female_adults.plot(label='$gA$', **kwargs)
            self.number_ga_female_adults.plot(label='$ga$', **kwargs)
        else:
            self.share_GA_female_adults.plot(label='$GA$', **kwargs)
            self.share_Ga_female_adults.plot(label='$Ga$', **kwargs)
            self.share_gA_female_adults.plot(label='$gA$', **kwargs)
            self.share_ga_female_adults.plot(label='$ga$', **kwargs)
            axis.set_ylim(0, 1)

        return axis

    def plot_adult_female_alpha_alleles(self, axis, share=False):
        """Plot the timepaths for female adult alpha alleles."""
        axis.set_title(r'Adult females ($\alpha$ alleles)', fontsize=20,
                       family='serif')
        kwargs = {'marker': '.', 'linestyle': 'none', 'legend': False,
                  'ax': axis, 'alpha': 0.5}
        if not share:
            self.number_A_female_adults.plot(label='$A$', **kwargs)
            self.number_a_female_adults.plot(label='$a$', **kwargs)
        else:
            self.share_A_female_adults.plot(label='$A$', **kwargs)
            self.share_a_female_adults.plot(label='$a$', **kwargs)
            axis.set_ylim(0, 1)

        return axis

    def plot_adult_female_gamma_alleles(self, axis, share=False):
        """Plot the timepaths for female adult gamma alleles."""
        axis.set_title('Adult females ($\gamma$ alleles)', fontsize=20,
                       family='serif')
        kwargs = {'marker': '.', 'linestyle': 'none', 'legend': False,
                  'ax': axis, 'alpha': 0.5}
        if not share:
            self.number_G_female_adults.plot(label='$G$', **kwargs)
            self.number_g_female_adults.plot(label='$g$', **kwargs)
        else:
            self.share_G_female_adults.plot(label='$G$', **kwargs)
            self.share_g_female_adults.plot(label='$g$', **kwargs)
            axis.set_ylim(0, 1)

        return axis

    def plot_adult_male_genotypes(self, axis):
        """Plot the timepaths for male adult genotypes."""
        axis.set_title('Adult males', fontsize=20, family='serif')
        kwargs = {'marker': '.', 'linestyle': 'none', 'legend': False,
                  'ax': axis, 'alpha': 0.5}
        self.number_GA_male_adults.plot(label='$GA$', **kwargs)
        self.number_Ga_male_adults.plot(label='$Ga$', **kwargs)
        self.number_gA_male_adults.plot(label='$gA$', **kwargs)
        self.number_ga_male_adults.plot(label='$ga$', **kwargs)

        return axis

    def plot_adult_male_alpha_alleles(self, axis):
        """Plot the timepaths for male adult alpha alleles."""
        axis.set_title(r'Adult males ($\alpha$ alleles)', fontsize=20,
                       family='serif')
        kwargs = {'marker': '.', 'linestyle': 'none', 'legend': False,
                  'ax': axis, 'alpha': 0.5}
        self.number_A_male_adults.plot(label='$A$', **kwargs)
        self.number_a_male_adults.plot(label='$a$', **kwargs)

        return axis

    def plot_adult_male_gamma_alleles(self, axis):
        """Plot the timepaths for male adult gamma alleles."""
        axis.set_title('Adult males ($\gamma$ alleles)', fontsize=20,
                       family='serif')
        kwargs = {'marker': '.', 'linestyle': 'none', 'legend': False,
                  'ax': axis, 'alpha': 0.5}
        self.number_G_male_adults.plot(label='$G$', **kwargs)
        self.number_g_male_adults.plot(label='$g$', **kwargs)

        return axis

    def plot_adult_female_simulation(self, axis, kind='genotypes', share=False):
        """Plot simulation results for female offspring."""
        if kind == 'genotypes':
            self.plot_adult_female_genotypes(axis, share)
        elif kind == 'alpha_allele':
            self.plot_adult_female_alpha_alleles(axis, share)
        elif kind == 'gamma_allele':
            self.plot_adult_female_gamma_alleles(axis, share)
        else:
            raise ValueError

        axis.set_xlabel('Time', fontsize=15, family='serif')
        axis.legend(loc=0, frameon=False)

        return axis

    def plot_adult_male_simulation(self, axis, kind='genotypes'):
        """Plot simulation results for male adults."""
        if kind == 'genotypes':
            self.plot_adult_male_genotypes(axis)
        elif kind == 'alpha_allele':
            self.plot_adult_male_alpha_alleles(axis)
        elif kind == 'gamma_allele':
            self.plot_adult_male_gamma_alleles(axis)
        else:
            raise ValueError

        axis.set_xlabel('Time', fontsize=15, family='serif')
        axis.legend(loc=0, frameon=False)

        return axis

    def plot_alpha_sexual_selection_pressure(self, axis):
        """
        Plot female and male contributions to sexual selection pressure on the
        alpha gene.

        """
        kwargs = {'ax': axis, 'linestyle': 'none', 'marker': '.', 'alpha': 0.5}
        self.alpha_sexual_selection_pressure_males.plot(label='Males',
                                                        color='red',
                                                        **kwargs)
        self.alpha_sexual_selection_pressure_females.plot(label='Females',
                                                          color='blue',
                                                          **kwargs)
        self.alpha_sexual_selection_pressure.plot(label='Total',
                                                  color='purple',
                                                  **kwargs)
        return axis

    def plot_alpha_natural_selection_pressure(self, axis):
        """
        Plot female and male contributions to natural selection pressure on the
        alpha gene.

        """
        kwargs = {'ax': axis, 'linestyle': 'none', 'marker': '.', 'alpha': 0.5}
        self.alpha_natural_selection_pressure_males.plot(label='Males',
                                                         color='red',
                                                         **kwargs)
        self.alpha_natural_selection_pressure_females.plot(label='Females',
                                                           color='blue',
                                                           **kwargs)
        self.alpha_natural_selection_pressure.plot(label='Total',
                                                   color='purple',
                                                   **kwargs)
        return axis

    def plot_gamma_sexual_selection_pressure(self, axis):
        """
        Plot female and male contributions to sexual selection pressure on the
        gamma gene.

        """
        kwargs = {'ax': axis, 'linestyle': 'none', 'marker': '.', 'alpha': 0.5}
        self.gamma_sexual_selection_pressure_males.plot(label='Males',
                                                        color='red',
                                                        **kwargs)
        self.gamma_sexual_selection_pressure_females.plot(label='Females',
                                                          color='blue',
                                                          **kwargs)
        self.gamma_sexual_selection_pressure.plot(label='Total',
                                                  color='purple',
                                                  **kwargs)
        return axis

    def plot_gamma_natural_selection_pressure(self, axis):
        """
        Plot female and male contributions to natural selection pressure on the
        gamma gene.

        """
        kwargs = {'ax': axis, 'linestyle': 'none', 'marker': '.', 'alpha': 0.5}
        self.gamma_natural_selection_pressure_males.plot(label='Males',
                                                         color='red',
                                                         **kwargs)
        self.gamma_natural_selection_pressure_females.plot(label='Females',
                                                           color='blue',
                                                           **kwargs)
        self.gamma_natural_selection_pressure.plot(label='Total',
                                                   color='purple',
                                                   **kwargs)
        return axis

    def plot_alpha_selection_pressure(self, axis):
        """Plot measures of alpha natural and sexual selection pressure."""
        kwargs = {'ax': axis, 'linestyle': 'none', 'marker': '.', 'alpha': 0.5}
        self.alpha_natural_selection_pressure.plot(label='Natural',
                                                   color='red',
                                                   **kwargs)
        self.alpha_sexual_selection_pressure.plot(label='Sexual',
                                                  color='blue',
                                                  **kwargs)
        self.alpha_selection_pressure.plot(label='Total',
                                           color='purple',
                                           **kwargs)

        return axis

    def plot_gamma_selection_pressure(self, axis):
        """Plot measures of gamma natural and sexual selection pressure."""
        kwargs = {'ax': axis, 'linestyle': 'none', 'marker': '.', 'alpha': 0.5}
        self.gamma_natural_selection_pressure.plot(label='Natural',
                                                   color='red',
                                                   **kwargs)
        self.gamma_sexual_selection_pressure.plot(label='Sexual',
                                                  color='blue',
                                                  **kwargs)
        self.gamma_selection_pressure.plot(label='Total',
                                           color='purple',
                                           **kwargs)

        return axis


def plot_selection_pressure(simulator, mGA0, T=None, rtol=None, **params):
    """Plot measures of selection pressure on the alpha and gamma genes."""
    # simulate the model
    simulator.family.params = params
    simulator.initial_condition = mGA0
    simulation = simulator.simulate(rtol, T)
    distribution = Distribution(simulator.family, simulation)

    fig, axes = plt.subplots(3, 2, figsize=(12, 18), sharey=True)

    distribution.plot_alpha_natural_selection_pressure(axes[0, 0])
    distribution.plot_gamma_natural_selection_pressure(axes[0, 1])
    distribution.plot_alpha_sexual_selection_pressure(axes[1, 0])
    distribution.plot_gamma_sexual_selection_pressure(axes[1, 1])
    distribution.plot_alpha_selection_pressure(axes[2, 0])
    distribution.plot_gamma_selection_pressure(axes[2, 1])

    # plot titles
    axes[0, 0].set_title(r'Natural selection, $\alpha$', family='serif',
                         fontsize=20)
    axes[0, 1].set_title(r'Natural selection, $\gamma$', family='serif',
                         fontsize=20)
    axes[1, 0].set_title(r'Sexual selection, $\alpha$', family='serif',
                         fontsize=20)
    axes[1, 1].set_title(r'Sexual selection, $\gamma$', family='serif',
                         fontsize=20)
    axes[2, 0].set_title(r'Combined selection, $\alpha$', family='serif',
                         fontsize=20)
    axes[2, 1].set_title(r'Combined selection, $\gamma$', family='serif',
                         fontsize=20)

    # add axis specific subplot options
    for i in range(3):
        for j in range(2):
            axes[i, j].set_xlabel('Time', fontsize=15, family='serif')
            axes[i, j].axhline(y=0, color='black')
            axes[i, j].legend(loc=0, frameon=False, prop={'family': 'serif'})

    # figure title
    title = (r'Selection pressure on $\alpha$ and $\gamma$ genes when' +
             '\n$M^{{GA}}(0)={0},\ e={e},\ \Pi^{{aA}}={PiaA},\ ' +
             '\Pi^{{AA}}={PiAA},\ \Pi^{{aa}}={Piaa},\ \Pi^{{Aa}}={PiAa}$')
    fig.suptitle(title.format(mGA0, **params), x=0.5, family='serif',
                 fontsize=25)

    return [fig, axes]


def plot_isolated_subpopulations_simulation(simulator, mGA0, T=None, rtol=None,
                                            males='genotypes', females='genotypes',
                                            share=False, **params):
    """
    Plot a simulated trajectory given an initial condition consistent with the
    isolated sub-populations assumption.

    Parameters
    ----------
    simulator : simulators.Simulator
    mGA0 : float
    T : int (default=None)
        The number of time steps to simulate.
    rtol : float (default=None)
        Simulate the model until the relative difference between timesteps
        is sufficiently small.
    males : str (default='genotypes')
        Which of 'genotypes', 'alpha_allele' or 'gamma_allele' do you wish to
        plot for adult males.
    females : str
        Which of 'genotypes', 'alpha_allele' or 'gamma_allele' do you wish to
        plot for female offspring.
    share : boolean (default=False)
        Flag indicating whether you wish to plot share or number of females.
    params : dict
        Dictionary of parameter values

    Returns
    -------
    A list containing...
        fig :
        axes : list

    """
    # simulate the model
    simulator.family.params = params
    simulator.initial_condition = mGA0
    simulation = simulator.simulate(rtol, T)

    # compute distributions
    distribution = Distribution(simulator.family, simulation)

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
    distribution.plot_adult_female_simulation(axes[0], females, share)
    distribution.plot_adult_male_simulation(axes[1], males)

    # figure title
    if not share:
        fig_title = ('Numbers when\n$^M{{GA}}(0)={0}$, $e={e}$, ' +
                     r'$\Pi^{{aA}}={PiaA},\ \Pi^{{AA}}={PiAA},\ ' +
                     r'\Pi^{{aa}}={Piaa},\ \Pi^{{Aa}}={PiAa}$')
    else:
        fig_title = ('Population shares when\n$M^{{GA}}(0)={0}$, $e={e}$, ' +
                     r'$\Pi^{{aA}}={PiaA},\ \Pi^{{AA}}={PiAA},\ ' +
                     r'\Pi^{{aa}}={Piaa},\ \Pi^{{Aa}}={PiAa}$')

    fig.suptitle(fig_title.format(mGA0, **params), x=0.5, y=1.05, fontsize=25,
                 family='serif')

    return [fig, axes]
