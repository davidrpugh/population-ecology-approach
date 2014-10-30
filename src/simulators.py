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
        if self.__distribution is None:
            self.__distribution = self.compute_distribution(self.simulation)
        return self.__distribution

    @property
    def alpha_natural_selection_pressure(self):
        pressure = (self.alpha_natural_selection_pressure_females +
                    self.alpha_natural_selection_pressure_males)
        return pressure

    @property
    def alpha_natural_selection_pressure_females(self):
        avg_A_female_children = (self.number_A_female_children /
                                 self.number_A_female_adults)
        avg_a_female_children = (self.number_a_female_children /
                                 self.number_a_female_adults)
        return np.log(avg_A_female_children) - np.log(avg_a_female_children)

    @property
    def alpha_natural_selection_pressure_males(self):
        A_ratio = (self.number_A_male_adults.shift(-1) /
                   self.number_A_male_children)
        a_ratio = (self.number_a_male_adults.shift(-1) /
                   self.number_a_male_children)
        return np.log(A_ratio) - np.log(a_ratio)

    @property
    def alpha_sexual_selection_pressure(self):
        pressure = (self.alpha_sexual_selection_pressure_females +
                    self.alpha_sexual_selection_pressure_males)
        return pressure

    @property
    def alpha_sexual_selection_pressure_females(self):
        A_ratio = (self.number_A_female_adults.shift(-1) /
                   self.number_A_female_children)
        a_ratio = (self.number_a_female_adults.shift(-1) /
                   self.number_a_female_children)
        return np.log(A_ratio) - np.log(a_ratio)

    @property
    def alpha_sexual_selection_pressure_males(self):
        avg_A_male_children = (self.number_A_male_children /
                               self.number_A_male_adults)
        avg_a_male_children = (self.number_a_male_children /
                               self.number_a_male_adults)
        return np.log(avg_A_male_children) - np.log(avg_a_male_children)

    @property
    def alpha_selection_pressure(self):
        total_pressure = (self.alpha_natural_selection_pressure +
                          self.alpha_sexual_selection_pressure)
        return total_pressure

    @property
    def alpha_sexual_selection_pressure_females(self):
        A_ratio = (self.number_A_female_adults.shift(-1) /
                   self.number_A_female_children)
        a_ratio = (self.number_a_female_adults.shift(-1) /
                   self.number_a_female_children)
        return np.log(A_ratio) - np.log(a_ratio)

    @property
    def gamma_natural_selection_pressure(self):
        pressure = (self.gamma_natural_selection_pressure_females +
                    self.gamma_natural_selection_pressure_males)
        return pressure

    @property
    def gamma_natural_selection_pressure_females(self):
        avg_G_female_children = (self.number_G_female_children /
                                 self.number_G_female_adults)
        avg_g_female_children = (self.number_g_female_children /
                                 self.number_g_female_adults)
        return np.log(avg_G_female_children) - np.log(avg_g_female_children)

    @property
    def gamma_natural_selection_pressure_males(self):
        G_ratio = (self.number_G_male_adults.shift(-1) /
                   self.number_G_male_children)
        g_ratio = (self.number_g_male_adults.shift(-1) /
                   self.number_g_male_children)
        return np.log(G_ratio) - np.log(g_ratio)

    @property
    def gamma_sexual_selection_pressure(self):
        pressure = (self.gamma_sexual_selection_pressure_females +
                    self.gamma_sexual_selection_pressure_males)
        return pressure

    @property
    def gamma_sexual_selection_pressure_females(self):
        G_ratio = (self.number_G_female_adults.shift(-1) /
                   self.number_G_female_children)
        g_ratio = (self.number_g_female_adults.shift(-1) /
                   self.number_g_female_children)
        return np.log(G_ratio) - np.log(g_ratio)

    @property
    def gamma_sexual_selection_pressure_males(self):
        avg_G_male_children = (self.number_G_male_children /
                               self.number_G_male_adults)
        avg_g_male_children = (self.number_g_male_children /
                               self.number_g_male_adults)
        return np.log(avg_G_male_children) - np.log(avg_g_male_children)

    @property
    def gamma_selection_pressure(self):
        total_pressure = (self.gamma_sexual_selection_pressure +
                          self.gamma_natural_selection_pressure)
        return total_pressure

    @property
    def number_A_female_adults(self):
        A_female_adults = (self.distribution.xs(0, level='female1_genotype') +
                           self.distribution.xs(2, level='female1_genotype') +
                           self.distribution.xs(0, level='female2_genotype') +
                           self.distribution.xs(2, level='female2_genotype'))
        return A_female_adults.sum(axis=0)

    @property
    def number_A_female_children(self):
        A_female_children = self.simulation['Female Children Genotypes'][[0, 2]]
        return A_female_children.sum(axis=1)

    @property
    def number_A_male_children(self):
        return self.number_A_female_children

    @property
    def number_a_female_adults(self):
        a_female_adults = (self.distribution.xs(1, level='female1_genotype') +
                           self.distribution.xs(3, level='female1_genotype') +
                           self.distribution.xs(1, level='female2_genotype') +
                           self.distribution.xs(3, level='female2_genotype'))
        return a_female_adults.sum(axis=0)

    @property
    def number_a_female_children(self):
        a_female_children = self.simulation['Female Children Genotypes'][[1, 3]]
        return a_female_children.sum(axis=1)

    @property
    def number_a_male_children(self):
        return self.number_a_female_children

    @property
    def number_female_children(self):
        return self.simulation['Female Children Genotypes'][[0, 1, 2, 3]]

    @property
    def number_G_female_adults(self):
        G_female_adults = (self.distribution.xs(0, level='female1_genotype') +
                           self.distribution.xs(1, level='female1_genotype') +
                           self.distribution.xs(0, level='female2_genotype') +
                           self.distribution.xs(1, level='female2_genotype'))
        return G_female_adults.sum(axis=0)

    @property
    def number_G_female_children(self):
        G_female_children = self.simulation['Female Children Genotypes'][[0, 1]]
        return G_female_children.sum(axis=1)

    @property
    def number_G_male_adults(self):
        G_male_adults = self.simulation['Male Adult Genotypes'][[0, 1]]
        return G_male_adults.sum(axis=1)

    @property
    def number_G_male_children(self):
        """

        Notes
        -----
        By construction, the number of male chidlren carrying the G allele of
        the gamme gene is the same as the number of female children carrying
        that allele.

        """
        return self.number_G_female_children

    @property
    def number_g_female_adults(self):
        g_female_adults = (self.distribution.xs(2, level='female1_genotype') +
                           self.distribution.xs(3, level='female1_genotype') +
                           self.distribution.xs(2, level='female2_genotype') +
                           self.distribution.xs(3, level='female2_genotype'))
        return g_female_adults.sum(axis=0)

    @property
    def number_g_female_children(self):
        g_female_children = self.simulation['Female Children Genotypes'][[2, 3]]
        return g_female_children.sum(axis=1)

    @property
    def number_g_male_adults(self):
        g_male_adults = self.simulation['Male Adult Genotypes'][[2, 3]]
        return g_male_adults.sum(axis=1)

    @property
    def number_g_male_children(self):
        """

        Notes
        -----
        By construction, the number of male chidlren carrying the g allele of
        the gamme gene is the same as the number of female children carrying
        that allele.

        """
        return self.number_g_female_children

    @property
    def number_GA_female_adults(self):
        GA_female_adults = (self.distribution.xs(0, level='female1_genotype') +
                            self.distribution.xs(0, level='female2_genotype'))
        return GA_female_adults.sum(axis=0)

    @property
    def number_GA_female_children(self):
        return self.simulation['Female Children Genotypes'][0]

    @property
    def number_Ga_female_adults(self):
        Ga_female_adults = (self.distribution.xs(1, level='female1_genotype') +
                            self.distribution.xs(1, level='female2_genotype'))
        return Ga_female_adults.sum(axis=0)

    @property
    def number_Ga_female_children(self):
        return self.simulation['Female Children Genotypes'][1]

    @property
    def number_gA_female_adults(self):
        gA_female_adults = (self.distribution.xs(2, level='female1_genotype') +
                            self.distribution.xs(2, level='female2_genotype'))
        return gA_female_adults.sum(axis=0)

    @property
    def number_gA_female_children(self):
        return self.simulation['Female Children Genotypes'][2]

    @property
    def number_ga_female_adults(self):
        ga_female_adults = (self.distribution.xs(3, level='female1_genotype') +
                            self.distribution.xs(3, level='female2_genotype'))
        return ga_female_adults.sum(axis=0)

    @property
    def number_ga_female_children(self):
        return self.simulation['Female Children Genotypes'][3]

    @property
    def number_A_male_adults(self):
        A_male_adults = self.simulation['Male Adult Genotypes'][[0, 2]]
        return A_male_adults.sum(axis=1)

    @property
    def number_a_male_adults(self):
        a_male_adults = self.simulation['Male Adult Genotypes'][[1, 3]]
        return a_male_adults.sum(axis=1)

    @property
    def number_G_male_adults(self):
        G_male_adults = self.simulation['Male Adult Genotypes'][[0, 1]]
        return G_male_adults.sum(axis=1)

    @property
    def number_g_male_adults(self):
        g_male_adults = self.simulation['Male Adult Genotypes'][[2, 3]]
        return g_male_adults.sum(axis=1)

    @property
    def number_GA_male_adults(self):
        return self.simulation['Male Adult Genotypes'][0]

    @property
    def number_Ga_male_adults(self):
        return self.simulation['Male Adult Genotypes'][1]

    @property
    def number_gA_male_adults(self):
        return self.simulation['Male Adult Genotypes'][2]

    @property
    def number_ga_male_adults(self):
        return self.simulation['Male Adult Genotypes'][3]

    @property
    def share_A_female_adults(self):
        return 0.5 * self.number_A_female_adults

    @property
    def share_A_female_children(self):
        return self.number_A_female_children / self.number_female_children

    @property
    def share_a_female_adults(self):
        return 0.5 * self.number_a_female_adults

    @property
    def share_a_female_children(self):
        return self.number_a_female_children / self.number_female_children

    @property
    def share_G_female_adults(self):
        return 0.5 * self.number_G_female_adults

    @property
    def share_G_female_children(self):
        return self.number_G_female_children / self.number_female_children

    @property
    def share_g_female_adults(self):
        return 0.5 * self.number_g_female_adults

    @property
    def share_g_female_children(self):
        return self.number_g_female_children / self.number_female_children

    @property
    def share_GA_female_adults(self):
        return 0.5 * self.number_GA_female_adults

    @property
    def share_GA_female_children(self):
        return self.number_GA_female_children / self.number_female_children

    @property
    def share_Ga_female_adults(self):
        return 0.5 * self.number_Ga_female_adults

    @property
    def share_Ga_female_children(self):
        return self.number_Ga_female_children / self.number_female_children

    @property
    def share_gA_female_adults(self):
        return 0.5 * self.number_gA_female_adults

    @property
    def share_gA_female_children(self):
        return self.number_gA_female_children / self.number_female_children

    @property
    def share_ga_female_adults(self):
        return 0.5 * self.number_ga_female_adults

    @property
    def share_ga_female_children(self):
        return self.number_ga_female_children / self.number_female_children

    def compute_distribution(self, dataframe):
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


def _plot_genotypes(axis, dataframe):
    """Plot the timepaths of individual genotypes."""
    dataframe.plot(marker='.', linestyle='none', legend=False, ax=axis,
                   alpha=0.5)
    return axis


def _plot_alpha_alleles(axis, dataframe):
    """Plot the timepaths of alpha alleles."""
    altruistic_agents = dataframe[[0, 2]].sum(axis=1)
    selfish_agents = dataframe[[1, 3]].sum(axis=1)

    altruistic_agents.plot(marker='.', linestyle='none', legend=False, ax=axis,
                           alpha=0.5)
    selfish_agents.plot(marker='.', linestyle='none', ax=axis, legend=False,
                        alpha=0.5)

    return axis


def _plot_gamma_alleles(axis, dataframe):
    """Plot the timepaths of alpha alleles."""
    Gatekeepers = dataframe[[0, 1]].sum(axis=1)
    gatekeepers = dataframe[[2, 3]].sum(axis=1)

    Gatekeepers.plot(marker='.', linestyle='none', ax=axis, alpha=0.5,
                     legend=False)
    gatekeepers.plot(marker='.', linestyle='none', ax=axis, alpha=0.5,
                     legend=False)

    return axis


def _plot_trajectories(ax, dataframe, kind='genotypes'):
    """Plot the timepaths of genotypes or alleles depending."""
    if kind == 'genotypes':
        female_lines = _plot_genotypes(ax, dataframe)
    elif kind == 'alpha_allele':
        female_lines = _plot_alpha_alleles(ax, dataframe)
    elif kind == 'gamma_allele':
        female_lines = _plot_gamma_alleles(ax, dataframe)
    else:
        raise ValueError
    return female_lines


def plot_isolated_subpopulations_simulation(simulator, mGA0, T=None, rtol=None,
                                            males='genotypes', females='genotypes',
                                            **params):
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

    Returns
    -------
    A list containing...

    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))

    simulator.family.params = params
    simulator.initial_condition = mGA0
    df = simulator.simulate(rtol, T)

    # draw the lines
    male_lines = _plot_trajectories(axes[0], df['Male Adult Genotypes'],
                                    kind=males)
    female_lines = _plot_trajectories(axes[1], df['Female Children Genotypes'],
                                      kind=females)

    # specify plot options
    axes[0].set_xlabel('Time', fontsize=15, family='serif')
    axes[1].set_xlabel('Time', fontsize=15, family='serif')
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Male adults', family='serif', fontsize=20)
    axes[0].grid('on')

    axes[1].grid('on')
    axes[1].set_title('Female children', family='serif', fontsize=20)

    # correct labels depend on what it being plotted!
    if males == 'genotypes':
        male_labels = ['$GA$', '$Ga$', '$gA$', '$ga$']
    elif males == 'alpha_allele':
        male_labels = ['$A$', '$a$']
    else:
        male_labels = ['$G$', '$g$']

    axes[0].legend(male_lines, labels=male_labels,
                   loc=0, frameon=False)

    if females == 'genotypes':
        female_labels = ['$GA$', '$Ga$', '$gA$', '$ga$']
    elif females == 'alpha_allele':
        female_labels = ['$A$', '$a$']
    else:
        female_labels = ['$G$', '$g$']

    axes[1].legend(female_lines, labels=female_labels,
                   loc=0, frameon=False)

    fig.suptitle('Number of males and females',
                 x=0.5, fontsize=25, family='serif')

    return [fig, axes, male_lines, female_lines]
