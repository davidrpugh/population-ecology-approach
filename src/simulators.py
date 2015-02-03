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

        # initial number female offspring
        fGA0 = self.family.params['c'] * self.family.params['PiAA'] * mGA
        fga0 = self.family.params['c'] * self.family.params['Piaa'] * mga
        initial_number_females = np.array([fGA0, 0.0, 0.0, fga0])

        initial_condition = np.hstack((initial_male_shares,
                                       initial_number_females))

        self._initial_condition = initial_condition


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
        avg_A_female_offspring = (self.number_A_female_offspring /
                                  self.number_A_female_adults)
        avg_a_female_offspring = (self.number_a_female_offspring /
                                  self.number_a_female_adults)
        return np.log(avg_A_female_offspring) - np.log(avg_a_female_offspring)

    @property
    def alpha_natural_selection_pressure_males(self):
        """
        Measure of the component of natural selection pressure on the alpha
        gene coming from the males.

        :getter: Return the current time series for natural selection pressure.
        :type: pandas.Series

        """
        A_ratio = (self.number_A_male_adults.shift(-1) /
                   self.number_A_male_offspring)
        a_ratio = (self.number_a_male_adults.shift(-1) /
                   self.number_a_male_offspring)
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
                   self.number_A_female_offspring)
        a_ratio = (self.number_a_female_adults.shift(-1) /
                   self.number_a_female_offspring)
        return np.log(A_ratio) - np.log(a_ratio)

    @property
    def alpha_sexual_selection_pressure_males(self):
        """
        Measure of the component of sexual selection pressure on the alpha
        gene from males.

        :getter: Return the current time series for sexual selection pressure.
        :type: pandas.Series

        """
        avg_A_male_offspring = (self.number_A_male_offspring /
                                self.number_A_male_adults)
        avg_a_male_offspring = (self.number_a_male_offspring /
                                self.number_a_male_adults)
        return np.log(avg_A_male_offspring) - np.log(avg_a_male_offspring)

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
        avg_G_female_offspring = (self.number_G_female_offspring /
                                  self.number_G_female_adults)
        avg_g_female_offspring = (self.number_g_female_offspring /
                                  self.number_g_female_adults)
        return np.log(avg_G_female_offspring) - np.log(avg_g_female_offspring)

    @property
    def gamma_natural_selection_pressure_males(self):
        """
        Measure of the component of natural selection pressure on the gamma
        gene coming from the males.

        :getter: Return the current time series for natural selection pressure.
        :type: pandas.Series

        """
        G_ratio = (self.number_G_male_adults.shift(-1) /
                   self.number_G_male_offspring)
        g_ratio = (self.number_g_male_adults.shift(-1) /
                   self.number_g_male_offspring)
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
                   self.number_G_female_offspring)
        g_ratio = (self.number_g_female_adults.shift(-1) /
                   self.number_g_female_offspring)
        return np.log(G_ratio) - np.log(g_ratio)

    @property
    def gamma_sexual_selection_pressure_males(self):
        """
        Measure of the component of sexual selection pressure on the gamma
        gene from males.

        :getter: Return the current time series for sexual selection pressure.
        :type: pandas.Series

        """
        avg_G_male_offspring = (self.number_G_male_offspring /
                                self.number_G_male_adults)
        avg_g_male_offspring = (self.number_g_male_offspring /
                                self.number_g_male_adults)
        return np.log(avg_G_male_offspring) - np.log(avg_g_male_offspring)

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
             '\n$M^{{GA}}(0)={0},\ \Pi^{{aA}}={PiaA},\ ' +
             '\Pi^{{AA}}={PiAA},\ \Pi^{{aa}}={Piaa},\ \Pi^{{Aa}}={PiAa}$')
    fig.suptitle(title.format(mGA0, **params), x=0.5, family='serif',
                 fontsize=25)

    return [fig, axes]


def plot_simulation(simulator, mGA0, T, males='genotypes', females='genotypes',
                    share=False, **params):
    """
    Plot a model simulation.

    Parameters
    ----------
    simulator : simulators.Simulator
    mGA0 : float
    T : int
        The number of time steps to simulate.
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 8), sharey=True)
    distribution.plot_adult_female_simulation(axes[0], females, share)
    distribution.plot_offspring_female_simulation(axes[1], females, share)
    distribution.plot_adult_male_simulation(axes[2], males)

    # figure title
    if not share:
        fig_title = ('Numbers when\n$M^{{GA}}(0)={0}$,' +
                     r'$\Pi^{{aA}}={PiaA},\ \Pi^{{AA}}={PiAA},\ ' +
                     r'\Pi^{{aa}}={Piaa},\ \Pi^{{Aa}}={PiAa}$')
    else:
        fig_title = ('Population shares when\n$M^{{GA}}(0)={0}$,' +
                     r'$\Pi^{{aA}}={PiaA},\ \Pi^{{AA}}={PiAA},\ ' +
                     r'\Pi^{{aa}}={Piaa},\ \Pi^{{Aa}}={PiAa}$')

    fig.suptitle(fig_title.format(mGA0, **params), x=0.5, y=1.05, fontsize=25,
                 family='serif')

    return [fig, axes]
