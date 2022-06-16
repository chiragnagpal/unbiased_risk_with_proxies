import numpy as np
import pandas as pd

from scipy import stats
import warnings


class CensoredSurvivalData:
	"""Class to specify the True Data Generating Process and sample from it."""

	def __init__(self, 
							 p=0.2,
							 event_dist_maj='Exponential',
							 event_dist_min=None,
							 censoring_dist_maj='Uniform',
							 censoring_dist_min=None, 
							 random_seed=0,
							 **kwargs):
		"""Initialize the True Data Generating Process."""

		self.random_seed = random_seed
		self.p = p

		# Set the Time-to-Event Distributions
		self.event_dist_maj = event_dist_maj
		if event_dist_min is not None:
			self.event_dist_min = event_dist_min
		else:
			warnings.warn('Time-to-Event Distribution not specified for the Minority Demographic.')
			self.event_dist_min = self.event_dist_maj

		# Set the Time-to-Censoring Distributions
		self.censoring_dist_maj = censoring_dist_maj
		# Set the Censoring Distributions
		if censoring_dist_min is not None:
			self.censoring_dist_min = censoring_dist_min
		else:
			warnings.warn('Censoring Distribution not specified for the Minority Demographic. Defaulting to majority distribution.')
			self.censoring_dist_min = self.censoring_dist_maj

		# Set the Time-to-Event Distributions
		event_dist_scale = kwargs.get('event_dist_shape', 1.)
		event_dist_min_scale = kwargs.get('event_dist_min_shape', event_dist_scale)
		event_dist_maj_scale = kwargs.get('event_dist_maj_shape', event_dist_scale)

		self.event_dist_min = self.gen_distribution(self.event_dist_min,
																								scale=event_dist_min_scale)
		self.event_dist_maj = self.gen_distribution(self.event_dist_maj,
																								scale=event_dist_maj_scale)

		# Set the Censoring Distributions
		censoring_dist_scale = kwargs.get('censoring_dist_scale', 5)
		censoring_dist_min_scale = kwargs.get('censoring_dist_min_scale', censoring_dist_scale)
		censoring_dist_maj_scale = kwargs.get('censoring_dist_maj_scale', censoring_dist_scale)

		self.censoring_dist_min = self.gen_distribution(self.censoring_dist_min,
																									  scale=censoring_dist_min_scale)
		self.censoring_dist_maj = self.gen_distribution(self.censoring_dist_maj,
																										scale=censoring_dist_maj_scale)

	def gen_distribution(self, distribution, **kwargs):
		"""Helper function to set a distribution."""
		if distribution == 'Exponential':
			return stats.expon(**kwargs)
		elif distribution == 'Uniform':
			return stats.uniform(**kwargs)
		else:
			raise NotImplementedError()

	def __call__(self, n=1000):

		print("Minority Demographic Distribution: {}".format(self.p))

		print("Majority Time-to-Event Distribution: {}".format(self.event_dist_maj))
		print("Minority Time-to-Event Distribution: {}".format(self.event_dist_min))

		print("Majority Censoring Distribution: {}".format(self.censoring_dist_maj))
		print("Minority Censoring Distribution: {}".format(self.censoring_dist_min))

	def sample(self, n=1, random_seed=None):
		"""Sample observations from the True Data Generating Process."""

		if random_seed is not None:
			np.random.seed(self.random_seed)

		is_minority = np.random.binomial(1, self.p, size=n).astype(bool)


		true_event_time_minority = self.event_dist_min.rvs(size=n)
		true_event_time_majority = self.event_dist_maj.rvs(size=n)

		true_censoring_time_minority = self.censoring_dist_min.rvs(size=n)
		true_censoring_time_majority = self.censoring_dist_maj.rvs(size=n)

		true_event_time = np.zeros_like(true_event_time_majority)
		true_event_time[is_minority] 	= true_event_time_minority[is_minority]
		true_event_time[~is_minority] = true_event_time_majority[~is_minority]

		true_censoring_time = np.zeros_like(true_censoring_time_majority)
		true_censoring_time[is_minority] 	= true_censoring_time_minority[is_minority]
		true_censoring_time[~is_minority] = true_censoring_time_majority[~is_minority]

		censored_survival_time = np.minimum(true_event_time, true_censoring_time)
		censoring_indicator = true_event_time <  true_censoring_time

		sample_data = pd.DataFrame({'is_minority': is_minority,
																'true_event_time': true_event_time,
																'true_censoring_time': true_censoring_time,
																'censoring_indicator': censoring_indicator,
																'censored_survival_time': censored_survival_time})

		return sample_data

	def plot_sample(self, n=1000):
		"""Plot a random sample from the True Data Generating Process."""

		# Sample from the True Data Generating Process
		sample = self.sample(n)
		
		from matplotlib import pyplot as plt
		
		x = np.linspace(0, 10, 1000)	

		plt.figure(figsize=(8*2,6*3))
		plt.subplots_adjust(wspace = 0.15, hspace = 0.25)

		plt.subplot(3, 2, 1)
		plt.hist(sample.true_event_time[~sample.is_minority],
						 bins=20, alpha=0.25, density=True)
		plt.plot(sorted(sample.true_event_time), 
						 self.event_dist_maj.pdf(sorted(sample.true_event_time)), lw=2, color='C0' )
	
		plt.title("True Time-to-Event (pdf)", fontsize=18)
		plt.xlabel("Time in Years", fontsize=16)
		plt.ylabel("Event Probability", fontsize=16)
		plt.legend(loc='best', frameon=False)

		plt.xlim(0, 10)
		plt.ylim(0, None)
		plt.grid(ls=':')

#		plt.show()

		plt.subplot(3, 2, 2)
		plt.hist(sample.true_event_time[~sample.is_minority],
						 bins=20, alpha=0.25, density=True, cumulative=True)
		plt.plot(x, self.event_dist_maj.sf(x), lw=2, color='C0')

		plt.title("True Time-to-Event (sf)", fontsize=18)
		plt.xlabel("Time in Years", fontsize=16)
		plt.ylabel("Event Survival Probability", fontsize=16)
		plt.legend(loc='best', frameon=False)

		plt.xlim(0, 10)
		plt.ylim(0, None)
		plt.grid(ls=':')

		plt.subplot(3, 2, 3)
		
		plt.hist(sample.true_censoring_time[~sample.is_minority],
						 bins=20, alpha=0.25, density=True)
		plt.plot(x,  self.censoring_dist_maj.pdf(x), lw=2, color='C0')

		plt.title("True Censoring Event (pdf)", fontsize=18)
		plt.xlabel("Time in Years", fontsize=16)
		plt.ylabel("Censoring Probability", fontsize=16)
		plt.legend(loc='best', frameon=False)

		plt.xlim(0, 10)
		plt.ylim(0, None)
		plt.grid(ls=':')

		plt.subplot(3, 2, 4)
		
		plt.hist(sample.true_censoring_time[~sample.is_minority],
						 bins=20, alpha=0.25, density=True, cumulative=True)
		plt.plot(x,  self.censoring_dist_maj.sf(x), lw=2, color='C0')

		plt.title("True Censoring Event (sf)", fontsize=18)
		plt.xlabel("Time in Years", fontsize=16)
		plt.ylabel("Censoring Probability", fontsize=16)
		plt.legend(loc='best', frameon=False)

		plt.xlim(0, 10)
		plt.ylim(0, None)
		plt.grid(ls=':')

		from lifelines import KaplanMeierFitter, NelsonAalenFitter

		plt.subplot(3, 2, 5)
		plt.title("Kaplan-Meier Survival Estimate", fontsize=18)
		KaplanMeierFitter().fit(sample.censored_survival_time,
													  sample.censoring_indicator).plot()
		plt.xlabel("Time in Years", fontsize=16)
		plt.ylim(0, None)
		plt.xlim(0, 10)
		plt.grid(ls=':')

		plt.subplot(3, 2, 6)
		plt.title("Nelson-Aalen Hazard Estimate", fontsize=18)
		KaplanMeierFitter().fit(sample.censored_survival_time,
													  sample.censoring_indicator).plot()
		plt.xlabel("Time in Years", fontsize=16)
		plt.ylim(0, None)
		plt.xlim(0, 10)
		plt.grid(ls=':')

		plt.show()
