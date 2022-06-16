import sys
sys.path.append('./')

from urp import dataset

data_generator = dataset.CensoredSurvivalData()
data_generator.plot_sample(n=10000)


data_generator = dataset.CensoredSurvivalData(censoring_dist_scale=10)
data_generator.plot_sample(n=10000)


data_generator = dataset.CensoredSurvivalData(censoring_dist_min_scale=2)
data_generator.plot_sample(n=10000)

data_generator = dataset.CensoredSurvivalData(censoring_dist_maj_loc=2, censoring_dist_min_scale=8)
data_generator.plot_sample(n=10000)

pass