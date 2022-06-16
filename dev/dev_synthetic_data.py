import sys
sys.path.append('./')

from urp import dataset

data_generator = dataset.CensoredSurvivalData()

sample = data_generator.sample(n=10000)

data_generator.plot_sample(n=10000)

pass