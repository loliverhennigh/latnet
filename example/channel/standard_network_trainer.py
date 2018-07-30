#!/usr/bin/env python


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from channel import TrainChannelDomain

sys.path.append('../../latnet')
from latnetwork import TrainLatNet
from archs.standard_sailfish_arch import StandardSailfishArch
from controller import LatNetController

class StandardTrainer(TrainLatNet, StandardSailfishArch):
  script_name = __file__
  domains = [TrainChannelDomain]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'dataset': 'sailfish',
        'visc': 0.01,
        'lb_to_ln': 128,
        'seq_length': 5,
        'filter_size': 17,
        'train_cshape': '32x32'})

  def __init__(self, config):
    super(StandardTrainer, self).__init__(config)

if __name__ == '__main__':
  sim = LatNetController(trainer=StandardTrainer)
  sim.run()

