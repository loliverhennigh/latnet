#!/usr/bin/env python


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from channel import ChannelDomain

sys.path.append('../../latnet')
from latnetwork import TrainLatNet
from archs.standard_arch import StandardArch
from controller import LatNetController

class StandardTrainer(TrainLatNet, StandardArch):
  script_name = __file__
  domains = [ChannelDomain]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'train_sim_dir': './train_data',
        'latnet_network_dir': './network_save',
        'dataset': 'sailfish',
        'visc': 0.01,
        'lb_to_ln': 128,
        'seq_length': 3,
        'input_cshape': '16x16',
        'max_sim_iters': 100})

  def __init__(self, config):
    super(StandardTrainer, self).__init__(config)

if __name__ == '__main__':
  sim = LatNetController(trainer=StandardTrainer)
  sim.run()

