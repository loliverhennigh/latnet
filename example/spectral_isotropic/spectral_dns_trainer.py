#!/usr/bin/env python

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import latnet
sys.path.append('../../latnet')
from latnetwork import TrainLatNet
from domain import TrainSpectralDNSDomain
from archs.standard_jhtdb_arch import StandardJHTDBArch
from controller import LatNetController


class FakeDomain(TrainSpectralDNSDomain):
  name = "SpectralDNS"
  num_simulations = 1
  periodic_x = True
  periodic_y = True
  periodic_z = True
  sim_shape = [128, 128, 128]
  #sim_shape = [32, 32, 32]
  #sim_shape = [64, 64, 64]
  #sim_shape = [48, 48, 48]

  def __init__(self, config, sim_dir):
    super(FakeDomain, self).__init__(config, sim_dir)

class JHTDBTrainer(TrainLatNet, StandardJHTDBArch):
  script_name = __file__
  domains = [FakeDomain]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'dataset': 'spectraldns',
        'lb_to_ln': 30,
        'train_cshape': '16x16x16',
        'seq_length': 1,
        'DxQy': 'D3Q4',
        'start_num_dps': 5000,
        'batch_size': 16})

  def __init__(self, config):
    super(JHTDBTrainer, self).__init__(config)

if __name__ == '__main__':
  sim = LatNetController(trainer=JHTDBTrainer)
  sim.run()

