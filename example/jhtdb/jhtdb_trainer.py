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
from domain import TrainJHTDBDomain
from archs.standard_jhtdb_arch import StandardJHTDBArch
from controller import LatNetController


class FakeDomain(TrainJHTDBDomain):
  name = "JHTDB"
  num_simulations = 8
  periodic_x = True
  periodic_y = True
  periodic_z = True
  sim_shape = [1024, 1024, 1024]

  def __init__(self, config, sim_dir):
    super(FakeDomain, self).__init__(config, sim_dir)

class JHTDBTrainer(TrainLatNet, StandardJHTDBArch):
  script_name = __file__
  domains = [FakeDomain]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'dataset': 'jhtdb',
        'lb_to_ln': 2,
        'train_cshape': '16x16x16',
        'seq_length': 1,
        'DxQy': 'D3Q4',
        'batch_size': 16})

  def __init__(self, config):
    super(JHTDBTrainer, self).__init__(config)

if __name__ == '__main__':
  sim = LatNetController(trainer=JHTDBTrainer)
  sim.run()

