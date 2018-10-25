#!/usr/bin/env python

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import latnet
sys.path.append('../../latnet')
from latnetwork import EvalLatNet
from domain import SpectralDNSDomain
from archs.standard_jhtdb_arch import StandardJHTDBArch
from controller import LatNetController

class FakeDomain(SpectralDNSDomain):
  name = "SpectralDNS"
  sim_shape = [128, 128, 128]
  periodic_x = True
  periodic_y = True
  periodic_z = True

class JHTDBSimulation(EvalLatNet, StandardJHTDBArch):
  script_name = __file__
  domain = FakeDomain

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'run_mode': 'eval',
        'dataset': 'spectraldns',
        'lb_to_ln': 30,
        #'lb_to_ln': 32,
        'eval_cshape': '32x32x32',
        #'eval_cshape': '16x16x16',
        'sim_save_every': 1,
        'num_iters': 32,
        'DxQy': 'D3Q4'})

if __name__ == '__main__':
  sim = LatNetController(simulation=JHTDBSimulation)
  sim.run()

