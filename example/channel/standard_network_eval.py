#!/usr/bin/env python


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from channel import ChannelDomain

sys.path.append('../../latnet')
from latnetwork import EvalLatNet
from archs.standard_arch import StandardArch
from controller import LatNetController

class StandardEval(EvalLatNet, StandardArch):
  script_name = __file__
  domain = ChannelDomain
  #domain.sim_shape = [512, 512]
  domain.sim_shape = [256, 256]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'latnet_network_dir': './network_save',
        'dataset': 'sailfish',
        'run_mode': 'eval',
        'visc': 0.01,
        'lb_to_ln': 128,
        'input_cshape': '16x16',
        'input_shape': '256x256',
        'sim_save_every': 1,
        'max_sim_iters': 16})

if __name__ == '__main__':
  sim = LatNetController(simulation=StandardEval)
  sim.run()

