#!/usr/bin/env python


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from channel import ChannelDomain

sys.path.append('../../latnet')
from latnetwork import EvalLatNet
from archs.standard_sailfish_arch import StandardSailfishArch
from controller import LatNetController
from shape_converter import SubDomain

class StandardEval(EvalLatNet, StandardSailfishArch):
  script_name = __file__
  domain = ChannelDomain
  #domain.sim_shape = [4096, 4096]
  #domain.sim_shape = [512, 512]
  domain.sim_shape = [256, 256]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'dataset': 'sailfish',
        'run_mode': 'eval',
        'visc': 0.01,
        'lb_to_ln': 128,
        'sim_save_every': 32,
        'num_iters': 128,
        'filter_size': 17,
        'eval_cshape': '64x64'})

  def __init__(self, config):
    super(StandardEval, self).__init__(config)

if __name__ == '__main__':
  sim = LatNetController(simulation=StandardEval)
  sim.run()

