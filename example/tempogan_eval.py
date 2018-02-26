#!/usr/bin/env python

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import domains
from channel import ChannelDomain

# import latnet
sys.path.append('../latnet')
from domain import Domain
from controller import LatNetController
from simulation import Simulation
from network_architectures.tempogan_network import TempoGAN
import numpy as np
import cv2
import glob

class TempoGanSimulation(Simulation):
  script_name = __file__
  network = TempoGAN
  domain = ChannelDomain

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'run_mode': 'eval',
        'latnet_network_dir': './network_save',
        'visc': 0.1,
        'lb_to_ln': 50,
        'input_cshape': '64x64',
        'max_sim_iters': 400})

if __name__ == '__main__':
  sim = LatNetController(simulation=TempoGanSimulation)
  sim.run()

