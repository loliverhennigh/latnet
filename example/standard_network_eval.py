#!/usr/bin/env python

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import domains
from channel import ChannelDomain
from ldc_2d import LDCDomain

# import latnet
sys.path.append('../latnet')
from simulation import Simulation
from controller import LatNetController
from network_architectures.standard_network import StandardNetwork
import numpy as np
import cv2
import glob

class LDCSimulation(Simulation):
  script_name = __file__
  network = StandardNetwork
  #domain = ChannelDomain
  domain = LDCDomain
  domain.sim_shape = [4000, 4000]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'latnet_network_dir': './network_save',
        'run_mode': 'eval',
        'visc': 0.0005,
        'lb_to_ln': 128,
        'input_cshape': '256x256',
        'input_shape': '512x512',
        #'sim_shape': '256x256',
        'max_sim_iters': 600})

if __name__ == '__main__':
  sim = LatNetController(simulation=LDCSimulation)
  sim.run()

