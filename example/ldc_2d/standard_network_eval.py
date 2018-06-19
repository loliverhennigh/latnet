#!/usr/bin/env python

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import domains
from ldc_2d import LDCDomain

# import latnet
sys.path.append('../../latnet')
from simulation import Simulation
from controller import LatNetController
from network_architectures.standard_network import StandardNetwork
import numpy as np
import cv2
import glob

class LDCSimulation(Simulation):
  script_name = __file__
  network = StandardNetwork
  domain = LDCDomain
  domain.sim_shape = [512, 512]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'latnet_network_dir': './network_save',
        'run_mode': 'eval',
        'visc': 0.001,
        'lb_to_ln': 256,
        'input_cshape': '256x256',
        'input_shape': '512x512',
        'max_sim_iters': 1600})

if __name__ == '__main__':
  sim = LatNetController(simulation=LDCSimulation)
  sim.run()

