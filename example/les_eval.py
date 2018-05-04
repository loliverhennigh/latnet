#!/usr/bin/env python

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import domains
from ldc_2d_les import LESDomain

# import latnet
sys.path.append('../latnet')
from domain import Domain
from controller import LatNetController
from simulation import Simulation 
from network_architectures.les_network import LESNet
import numpy as np
import cv2
import glob

class LESSimulation(Simulation):
  script_name = __file__
  network = LESNet
  domain = LESDomain

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'run_mode': 'eval',
        'latnet_network_dir': './network_save',
        'visc': 0.001,
        'lb_to_ln': 8,
        'input_cshape': '64x64',
        'max_sim_iters': 800})

if __name__ == '__main__':
  sim = LatNetController(simulation=LESSimulation)
  sim.run()

