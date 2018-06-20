#!/usr/bin/env python

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import latnet
sys.path.append('../../latnet')
from domain import Domain
from simulation import Simulation
from controller import LatNetController
from network_architectures.standard_network import StandardNetwork
import utils.binvox_rw as binvox_rw
import numpy as np
import cv2
import glob

class FakeDomain(Domain):
  sim_shape = [200,200,200]
  name = "JHTDB"
  num_simulations = 10
  periodic_x = True
  periodic_y = True
  periodic_z = True

class JHTDBSimulation(Simulation):
  script_name = __file__
  network = StandardNetwork
  domain = FakeDomain

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'train_sim_dir': './train_data',
        'run_mode': 'eval',
        'latnet_network_dir': './network_save',
        'input_cshape': '16x16x16',
        'input_shape': '200x200x200',
        'sim_shape': '128x128x128',
        'nr_downsamples': 2,
        'filter_size': 16,
        'filter_size_compression': 32,
        'nr_residual_compression': 1,
        'nr_residual_encoder': 2,
        'seq_length': 4,
        'compare': False,
        'DxQy': 'D3Q4'})

if __name__ == '__main__':
  sim = LatNetController(simulation=JHTDBSimulation)
  sim.run()

