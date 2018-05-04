#!/usr/bin/env python

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import domains

# import latnet
sys.path.append('../latnet')
from domain import Domain
from controller import LatNetController
from trainer import Trainer
from network_architectures.standard_network import StandardNetwork
import utils.binvox_rw as binvox_rw
import numpy as np
import cv2
import glob

class FakeDomain:
  sim_shape = [1024, 1024, 1024]
  name = "JHTDB"
  num_simulations = 4

class StandardTrainer(Trainer):
  script_name = __file__
  network = StandardNetwork
 
  domains = [FakeDomain]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'train_sim_dir': './train_data',
        'latnet_network_dir': './network_save',
        'input_cshape': '8x8x8',
        'input_shape': '16x16x16',
        'sim_shape': '1024x1024x1024',
        'nr_downsamples': 2,
        'filter_size': 16,
        'filter_size_compression': 32,
        'nr_residual_compression': 2,
        'seq_length': 6,
        'DxQy': 'D3Q4'})

if __name__ == '__main__':
  sim = LatNetController(trainer=StandardTrainer)
  sim.run()

