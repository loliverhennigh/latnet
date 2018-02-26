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
from domain import Domain
from controller import LatNetController
from trainer import Trainer
from network_architectures.standard_network import StandardNetwork
import utils.binvox_rw as binvox_rw
import numpy as np
import cv2
import glob

class TempoGanTrainer(Trainer):
  script_name = __file__
  network = StandardNetwork
  domains = [ChannelDomain, LDCDomain]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'train_sim_dir': './train_data',
        'latnet_network_dir': './network_save',
        'visc': 0.01,
        'lb_to_ln': 50,
        'input_cshape': '32x32',
        'max_sim_iters': 400})

if __name__ == '__main__':
  sim = LatNetController(trainer=TempoGanTrainer)
  sim.run()

