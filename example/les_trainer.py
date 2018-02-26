#!/usr/bin/env python

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import domains
from channel import ChannelDomain
#from ldc_2d import LDCDomain

# import latnet
sys.path.append('../latnet')
from domain import Domain
from controller import LatNetController
from trainer import Trainer
from network_architectures.les_network import LESNet
import numpy as np
import cv2
import glob

class TempoGanTrainer(Trainer):
  script_name = __file__
  network = LESNet
  domains = [ChannelDomain]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'train_sim_dir': './train_data',
        'latnet_network_dir': './network_save',
        'visc': 0.1,
        'lb_to_ln': 16,
        'input_cshape': '64x64',
        'max_sim_iters': 400})

if __name__ == '__main__':
  sim = LatNetController(trainer=TempoGanTrainer)
  sim.run()

