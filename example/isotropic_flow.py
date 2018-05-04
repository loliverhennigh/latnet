#!/usr/bin/env python

import sys
import os
import time
import scipy.io

import matplotlib.pyplot as plt

# import latnet
sys.path.append('../latnet')
from domain import Domain
from trainer import Trainer
from controller import LatNetController
import utils.binvox_rw as binvox_rw
import numpy as np
import cv2
import glob

class ChannelDomain(Domain):
  name = "channel"
  vel = (0.0, 0.0)
  sim_shape = [512, 512]
  num_simulations = 4
  periodic_x = True
  periodic_y = True
  force = (0.0, 0.0)

  def geometry_boundary_conditions(self, hx, hy, shape):
    obj_boundary = (hx == -2)
    return obj_boundary

  def velocity_boundary_conditions(self, hx, hy, shape):
    where_velocity = (hx == -2)
    velocity = self.vel
    return where_velocity, velocity
 
  def density_boundary_conditions(self, hx, hy, shape):
    where_density = (hx == -2)
    density = 1.0
    return where_density, density

  def velocity_initial_conditions(self, hx, hy, shape):
    strip_size = 64
    velocity = np.zeros([2] + shape)
    for i in xrange(shape[1]/strip_size):
      velocity[0,(i*strip_size):(((i+1)*strip_size)-strip_size/2)] = 0.1 
      velocity[0,(((i+1)*strip_size)-strip_size/2):((i+1)*strip_size)] = -0.1
    velocity += (np.random.rand(*velocity.shape)-0.5)/200.
    return velocity

  def density_initial_conditions(self, hx, hy, shape):
    return 1.0

  def __init__(self, *args, **kwargs):
    super(ChannelDomain, self).__init__(*args, **kwargs)

class EmptyTrainer(Trainer):
  domains = [ChannelDomain]
  network = None

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'visc': 0.01,
        'domain_name': "channel",
        'run_mode': 'generate_data',
        'subgrid': 'none',
        'mode': 'visualization',
        'lb_to_ln': 128,
        'max_sim_iters': 40000})

if __name__ == '__main__':
  sim = LatNetController(trainer=EmptyTrainer)
  sim.run()

