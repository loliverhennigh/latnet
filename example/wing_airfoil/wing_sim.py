#!/usr/bin/env python

import sys
import os
import time

import matplotlib.pyplot as plt

from boundary_utils import *

# import latnet
sys.path.append('../../latnet')
from domain import Domain
from trainer import Trainer
from controller import LatNetController
from utils.numpy_utils import *
import utils.binvox_rw as binvox_rw
from shape_converter import *
import numpy as np
import cv2
import glob

class WingDomain(Domain):
  name = "channel"
  vel = (0.05, 0.0)
  sim_shape = [512, 1024]
  wing_shape = [256, 256]
  num_simulations = 20
  periodic_x = False
  periodic_y = False
  force = (0.0, 0.0)
  _, wing_boundary = wing_boundary_batch(25, 1, wing_shape, 2)
  wing_boundary = wing_boundary[0,:,:]
  #pos = [np.random.randint(0, wing_shape[0]), np.random.randint(wing_shape[1]/2 - sim_shape[1], wing_shape[1]/2)]
  pos = [-128, -128]
  subdomain = SubDomain(pos, sim_shape)
  wing_boundary = mobius_extract(wing_boundary, subdomain, padding_type=['zero', 'zero'])
  wing_boundary = mobius_extract(wing_boundary, SubDomain([-1,-1], [sim_shape[0]+2, sim_shape[1]+2]))
  wing_boundary = wing_boundary.astype(np.bool)
  wing_boundary = wing_boundary[:,:,0]


  def geometry_boundary_conditions(self, hx, hy, shape):
    return self.wing_boundary

  def velocity_boundary_conditions(self, hx, hy, shape):
    where_velocity = (hx == 0) & np.logical_not(self.wing_boundary)
    velocity = self.vel
    return where_velocity, velocity
 
  def density_boundary_conditions(self, hx, hy, shape):
    where_density = (hx == shape[0]-1) & np.logical_not(self.wing_boundary)
    density = 1.0
    return where_density, density

  def velocity_initial_conditions(self, hx, hy, shape):
    velocity = self.vel
    return velocity

  def density_initial_conditions(self, hx, hy, shape):
    rho = 1.0
    return rho

  def __init__(self, *args, **kwargs):
    super(WingDomain, self).__init__(*args, **kwargs)

class EmptyTrainer(Trainer):
  domains = [WingDomain]
  network = None

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'visc': 0.0002,
        'domain_name': "channel",
        'run_mode': 'generate_data',
        'mode': 'visualization',
        'lb_to_ln': 128,
        'max_sim_iters': 40000})

if __name__ == '__main__':
  sim = LatNetController(trainer=EmptyTrainer)
  sim.run()

