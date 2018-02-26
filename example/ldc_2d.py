#!/usr/bin/env python

import sys
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import latnet
sys.path.append('../latnet')
from domain import Domain
from controller import LatNetController
from trainer import Trainer
from network_architectures.tempogan_network import TempoGAN
import utils.binvox_rw as binvox_rw
import numpy as np
import cv2
import glob

def rand_vel():
  vel_x = np.random.uniform(0.5, 0.4)
  return (vel_x, 0.0)

class LDCDomain(Domain):
  script_name = __file__
  name = "lid_driven_cavity"
  vel = rand_vel()
  num_simulations = 2
  sim_shape = [512, 512]
  periodic_x = False
  periodic_y = False

  def geometry_boundary_conditions(self, hx, hy, shape):
    where_boundary = (shape[0]-9 < hx) & (hx <= shape[0]-1)
    where_boundary = where_boundary | ((0 <= hy) & (hy < 8))
    where_boundary = where_boundary | ((shape[1]-9 < hy) & (hy <= shape[1]-1))
    return where_boundary

  def velocity_boundary_conditions(self, hx, hy, shape):
    where_velocity = (hy == shape[1]-1) & (hx >= 8) & (hx < shape[0]-1)
    velocity = self.vel
    return where_velocity, velocity
 
  def density_boundary_conditions(self, hx, hy, shape):
    where_density = (hx == -2) # all false
    density = 1.0
    return where_density, density

  def velocity_initial_conditions(self, hx, hy, shape):
    return (0.0, 0.0)

  def density_initial_conditions(self, hx, hy, shape):
    rho = 1.0
    return rho

  def __init__(self, *args, **kwargs):
    super(LDCDomain, self).__init__(*args, **kwargs)

class EmptyTrainer(Trainer):
  domains = [LDCDomain]
  network = None

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'visc': 0.1,
        'domain_name': "lid_driven_cavity",
        'run_mode': 'generate_data',
        'mode': 'visualization',
        'subgrid': 'les-smagorinsky',
        'max_sim_iters': 40000})

if __name__ == '__main__':
  sim = LatNetController(trainer=EmptyTrainer)
  sim.run()

