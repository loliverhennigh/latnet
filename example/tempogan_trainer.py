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
import utils.binvox_rw as binvox_rw
import numpy as np
import cv2
import glob

class ChannelDomain(Domain):
  script_name = __file__
  vel = (0.04, 0.00)
  sim_shape = [256, 512]
  periodic_x = False
  periodic_y = True

  def geometry_boundary_conditions(self, hx, hy, shape):
    walls = (hx == -2)
    obj_boundary = make_boundary(hx, hy)
    where_boundary = walls | obj_boundary
    return where_boundary

  def velocity_boundary_conditions(self, hx, hy, shape):
    where_velocity = (hx == 0)
    velocity = self.vel
    return where_velocity, velocity
 
  def density_boundary_conditions(self, hx, hy, shape):
    where_density = (hx == shape[0] - 1)
    density = 1.0
    return where_density, density

  def velocity_initial_conditions(self, hx, hy, shape):
    velocity = self.vel
    return velocity

  def density_initial_conditions(self, hx, hy, shape):
    rho = 1.0
    return rho

  def __init__(self, *args, **kwargs):
    super(ChannelDomain, self).__init__(*args, **kwargs)

class LDCDomain(Domain):
  script_name = __file__
  vel = rand_vel()
  sim_shape = [256, 512]
  periodic_x = False
  periodic_y = True

  def geometry_boundary_conditions(self, hx, hy, shape):
    where_boundary = (hx == shape[0]-1) | (hx == 0) | (hy == 0)
    return where_boundary

  def velocity_boundary_conditions(self, hx, hy, shape):
    where_velocity = (hy == shape[1]-1) & (hx > 0) & (hx < shape[0]-1)
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

class NetworkTrainer():
  network = TempoGan 
  domains = [ChannelDomain, LDCDomain]
  num_simulations = [10, 10]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'train_sim_dir': './train_data',
        'visc': 0.005,
        'lb_to_ln': 50,
        'input_cshape': '64x64',
        'periodic_x': False,
        'periodic_y': False,
        'max_sim_iters': 400,
        'num_simulations': 10,
        'sim_shape': '256x512'})

class Simulation():
  network = TempoGan 
  domains = [ChannelDomain, LDCDomain]
  num_simulations = [10, 10]

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'train_sim_dir': './train_data',
        'visc': 0.005,
        'lb_to_ln': 50,
        'input_cshape': '64x64',
        'periodic_x': False,
        'periodic_y': False,
        'max_sim_iters': 400,
        'num_simulations': 10,
        'sim_shape': '256x512'})


if __name__ == '__main__':
  sim = LatNetController(network=TempoGan, sim=TrainDomain)
  sim.run()
