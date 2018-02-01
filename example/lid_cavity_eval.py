#!/usr/bin/env python

import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# import latnet
sys.path.append('../latnet')
from domain import Domain
from controller import LatNetController
from sim_saver import SimSaver
import utils.binvox_rw as binvox_rw
import numpy as np
import cv2
import glob

class EvalDomain(Domain):
  script_name = __file__
  network_name = 'advanced_network'

  vel = (0.2, -0.005)

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

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'latnet_network_dir': './network_checkpoint_lid_driven_cavity',
        'sim_dir': './eval_data_lid_driven_cavity',
        'visc': 0.01,
        'lb_to_ln': 500,
        'seq_length': 5,
        'input_shape': '512x512',
        'nr_downsamples': 3,
        'nr_residual_encoder': 1,
        'nr_residual_compression': 2,
        'nonlinearity': 'relu',
        'filter_size': 32,
        'filter_size_compression': 32,
        'gated': True,
        'max_sim_iters': 200,
        'sim_shape': '512x512',
        'num_iters': 15,
        'run_mode': 'eval'})

  def compare_script(self, iteration, true_vel, true_rho, generated_vel, generated_rho):
    #plt.imshow(np.concatenate([true_vel[:,:,0], generated_vel[:,:,0]], axis=0))
    plt.imshow(self.DxQy.vel_to_norm(true_vel)[:,:,0])
    plt.savefig('./figs/compare_' + str(iteration) + '.png')

  def __init__(self, *args, **kwargs):
    super(EvalDomain, self).__init__(*args, **kwargs)

if __name__ == '__main__':
  sim = LatNetController(_sim=EvalDomain)
  sim.run()

