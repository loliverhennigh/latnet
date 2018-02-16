#!/usr/bin/env python

import sys

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

def rand_vel():
  vel_x = np.random.uniform(0.2, 0.1)
  vel_y = np.random.uniform(0.05, -0.05)
  return (vel_y, 0.2)

class TrainDomain(Domain):
  script_name = __file__
  network_name = 'advanced_network'

  vel = rand_vel()

  def geometry_boundary_conditions(self, hx, hy, shape):
    where_boundary = (hx == -2)
    return where_boundary

  def velocity_boundary_conditions(self, hx, hy, shape):
    where_velocity = (hy == shape[1] - 1) & (hx > 0) & (hx < shape[0]/2+10)
    where_velocity = (hy < 32) & (hy >= 0) & (hx >= 0) & (hx < 32) 
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
        'latnet_network_dir': './network_checkpoint_jet',
        'train_sim_dir': './train_data_jet',
        'sim_dir': './eval_data_jet',
        'visc': 0.02,
        'lb_to_ln': 250,
        'seq_length': 2,
        'input_cshape': '64x64',
        'periodic_x': True,
        'periodic_y': True,
        'max_sim_iters': 200,
        'num_simulations': 10,
        'batch_size': 2,
        'sim_shape': '512x512'})

  def compare_script(self, iteration, true_vel, true_rho, generated_vel, generated_rho):
    #plt.imshow(np.concatenate([true_vel[:,:,0], generated_vel[:,:,0]], axis=0))
    plt.imshow(np.concatenate([self.DxQy.vel_to_norm(true_vel)[:,:,0], self.DxQy.vel_to_norm(generated_vel)[:,:,0]], axis=0))
    plt.savefig('./figs/compare_' + str(iteration).zfill(4) + '.png')


  def __init__(self, *args, **kwargs):
    super(TrainDomain, self).__init__(*args, **kwargs)

if __name__ == '__main__':
  sim = LatNetController(_sim=TrainDomain)
  sim.run()
    
