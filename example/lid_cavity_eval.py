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

class LDCSaver(SimSaver):

  def compare_true_generated(self, iteration, sailfish_state, network_state):
    if iteration == 10:
      sailfish_vel = self.DxQy.lattice_to_vel(sailfish_state)
      latnet_vel = self.DxQy.lattice_to_vel(network_state)
      plt.imshow(np.concatenate([sailfish_vel[:,:,:,0], latnet_vel[:,:,:,0]], axis=0))
      plt.show()

  def visualizer(self, iteration, state):
    if iteration == 10:
      vel = self.DxQy.lattice_to_norm(state)
      plt.imshow(vel[:,:,:,0])
      plt.show()

class LDCDomain(Domain):
  script_name = __file__

  vel = (0.2, 0.0)

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

if __name__ == '__main__':
  sim = LatNetController(_sim=LDCDomain)
  sim.run()

