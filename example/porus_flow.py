#!/usr/bin/env python

import sys
import os
import time

import matplotlib.pyplot as plt
from PIL import Image

# import latnet
sys.path.append('../latnet')
from domain import Domain
from trainer import Trainer
from controller import LatNetController
from shape_converter import *
from utils.numpy_utils import *
import utils.binvox_rw as binvox_rw
import numpy as np
import cv2
import glob


def read_tif(sim_shape, file_name="porus.tif"):
  im = Image.open(file_name)
  im = im.resize((im.size[0]/4, im.size[1]/4)) 
  boundary = np.array(im)
  boundary = np.expand_dims(boundary, axis=-1)
  pos = [np.random.randint(0, boundary.shape[0]), np.random.randint(0, boundary.shape[1])]
  boundary = mobius_extract(boundary, SubDomain(pos, [sim_shape[0], sim_shape[1]]))
  boundary = np.minimum(boundary, 1)
  boundary = 1 - boundary 
  boundary[:, 0:16] = 0
  boundary[:,-16:] = 0
  boundary[ 0] = 0
  boundary[-1] = 0
  boundary = mobius_extract(boundary, SubDomain([-1,-1], [sim_shape[0]+2, sim_shape[1]+2]))
  boundary = boundary[:,:,0] 
  boundary = boundary.astype(np.bool)
  plt.imshow(boundary)
  plt.savefig("figs/boundary.png")
  return boundary

class ChannelDomain(Domain):
  name = "porous_flow"
  vel = (0.0, 0.0)
  sim_shape = [512, 512]
  num_simulations = 20
  periodic_x = False
  periodic_y = False
  force = (4e-4, 0.0)
  boundary = read_tif(sim_shape=sim_shape)

  def geometry_boundary_conditions(self, hx, hy, shape):
    return self.boundary

  def velocity_boundary_conditions(self, hx, hy, shape):
    #where_velocity = (hx == 0) & np.logical_not(self.boundary)
    where_velocity = (hx == -2)
    velocity = self.vel
    return where_velocity, velocity
 
  def density_boundary_conditions(self, hx, hy, shape):
    #where_density = (hx == shape[0]-1) & np.logical_not(self.boundary)
    #where_density = (hx == 0) & np.logical_not(self.boundary)
    where_density = (hx == -2)
    density = 1.0
    return where_density, density

  def velocity_initial_conditions(self, hx, hy, shape):
    velocity = (0.0, 0.0)
    return velocity

  def density_initial_conditions(self, hx, hy, shape):
    rho = 1.0
    return rho

  def __init__(self, *args, **kwargs):
    super(ChannelDomain, self).__init__(*args, **kwargs)

class EmptyTrainer(Trainer):
  domains = [ChannelDomain]
  network = None

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'visc': 0.1,
        'domain_name': "channel",
        'run_mode': 'generate_data',
        'mode': 'visualization',
        'lb_to_ln': 128,
        'max_sim_iters': 6401})

if __name__ == '__main__':
  sim = LatNetController(trainer=EmptyTrainer)
  sim.run()

