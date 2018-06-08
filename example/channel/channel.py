#!/usr/bin/env python

import sys
import os
import time

import matplotlib.pyplot as plt

# import latnet
sys.path.append('../../latnet')
from domain import Domain
from trainer import Trainer
from controller import LatNetController
import utils.binvox_rw as binvox_rw
import numpy as np
import cv2
import glob

def draw_circle(boundary, hx, hy, vertex, radius):
  where_circle = (hx == vertex[0]) & (hy == vertex[1])
  where_circle_vertex = np.where(where_circle)
  for spot in xrange(where_circle_vertex[0].shape[0]):
    for i in range(where_circle_vertex[0][spot]-radius, 
                   where_circle_vertex[0][spot]+radius):
      for j in range(where_circle_vertex[1][spot]-radius, 
                     where_circle_vertex[1][spot]+radius):
        if (((i - where_circle_vertex[0][spot])**2 + 
             (j - where_circle_vertex[1][spot])**2)
            < (radius**2)):
          if (i < boundary.shape[0]) and (j < boundary.shape[1]):
            boundary[i, j] = True

def rand_vertex(range_x, range_y, radius):
  pos_x = np.random.randint(2*radius, range_x-(2*radius))
  pos_y = np.random.randint(2*radius, range_y-(2*radius))
  vertex = np.array([pos_x, pos_y])
  return vertex

def make_boundary(hx, hy, shape):
 
  # make circles 
  circles = []
  nr_circles = int(10*(shape[0]*shape[1])/(256*256))
  for i in xrange(nr_circles):
    radius = 10 # make this random after testing
    vertex = rand_vertex(np.max(hx), np.max(hy), radius)
    circles.append((vertex, radius))

  # draw circles
  boundary = (hx == -2)
  boundary = boundary.astype(np.bool)
  for i in xrange(nr_circles): 
    draw_circle(boundary, hx, hy, circles[i][0], circles[i][1])
  return boundary

class ChannelDomain(Domain):
  name = "channel"
  vel = (0.05, 0.0)
  sim_shape = [256, 256]
  num_simulations = 20
  periodic_x = False
  periodic_y = False
  force = (0.0, 0.0)

  def geometry_boundary_conditions(self, hx, hy, shape):
    obj_boundary = make_boundary(hx, hy, shape)
    return obj_boundary

  def velocity_boundary_conditions(self, hx, hy, shape):
    where_velocity = (hx == 0)
    velocity = self.vel
    return where_velocity, velocity
 
  def density_boundary_conditions(self, hx, hy, shape):
    where_density = (hx == shape[0]-1)
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

class EmptyTrainer(Trainer):
  domains = [ChannelDomain]
  network = None

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'visc': 0.01,
        'domain_name': "channel",
        'run_mode': 'generate_data',
        'mode': 'visualization',
        'lb_to_ln': 128,
        'max_sim_iters': 40000})

if __name__ == '__main__':
  sim = LatNetController(trainer=EmptyTrainer)
  sim.run()

        #'sim_shape': '256x256',
