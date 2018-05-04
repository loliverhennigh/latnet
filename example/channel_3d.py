#!/usr/bin/env python

import sys
import os
import time

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

def draw_circle(boundary, hx, hy, hz, vertex, radius):
  where_circle = (hx == vertex[0]) & (hy == vertex[1]) & (hz == vertex[2])
  where_circle_vertex = np.where(where_circle)
  for spot in xrange(where_circle_vertex[0].shape[0]):
    for i in range(where_circle_vertex[0][spot]-radius, 
                   where_circle_vertex[0][spot]+radius):
      for j in range(where_circle_vertex[1][spot]-radius, 
                     where_circle_vertex[1][spot]+radius):
        for k in range(where_circle_vertex[2][spot]-radius, 
                       where_circle_vertex[2][spot]+radius):
          if (((i - where_circle_vertex[0][spot])**2 + 
               (j - where_circle_vertex[1][spot])**2 +
               (k - where_circle_vertex[2][spot])**2)
              < radius**2):
            boundary[i, j, k] = True

def rand_vertex(range_x, range_y, radius):
  pos_x = np.random.randint(2*radius, range_x-(2*radius))
  pos_y = np.random.randint(2*radius, range_y-(2*radius))
  pos_z = np.random.randint(2*radius, range_z-(2*radius))
  vertex = np.array([pos_x, pos_y, pos_z])
  return vertex

def make_boundary(hx, hy, hz, shape):
 
  # make circles 
  circles = []
  nr_circles = int(3*(shape[0]*shape[1]*shape[2])/(64*64*64))
  for i in xrange(nr_circles):
    radius = 25 # make this random after testing
    vertex = rand_vertex(np.max(hx), np.max(hy), np.max(hz), radius)
    circles.append((vertex, radius))

  # draw circles
  boundary = (hx == -2)
  boundary = boundary.astype(np.bool)
  for i in xrange(nr_circles): 
    draw_circle(boundary, hx, hy, circles[i][0], circles[i][1])
  return boundary

class ChannelDomain(Domain):
  name = "channel_3d"
  vel = (0.03, 0.00, 0.00)
  sim_shape = [128, 128, 128]
  num_simulations = 10
  grid = 'D3Q15'
  periodic_x = False
  periodic_y = True
  periodic_z = False

  def geometry_boundary_conditions(self, hx, hy, hz, shape):
    walls = (hx == -2)
    obj_boundary = make_boundary(hx, hy, hz, shape)
    where_boundary = walls | obj_boundary
    return where_boundary

  def velocity_boundary_conditions(self, hx, hy, hz, shape):
    where_velocity = (hx == 0)
    velocity = self.vel
    return where_velocity, velocity
 
  def density_boundary_conditions(self, hx, hy, hz, shape):
    where_density = (hx == shape[0] - 1)
    density = 1.0
    return where_density, density

  def velocity_initial_conditions(self, hx, hy, hz, shape):
    velocity = self.vel
    return velocity

  def density_initial_conditions(self, hx, hy, hz, shape):
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
        'visc': 0.001,
        'domain_name': "channel_3d",
        'run_mode': 'generate_data',
        'mode': 'visualization',
        'lb_to_ln': 32,
        'max_sim_iters': 40000})

if __name__ == '__main__':
  sim = LatNetController(trainer=EmptyTrainer)
  sim.run()

