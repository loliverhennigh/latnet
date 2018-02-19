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
            < radius**2):
          boundary[i, j] = True

def rand_vertex(range_x, range_y, radius):
  pos_x = np.random.randint(2*radius, range_x-(2*radius))
  pos_y = np.random.randint(2*radius, range_y-(2*radius))
  vertex = np.array([pos_x, pos_y])
  return vertex

def make_boundary(hx, hy):
 
  # make circles 
  circles = []
  nr_circles = 5
  for i in xrange(nr_circles):
    radius = 20 # make this random after testing
    vertex = rand_vertex(np.max(hx), np.max(hy), radius)
    circles.append((vertex, radius))

  # draw circles
  boundary = (hx == -2)
  boundary = boundary.astype(np.bool)
  for i in xrange(nr_circles): 
    draw_circle(boundary, hx, hy, circles[i][0], circles[i][1])
  return boundary

class TrainDomain(Domain):
  script_name = __file__
  network_name = 'tempogan_network'
  vel = (0.04, 0.00)

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

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
        'latnet_network_dir': './network_checkpoint_channel',
        'train_sim_dir': './train_data_channel',
        'sim_dir': './eval_channel',
        'visc': 0.01,
        'lb_to_ln': 50,
        'seq_length': 5,
        'input_cshape': '64x64',
        'periodic_x': False,
        'periodic_y': True,
        'nonlinearity': 'relu',
        'max_sim_iters': 400,
        'num_simulations': 10,
        'sim_shape': '512x256'})

  def compare_script(self, iteration, true_vel, true_rho, generated_vel, generated_rho):
    plt.imshow(np.concatenate([self.DxQy.vel_to_norm(true_vel)[:,:,0], self.DxQy.vel_to_norm(generated_vel)[:,:,0]], axis=0))
    #plt.imshow(self.DxQy.vel_to_norm(true_vel)[:,:,0])
    plt.savefig('./figs/compare_' + str(iteration).zfill(4) + '.png')

  def __init__(self, *args, **kwargs):
    super(TrainDomain, self).__init__(*args, **kwargs)

if __name__ == '__main__':
  sim = LatNetController(_sim=TrainDomain)
  sim.run()

  """
  # generate video of plots
  time.sleep(5.0)
  os.system("rm ./figs/channel_video.mp4")
  os.system("cat ./figs/compare* | ffmpeg -f image2pipe -r 10 -vcodec png -i - -vcodec libx264 ./figs/channel_video.mp4")
  os.system("rm ./figs/compare*")
  """


    

