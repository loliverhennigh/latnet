#!/usr/bin/env python

import sys

# import latnet
sys.path.append('../latnet')
from domain import Domain
from controller import LatNetController
import utils.binvox_rw as binvox_rw
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def rand_vel(max_vel=.10, min_vel=.09):
  vel = np.random.uniform(min_vel, max_vel)
  angle = np.random.uniform(-np.pi/2, np.pi/2)
  vel_x = vel * np.cos(angle)
  vel_y = vel * np.sin(angle)
  return (vel_x, vel_y)

def draw_triangle(boundary, vertex_1, vertex_2, vertex_3):
  # just using cv2 imp
  triangle = np.array([[vertex_1[1],vertex_1[0]],[vertex_2[1],vertex_2[0]],[vertex_3[1],vertex_3[0]]], np.int32)
  triangle = triangle.reshape((-1,1,2))
  cv2.fillConvexPoly(boundary,triangle,1)
  return boundary

def rand_vertex(range_x, range_y):
  pos_x = np.random.randint(range_x[0], range_x[1])
  pos_y = np.random.randint(range_y[0], range_y[1])
  vertex = np.array([pos_x, pos_y])
  return vertex

def draw_random_triangle(boundary, size_range):
  shape = boundary.shape
  size_x_1 = np.random.randint(-size_range, size_range)
  size_y_1 = np.random.randint(-size_range, size_range)
  size_x_2 = np.random.randint(-size_range, size_range)
  size_y_2 = np.random.randint(-size_range, size_range)
  max_length_x = np.max([np.abs(size_x_1), np.abs(size_x_2)])
  max_length_y = np.max([np.abs(size_y_1), np.abs(size_y_2)])
  vertex = rand_vertex([max_length_x, shape[0]-max_length_x], [max_length_y+int(.1*shape[1]), shape[1]-max_length_y-int(.1*shape[1])])
  boundary = draw_triangle(boundary, vertex, [vertex[0]+size_x_2, vertex[1]+size_y_2], [vertex[0]+size_x_1, vertex[1]+size_y_1])
  return boundary

def make_boundary(hx):
  boundary = (hx == -2)
  boundary = boundary.astype(np.uint8)
  for i in xrange(5):
    boundary = draw_random_triangle(boundary, 50)
  boundary = boundary.astype(np.bool)
  return boundary

class EvalDomain(Domain):
  script_name = __file__
  max_v = 0.1
  vel = rand_vel()

  def geometry_boundary_conditions(self, hx, hy, shape):
    walls = (hx == -2)
    obj_boundary = make_boundary(hx)
    where_boundary = walls | obj_boundary
    return where_boundary

  def velocity_boundary_conditions(self, hx, hy, shape):
    where_velocity = (hx == 0)
    plt.imshow(hx)
    plt.show()
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
    super(EvalDomain, self).__init__(*args, **kwargs)

if __name__ == '__main__':
  sim = LatNetController(eval_sim=EvalDomain)
  sim.run()
    

