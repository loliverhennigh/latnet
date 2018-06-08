
import sys
import time

import matplotlib.pyplot as plt


# import latnet files
import utils.numpy_utils as numpy_utils
from shape_converter import SubDomain
import lattice

# import sailfish
sys.path.append('../../sailfish')
from sailfish.subdomain import Subdomain2D, Subdomain3D
from sailfish.node_type import NTEquilibriumVelocity, NTFullBBWall, NTDoNothing, NTZouHeVelocity

# import external librarys
import numpy as np
import math
import itertools
from tqdm import *
from copy import copy


class Domain(object):

  def __init__(self, config):
    self.config = config
    self.DxQy = lattice.TYPES[config.DxQy]()

  @classmethod
  def update_defaults(cls, defaults):
      pass

  def geometry_boundary_conditions(self, hx, hy, shape):
    pass

  def velocity_boundary_conditions(self, hx, hy, shape):
    pass

  def density_boundary_conditions(self, hx, hy, shape):
    pass

  def velocity_initial_conditions(self, hx, hy, shape):
    pass

  def density_initial_conditions(self, hx, hy, shape):
    pass

  def make_geometry_input(self, where_boundary, velocity, where_velocity, density, where_density):
    # TODO Clean this
    boundary_array = np.expand_dims(where_boundary, axis=-1).astype(np.float32)
    velocity_array = np.array(velocity).reshape(len(where_velocity.shape) * [1] + [self.DxQy.dims])
    velocity_array = velocity_array * np.expand_dims(where_velocity, axis=-1).astype(np.float32)
    density_array = density * np.expand_dims(where_density, axis=-1).astype(np.float32)
    force_array = 1e5 * np.array(self.force) * np.ones_like(velocity_array).astype(np.float32) # 1e5 to scale force to same range as vel

    input_geometry = np.concatenate([boundary_array,
                                     velocity_array,
                                     density_array,
                                     force_array], axis=-1)
    return input_geometry

  def make_sailfish_subdomain(self):

    velocity_initial_conditions = self.velocity_initial_conditions
    density_initial_conditions = self.density_initial_conditions

    geometry_boundary_conditions = self.geometry_boundary_conditions
    velocity_boundary_conditions = self.velocity_boundary_conditions
    density_boundary_conditions = self.density_boundary_conditions

    make_geometry_input = self.make_geometry_input    
    train_sim_dir = self.config.train_sim_dir

    if hasattr(self, 'force'):
      dom_force = self.force
    else:
      dom_force = None

    bc = NTFullBBWall

    if self.DxQy.dims == 2:
      class SailfishSubdomain(Subdomain2D):

        print(dom_force)
        if dom_force is not None:
          force = dom_force
          
        def boundary_conditions(self, hx, hy):
  
          # restore from old dir or make new geometry
          if self.config.restore_geometry:
            restore_boundary_conditions = np.load(train_sim_dir[:-10] + "flow_geometry.npy")
            where_boundary = restore_boundary_conditions[...,0].astype(np.bool)
            where_velocity = np.logical_or(restore_boundary_conditions[...,1].astype(np.bool), restore_boundary_conditions[...,2].astype(np.bool))
            if len(np.where(where_velocity)[0]) == 0:
              velocity = (0.0,0.0)
            else:
              velocity = (restore_boundary_conditions[np.where(where_velocity)[0][0], np.where(where_velocity)[1][0], 1],
                          restore_boundary_conditions[np.where(where_velocity)[0][0], np.where(where_velocity)[1][0], 2])
            where_density  = restore_boundary_conditions[...,3].astype(np.bool)
            density = 1.0
            #self.force = (restore_boundary_conditions[0,0,4], restore_boundary_conditions[0,0,5])
          else:
            where_boundary = geometry_boundary_conditions(hx, hy, [self.gx, self.gy])
            where_velocity, velocity = velocity_boundary_conditions(hx, hy, [self.gx, self.gy])
            where_density, density = density_boundary_conditions(hx, hy, [self.gx, self.gy])
            #self.force = force
  
          # set boundarys
          self.set_node(where_boundary, bc)
  
          # set velocities
          self.set_node(where_velocity, NTEquilibriumVelocity(velocity))
          #self.set_node(where_velocity, NTZouHeVelocity(velocity))
  
          # set densitys
          self.set_node(where_density, NTDoNothing)
  
          # save geometry
          save_geometry = make_geometry_input(where_boundary, velocity, where_velocity, density, where_density)
          np.save(train_sim_dir + "_geometry.npy", save_geometry)
  
        def initial_conditions(self, sim, hx, hy):
          # set start density
          rho = density_initial_conditions(hx, hy,  [self.gx, self.gy])
          sim.rho[:] = rho
  
          # set start velocity
          vel = velocity_initial_conditions(hx, hy,  [self.gx, self.gy])
          sim.vx[:] = vel[0]
          sim.vy[:] = vel[1]
   


    elif self.DxQy.dims == 3:
      class SailfishSubdomain(Subdomain3D):
  
        def boundary_conditions(self, hx, hy, hz):
  
          # restore from old dir or make new geometry
          if self.config.restore_geometry:
            restore_boundary_conditions = np.load(train_sim_dir[:-10] + "flow_geometry.npy")
            where_boundary = restore_boundary_conditions[...,0].astype(np.bool)
            where_velocity = np.logical_or(restore_boundary_conditions[...,1].astype(np.bool), restore_boundary_conditions[...,2].astype(np.bool), restore_boundary_conditions[...,3].astype(np.bool))
            vel_ind = np.where(where_velocity)
            velocity = restore_boundary_conditions[vel_ind[0][0], vel_ind[1][0], vel_ind[2][0], :]
            velocity = (velocity[1], velocity[2], velocity[3])
            where_density  = restore_boundary_conditions[...,4].astype(np.bool)
            density = 1.0
          else:
            where_boundary = geometry_boundary_conditions(hx, hy, hz, [self.gx, self.gy, self.gz])
            where_velocity, velocity = velocity_boundary_conditions(hx, hy, hz, [self.gx, self.gy, self.gz])
            where_density, density = density_boundary_conditions(hx, hy, hz, [self.gx, self.gy, self.gz])
  
          # set boundarys
          self.set_node(where_boundary, bc)
  
          # set velocities
          self.set_node(where_velocity, NTEquilibriumVelocity(velocity))
          #self.set_node(where_velocity, NTZouHeVelocity(velocity))
  
          # set densitys
          self.set_node(where_density, NTDoNothing)
  
          # save geometry
          save_geometry = make_geometry_input(where_boundary, velocity, where_velocity, density, where_density)
          np.save(train_sim_dir + "_geometry.npy", save_geometry)
  
        def initial_conditions(self, sim, hx, hy):
          # set start density
          rho = density_initial_conditions(hx, hy,  [self.gx, self.gy])
          sim.rho[:] = rho
  
          # set start velocity
          vel = velocity_initial_conditions(hx, hy,  [self.gx, self.gy])
          sim.vx[:] = vel[0]
          sim.vy[:] = vel[1]
          sim.vy[:] = vel[1]

    return SailfishSubdomain
   

