#!/usr/bin/env python

import sys
sys.path.append('../sailfish/')

# import sailfish
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTHalfBBWall, NTRegularizedVelocity, NTRegularizedDensity, DynamicValue, NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_base import ForceObject
from sailfish.lb_single import LBFluidSim
from sailfish.sym import S

# import latnet librarys
from latnet.random_boundary.random_obj
from latnet.boundary

# import important librarys
import numpy as np

def rand_vel(max_vel=.10, min_vel=.09):
  vel = np.random.uniform(min_vel, max_vel)
  angle = np.random.uniform(-np.pi/2, np.pi/2)
  vel_x = vel * np.cos(angle)
  vel_y = vel * np.sin(angle)
  return (vel_x, vel_y)

class BoxSubdomain(Subdomain2D):
  bc = NTFullBBWall
  max_v = 0.1
  vel = rand_vel()

  def boundary_conditions(self, hx, hy):

    # set walls
    walls = (hx == -2) # set to all false
    y_wall = np.random.randint(0,2)
    if y_wall == 0:
      walls = (hy == 0) | (hy == self.gy - 1) | walls
    self.set_node(walls, self.bc)
    self.set_node((hx == 0) & np.logical_not(walls),
                  NTEquilibriumVelocity(self.vel))

    # set open boundarys 
    self.set_node((hx == self.gx - 1) & np.logical_not(walls),
                  NTEquilibriumDensity(1))
    boundary = self.rand_obj
    self.set_node(boundary, self.bc)

    # save geometry (boundary, velocity, pressure)
    boudnary.save(self.config.checkpoint_file + "_geometry", geometry)

  def initial_conditions(self, sim, hx, hy):
    H = self.config.lat_ny
    sim.rho[:] = 1.0
    sim.vy[:] = self.vel[1]
    sim.vx[:] = self.vel[0]

class BoxSimulation(LBFluidSim):
  subdomain = BoxSubdomain

  @classmethod
  def add_options(cls, group, defaults):
    group.add_argument('--sim_size',
            help='size of simulation to run ',
            type=int, default=300)

  @classmethod
  def update_defaults(cls, defaults):
    defaults.update({
      'max_iters': 60000,
      'output_format': 'npy',
      'periodic_y': True,
      'periodic_x': True,
      'checkpoint_file': '/data/sailfish_store/test_checkpoint',
      'checkpoint_every': 120,
      })

  @classmethod
  def modify_config(cls, config):
    config.lat_nx = config.sim_size
    config.lat_ny = config.sim_size
    config.visc   = 0.1

  def __init__(self, *args, **kwargs):
    super(BoxSimulation, self).__init__(*args, **kwargs)



if __name__ == '__main__':
  network_ctrl = latnetSimulationController(name="blaa", lskdjf, lsdjf)
  network.train()
    

