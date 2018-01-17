
import sys

import matplotlib.pyplot as plt


# import latnet files
import utils.numpy_utils as numpy_utils
from shape_converter import SubDomain

# import sailfish
sys.path.append('../sailfish')
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTHalfBBWall, NTEquilibriumVelocity, NTEquilibriumDensity, DynamicValue, NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_base import ForceObject
from sailfish.lb_single import LBFluidSim
from sailfish.sym import S

# import external librarys
import numpy as np
import math
import itertools
from tqdm import *
from copy import copy


class Domain(object):

  def __init__(self, config, nr_downsamples=0): # once network config is in correctly will just need config

    sim_shape = config.sim_shape.split('x')
    sim_shape = map(int, sim_shape)
    self.sim_shape = sim_shape

    downsample_factor = pow(2, nr_downsamples)
    self.sim_cshape = [x/downsample_factor for x in sim_shape] 

    input_shape = config.input_shape.split('x')
    input_shape = map(int, input_shape)
    self.input_shape = input_shape

    self.input_cshape = [32,32] # hard set for now

    self.sailfish_sim_dir = config.sailfish_sim_dir
    self.max_sim_iters = config.max_sim_iters
    self.lb_to_ln = config.lb_to_ln
    self.visc = config.visc
    self.restore_geometry = config.restore_geometry

  def boundary_conditions(self, hx, hy):
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

  def vel_to_lattice(self, vel):
    vel = np.array(vel)
    C = np.array([ [0,0], [1,0], [0,1], [-1,0], [0,-1], [1,1], [-1,1], [-1,-1], [1,-1] ])
    W = np.array(  [4./9., 1./9., 1./9., 1./9., 1./9., 1./36., 1./36., 1./36., 1./36.])

    vel_dot_vel = np.sum(vel * vel)
    vel_dot_c = np.sum(np.expand_dims(vel, axis=0) * C, axis=-1)
    feq = W * (1.0 + 
               3.0*vel_dot_c + 
               4.5*vel_dot_c*vel_dot_c - 
               1.5*vel_dot_vel)
    feq = feq - W
    return feq 

  def make_geometry_input(self, where_boundary, velocity, where_velocity, density, where_density):
    input_geometry = np.concatenate([np.expand_dims(where_boundary, axis=-1).astype(np.float32),
                                     np.array(velocity).reshape(1,1,2) * np.expand_dims(where_velocity, axis=-1).astype(np.float32),
                                     density *  np.expand_dims(where_density, axis=-1).astype(np.float32)], axis=-1)
    return input_geometry

  def create_sailfish_simulation(self):

    # I think I can fix these problems with inheritence but screw it for now
    # boundary conditions
    geometry_boundary_conditions = self.geometry_boundary_conditions
    velocity_boundary_conditions = self.velocity_boundary_conditions
    density_boundary_conditions = self.density_boundary_conditions

    # init conditions
    velocity_initial_conditions = self.velocity_initial_conditions
    density_initial_conditions = self.density_initial_conditions

    # update defaults
    shape = self.sim_shape
    sailfish_sim_dir = self.sailfish_sim_dir
    max_iters = self.max_sim_iters
    lb_to_ln = self.lb_to_ln
    visc = self.visc
    restore_geometry = self.restore_geometry

    # inportant func
    make_geometry_input = self.make_geometry_input

    class SailfishSubdomain(Subdomain2D):
      
      bc = NTFullBBWall

      def boundary_conditions(self, hx, hy):

        # restore from old dir or make new geometry
        if restore_geometry:
          restore_boundary_conditions = np.load(sailfish_sim_dir[:-10] + "flow_geometry.npy")
          where_boundary = restore_boundary_conditions[:,:,0].astype(np.bool)
          where_velocity = restore_boundary_conditions[:,:,1].astype(np.bool)
          velocity = (restore_boundary_conditions[-1,-1,1], restore_boundary_conditions[-1,-1,2])
          where_density  = restore_boundary_conditions[:,:,3].astype(np.bool)
          density = 1.0
        else:
          where_boundary = geometry_boundary_conditions(hx, hy, [self.gx, self.gy])
          where_velocity, velocity = velocity_boundary_conditions(hx, hy, [self.gx, self.gy])
          where_density, density = density_boundary_conditions(hx, hy, [self.gx, self.gy])

        # set boundarys
        self.set_node(where_boundary, self.bc)

        # set velocities
        self.set_node(where_velocity, NTEquilibriumVelocity(velocity))

        # set densitys
        self.set_node(where_density, NTEquilibriumDensity(density))

        # save geometry
        save_geometry = make_geometry_input(where_boundary, velocity, where_velocity, density, where_density)
        np.save(sailfish_sim_dir + "_geometry.npy", save_geometry)

      def initial_conditions(self, sim, hx, hy):
        # set start density
        rho = density_initial_conditions(hx, hy,  [self.gx, self.gy])
        sim.rho[:] = rho

        # set start velocity
        vel = velocity_initial_conditions(hx, hy,  [self.gx, self.gy])
        sim.vx[:] = vel[0]
        sim.vy[:] = vel[1]
   
    class SailfishSimulation(LBFluidSim): 
      subdomain = SailfishSubdomain

      
      @classmethod
      def add_options(cls, group, dim):
        group.add_argument('--sailfish_sim_dir', help='all modes', type=str,
                              default='')
        group.add_argument('--run_mode', help='all modes', type=str,
                              default='')
        group.add_argument('--max_sim_iters', help='all modes', type=int,
                              default=1000)
        group.add_argument('--restore_geometry', help='all modes', type=bool,
                              default=False)

      @classmethod
      def update_defaults(cls, defaults):
        defaults.update({
          'max_iters': max_iters,
          'output_format': 'npy',
          'periodic_y': True,
          'periodic_x': True,
          'checkpoint_file': sailfish_sim_dir,
          'checkpoint_every': lb_to_ln,
          'lat_nx': shape[0],
          'lat_ny': shape[1]
          })

      @classmethod
      def modify_config(cls, config):
        config.visc   = visc
        config.mode   = "batch"

      def __init__(self, *args, **kwargs):
        super(SailfishSimulation, self).__init__(*args, **kwargs)

    ctrl = LBSimulationController(SailfishSimulation)

    return ctrl

  def state_to_cstate(self, encoder, encoder_shape_converter):

    nr_subdomains = [int(math.ceil(x/float(y))) for x, y in zip(self.sim_cshape, self.input_cshape)]
    cstate = []
    for i, j in itertools.product(xrange(nr_subdomains[0]), xrange(nr_subdomains[1])):
      pos = [i * self.input_cshape[0], j * self.input_cshape[1]]
      subdomain = SubDomain(pos, self.input_cshape)
      input_subdomain = encoder_shape_converter.out_in_subdomain(subdomain)
      vel = self.velocity_initial_conditions(0,0,None)
      start_state = np.zeros([1] + input_subdomain.size + [9]) + self.vel_to_lattice(vel).reshape((1,1,1,9))
      cstate.append(encoder(start_state))

    # list to full tensor
    cstate = numpy_utils.stack_grid(cstate, nr_subdomains, has_batch=True)

    # trim edges TODO add smarter padding making this unnessasary
    #cstate = cstate[:,:self.sim_cshape[0],:self.sim_cshape[1]]

    return cstate

  def boundary_to_cboundary(self, encoder, encoder_shape_converter):

    nr_subdomains = [int(math.ceil(x/float(y))) for x, y in zip(self.sim_cshape, self.input_cshape)]
    cboundary = []
    for i, j in itertools.product(xrange(nr_subdomains[0]), xrange(nr_subdomains[1])):
      pos = [i * self.input_cshape[0], j * self.input_cshape[1]]
      subdomain = SubDomain(pos, self.input_cshape)
      input_subdomain = encoder_shape_converter.out_in_subdomain(subdomain)
      h = np.mgrid[input_subdomain.pos[0]:input_subdomain.pos[0] + input_subdomain.size[0],
                   input_subdomain.pos[1]:input_subdomain.pos[1] + input_subdomain.size[1]]
      hx = np.mod(h[1], self.sim_shape[0])
      hy = np.mod(h[0], self.sim_shape[1])
      where_boundary = self.geometry_boundary_conditions(hx, hy, self.sim_shape)
      where_velocity, velocity = self.velocity_boundary_conditions(hx, hy, self.sim_shape)
      where_density, density = self.density_boundary_conditions(hx, hy, self.sim_shape)
      input_geometry = self.make_geometry_input(where_boundary, velocity, where_velocity, density, where_density)
      input_geometry = np.expand_dims(input_geometry, axis=0)
      """
      plt.imshow(hx[:,:])
      plt.savefig('figs/foo_4.png')
      plt.imshow(hy[:,:])
      plt.savefig('figs/foo_5.png')
      plt.imshow(input_geometry[0,:,:,0])
      plt.savefig('figs/foo_0.png')
      plt.imshow(input_geometry[0,:,:,1])
      plt.savefig('figs/foo_1.png')
      plt.imshow(input_geometry[0,:,:,2])
      plt.savefig('figs/foo_2.png')
      plt.imshow(input_geometry[0,:,:,3])
      plt.savefig('figs/foo_3.png')
      #plt.show()
      """
      cboundary.append(encoder(input_geometry))

    # list to full tensor
    cboundary = numpy_utils.stack_grid(cboundary, nr_subdomains, has_batch=True)

    # trim edges TODO add smarter padding making this unnessasary
    #cboundary = cboundary[:,:self.sim_cshape[0],:self.sim_cshape[1]]

    return cboundary

  def cstate_to_cstate(self, cmapping, cmapping_shape_converter, cstate, cboundary):

    nr_subdomains = [int(math.ceil(x/float(y))) for x, y in zip(self.sim_cshape, self.input_cshape)]
    new_cstate = []
    for i, j in itertools.product(xrange(nr_subdomains[0]), xrange(nr_subdomains[1])):
      pos = [i * self.input_cshape[0], j * self.input_cshape[1]]
      subdomain = SubDomain(pos, self.input_cshape)
      input_subdomain = cmapping_shape_converter.out_in_subdomain(copy(subdomain))
      new_cstate.append(cmapping(numpy_utils.mobius_extract(cstate, input_subdomain, has_batch=True),
                                 numpy_utils.mobius_extract(cboundary, subdomain, has_batch=True )))

    # list to full tensor
    new_cstate = numpy_utils.stack_grid(new_cstate, nr_subdomains, has_batch=True)

    # trim edges TODO add smarter padding making this unnessasary
    new_cstate = new_cstate[:,:self.sim_cshape[0],:self.sim_cshape[1]]

    return new_cstate

  def cstate_to_state(self, decoder, decoder_shape_converter, cstate):

    nr_subdomains = [int(math.ceil(x/float(y))) for x, y in zip(self.sim_shape, self.input_shape)]
    state = []
    for i, j in itertools.product(xrange(nr_subdomains[0]), xrange(nr_subdomains[1])):
      pos = [i * self.input_shape[0], j * self.input_shape[1]]
      subdomain = SubDomain(pos, self.input_shape)
      input_subdomain = decoder_shape_converter.out_in_subdomain(copy(subdomain))
      output_subdomain = decoder_shape_converter.in_out_subdomain(copy(input_subdomain))
      state_store = decoder(numpy_utils.mobius_extract(cstate, input_subdomain, has_batch=True))
      left_pad  = [x - y for x, y in zip(subdomain.pos, output_subdomain.pos)]
      state_store = state_store[:,left_pad[0]:,left_pad[1]:,:]
      state_store = state_store[:,:self.input_shape[0],:self.input_shape[1],:]
      state.append(state_store)

    # list to full tensor
    state = numpy_utils.stack_grid(state, nr_subdomains, has_batch=True)

    # trim edges TODO add smarter padding making this unnessasary
    #state = state[:,:self.sim_shape[0],:self.sim_shape[1]]

    return state

