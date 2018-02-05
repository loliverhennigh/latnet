
import sys
import time

import matplotlib.pyplot as plt


# import latnet files
import utils.numpy_utils as numpy_utils
from shape_converter import SubDomain
from vis import Visualizations
from sim_saver import SimSaver
from sailfish_runner import SailfishRunner
import lattice

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

  def __init__(self, config):

    sim_shape = config.sim_shape.split('x')
    sim_shape = map(int, sim_shape)
    self.sim_shape = sim_shape

    downsample_factor = pow(2, config.nr_downsamples)
    self.sim_cshape = [x/downsample_factor for x in sim_shape] 

    input_shape = config.input_shape.split('x')
    input_shape = map(int, input_shape)
    self.input_shape = input_shape

    input_cshape = config.input_cshape.split('x')
    input_cshape = map(int, input_cshape)
    self.input_cshape = input_cshape

    self.config = config
    self.train_sim_dir = config.train_sim_dir
    self.max_sim_iters = config.max_sim_iters
    self.lb_to_ln = config.lb_to_ln
    self.visc = config.visc
    self.DxQy = lattice.TYPES[config.DxQy]()
    self.restore_geometry = config.restore_geometry
    self.sim_dir = config.sim_dir
    self.num_iters = config.num_iters
    self.sim_save_every = config.sim_save_every
    self.sim_restore_iter = config.sim_restore_iter
    self.compare = config.compare

  @classmethod
  def update_defaults(cls, defaults):
      pass


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

  def compare_script(self, iteration, true_vel, true_rho, generated_vel, generated_rho):
    pass

  def make_geometry_input(self, where_boundary, velocity, where_velocity, density, where_density):
    # TODO Clean this
    input_geometry = np.concatenate([np.expand_dims(where_boundary, axis=-1).astype(np.float32),
                                     np.array(velocity).reshape(len(where_velocity.shape) * [1] + [2]) 
                                       * np.expand_dims(where_velocity, axis=-1).astype(np.float32),
                                     density 
                                       *  np.expand_dims(where_density, axis=-1).astype(np.float32)], axis=-1)
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
    train_sim_dir = self.train_sim_dir
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
          restore_boundary_conditions = np.load(train_sim_dir[:-10] + "flow_geometry.npy")
          where_boundary = restore_boundary_conditions[...,0].astype(np.bool)
          where_velocity = np.logical_or(restore_boundary_conditions[...,1].astype(np.bool), restore_boundary_conditions[...,1].astype(np.bool))
          velocity = (restore_boundary_conditions[np.where(where_velocity)[0][0], np.where(where_velocity)[1][0], 1],
                      restore_boundary_conditions[np.where(where_velocity)[0][0], np.where(where_velocity)[1][0], 2])
          print(velocity)
          where_density  = restore_boundary_conditions[...,3].astype(np.bool)
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
        np.save(train_sim_dir + "_geometry.npy", save_geometry)

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
        group.add_argument('--train_sim_dir', help='all modes', type=str,
                              default='')
        group.add_argument('--sim_dir', help='all modes', type=str,
                              default='')
        group.add_argument('--run_mode', help='all modes', type=str,
                              default='')
        group.add_argument('--max_sim_iters', help='all modes', type=int,
                              default=1000)
        group.add_argument('--restore_geometry', help='all modes', type=bool,
                              default=False)
        group.add_argument('--sim_shape', help='all modes', type=str,
                              default='512x512')
        group.add_argument('--lb_to_ln', help='all modes', type=int,
                              default=60)

      @classmethod
      def update_defaults(cls, defaults):
        defaults.update({
          'max_iters': max_iters,
          'output_format': 'npy',
          'periodic_y': True,
          'periodic_x': True,
          'checkpoint_file': train_sim_dir,
          'checkpoint_every': lb_to_ln,
          'lat_nx': shape[0],
          'lat_ny': shape[1]
          })
        if len(shape) == 3:
          defaults.update({
            'periodic_z': True,
            'lat_nz': shape[0],
            'grid': 'D3Q15'
          })

      @classmethod
      def modify_config(cls, config):
        config.visc   = visc

      def __init__(self, *args, **kwargs):
        super(SailfishSimulation, self).__init__(*args, **kwargs)

    ctrl = LBSimulationController(SailfishSimulation)

    return ctrl

  def run(self, state_encoder, 
          boundary_encoder, 
          cmapping, decoder,
          encoder_shape_converter, 
          cmapping_shape_converter, 
          decoder_shape_converter):

    # make visualizer
    self.vis = Visualizations(self.config)

    # make saver
    self.saver = SimSaver(self.config)

    # possibly generate start state and boundary (really just used for testing)
    if self.sim_restore_iter > 0:
      self.sailfish_runner = SailfishRunner(self.config, self.sim_dir + '/sailfish', self.script_name)
      self.sailfish_runner.new_sim(self.sim_restore_iter)
      self.start_state = self.sailfish_runner.read_state(self.sim_restore_iter)
      self.start_boundary = self.sailfish_runner.read_boundary()
    else:
      self.sailfish_runner = None
      self.start_state = None
      self.start_boundary = None

    # generate compressed state
    cstate    = self.state_to_cstate(state_encoder, 
                                     encoder_shape_converter)
    cboundary = self.boundary_to_cboundary(boundary_encoder, 
                                           encoder_shape_converter)

    # run simulation
    for i in tqdm(xrange(self.num_iters)):
      if i % self.sim_save_every == 0:
        # decode state
        vel, rho = self.cstate_to_state(decoder, 
                                        decoder_shape_converter, 
                                        cstate)
        # vis and save
        self.vis.update_vel_rho(i, vel, rho)
        self.saver.save(i, vel, rho, cstate)

      cstate = self.cstate_to_cstate(cmapping, 
                                     cmapping_shape_converter, 
                                     cstate, cboundary)

    # generate comparision simulation
    if self.compare:
      if self.sailfish_runner is not None:
        self.sailfish_runner.restart_sim(self.num_iters)
      else:
        self.sailfish_runner = SailfishRunner(self.config, self.sim_dir + '/sailfish', self.script_name)
        self.sailfish_runner.new_sim(self.num_iters)

      # run comparision function
      for i in xrange(self.num_iters):
        if i % self.sim_save_every == 0:
          # this functionality will probably be changed TODO
          true_vel, true_rho = self.sailfish_runner.read_vel_rho(i + self.sim_restore_iter + 1)
          generated_vel, generated_rho = self.saver.read_vel_rho(i)
          self.compare_script(i, true_vel, true_rho, generated_vel, generated_rho)

  def state_to_cstate(self, encoder, encoder_shape_converter):

    nr_subdomains = [int(math.ceil(x/float(y))) for x, y in zip(self.sim_cshape, self.input_cshape)]
    cstate = []
    for i, j in itertools.product(xrange(nr_subdomains[0]), xrange(nr_subdomains[1])):
      pos = [i * self.input_cshape[0], j * self.input_cshape[1]]
      subdomain = SubDomain(pos, self.input_cshape)
      input_subdomain = encoder_shape_converter.out_in_subdomain(subdomain)
      if self.start_state is not None:
        start_state = numpy_utils.mobius_extract(self.start_state, input_subdomain, has_batch=False)
        input_geometry = numpy_utils.mobius_extract(self.start_boundary, input_subdomain, has_batch=False)
        start_state = np.concatenate([start_state, input_geometry], axis=-1)
        start_state = np.expand_dims(start_state, axis=0)
      else:
        vel = self.velocity_initial_conditions(0,0,None)
        feq = self.DxQy.vel_to_feq(vel).reshape([1] + self.DxQy.dims*[1] + [self.DxQy.Q])
        start_state = np.zeros([1] + input_subdomain.size + [self.DxQy.Q]) + feq
      cstate.append(encoder(start_state))

    # list to full tensor
    cstate = numpy_utils.stack_grid(cstate, nr_subdomains, has_batch=True)

    return cstate

  def boundary_to_cboundary(self, encoder, encoder_shape_converter):

    #nr_subdomains = [xrange(int(math.ceil(x/float(y)))) for x, y in zip(self.sim_cshape, self.input_cshape)]
    #for ind in itertools.product(*nr_subdomains):
      #pos = [x * y for x, y in zip(ind, self.input_cshape)]
    nr_subdomains = [int(math.ceil(x/float(y))) for x, y in zip(self.sim_cshape, self.input_cshape)]
    cboundary = []
    for i, j in itertools.product(xrange(nr_subdomains[0]), xrange(nr_subdomains[1])):
      pos = [i * self.input_cshape[0], j * self.input_cshape[1]]
      subdomain = SubDomain(pos, self.input_cshape)
      input_subdomain = encoder_shape_converter.out_in_subdomain(subdomain)
      if self.start_boundary is not None:
        input_geometry = numpy_utils.mobius_extract(self.start_boundary, input_subdomain, has_batch=False)
      else:
        h = np.mgrid[input_subdomain.pos[0]:input_subdomain.pos[0] + input_subdomain.size[0],
                     input_subdomain.pos[1]:input_subdomain.pos[1] + input_subdomain.size[1]]
        hx = np.mod(h[1], self.sim_shape[0])
        hy = np.mod(h[0], self.sim_shape[1])
        where_boundary = self.geometry_boundary_conditions(hx, hy, self.sim_shape)
        where_velocity, velocity = self.velocity_boundary_conditions(hx, hy, self.sim_shape)
        where_density, density = self.density_boundary_conditions(hx, hy, self.sim_shape)
        input_geometry = self.make_geometry_input(where_boundary, velocity, where_velocity, density, where_density)
      input_geometry = np.expand_dims(input_geometry, axis=0)
      cboundary.append(encoder(input_geometry))

    # list to full tensor
    cboundary = numpy_utils.stack_grid(cboundary, nr_subdomains, has_batch=True)

    return cboundary

  def cstate_to_cstate(self, cmapping, cmapping_shape_converter, cstate, cboundary):

    nr_subdomains = [int(math.ceil(x/float(y))) for x, y in zip(self.sim_cshape, self.input_cshape)]
    new_cstate = []
    for i, j in itertools.product(xrange(nr_subdomains[0]), xrange(nr_subdomains[1])):
      pos = [i * self.input_cshape[0], j * self.input_cshape[1]]
      subdomain = SubDomain(pos, self.input_cshape)
      input_subdomain = cmapping_shape_converter.out_in_subdomain(copy(subdomain))
      new_cstate.append(cmapping(numpy_utils.mobius_extract(cstate, copy(input_subdomain), has_batch=True),
                                 numpy_utils.mobius_extract(cboundary, copy(input_subdomain), has_batch=True )))

    # list to full tensor
    new_cstate = numpy_utils.stack_grid(new_cstate, nr_subdomains, has_batch=True)

    return new_cstate

  def cstate_to_state(self, decoder, decoder_shape_converter, cstate):

    nr_subdomains = [int(math.ceil(x/float(y))) for x, y in zip(self.sim_shape, self.input_shape)]
    vel = []
    rho = []
    for i, j in itertools.product(xrange(nr_subdomains[0]), xrange(nr_subdomains[1])):
      pos = [i * self.input_shape[0], j * self.input_shape[1]]
      subdomain = SubDomain(pos, self.input_shape)
      input_subdomain = decoder_shape_converter.out_in_subdomain(copy(subdomain))
      output_subdomain = decoder_shape_converter.in_out_subdomain(copy(input_subdomain))
      vel_store, rho_store = decoder(numpy_utils.mobius_extract(cstate, input_subdomain, has_batch=True))
      left_pad  = [x - y for x, y in zip(subdomain.pos, output_subdomain.pos)]
      vel_store = vel_store[:,left_pad[0]:,left_pad[1]:,:]
      vel_store = vel_store[:,:self.input_shape[0],:self.input_shape[1],:]
      vel.append(vel_store)
      rho_store = rho_store[:,left_pad[0]:,left_pad[1]:,:]
      rho_store = rho_store[:,:self.input_shape[0],:self.input_shape[1],:]
      rho.append(rho_store)

    # list to full tensor
    vel = numpy_utils.stack_grid(vel, nr_subdomains, has_batch=True)
    rho = numpy_utils.stack_grid(rho, nr_subdomains, has_batch=True)

    # trim edges TODO add smarter padding making this unnessasary
    vel = vel[:,:self.sim_shape[0],:self.sim_shape[1]]
    rho = rho[:,:self.sim_shape[0],:self.sim_shape[1]]

    return vel, rho

