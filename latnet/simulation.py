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
from sailfish.node_type import NTEquilibriumVelocity, NTFullBBWall, NTDoNothing
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


class Simulation(object):

  def __init__(self, config, domain):

    # domain of simulation
    self.domain = domain

    # shapes
    self.sim_shape = sim_shape[0])
    # needed configs
    self.config = config
    self.DxQy = lattice.TYPES[config.DxQy]()
    self.sim_dir = config.sim_dir
    self.num_iters = config.num_iters
    self.sim_save_every = config.sim_save_every
    self.sim_restore_iter = config.sim_restore_iter
    self.compare = config.compare

    # make visualizer
    self.vis = Visualizations(self.config)

    # make saver
    self.saver = SimSaver(self.config)

  def run(self, state_encoder, 
          boundary_encoder, 
          cmapping, decoder,
          encoder_shape_converter, 
          cmapping_shape_converter, 
          decoder_shape_converter):

    # possibly generate start state and boundary (really just used for testing)
    if self.sim_restore_iter > 0:
      self.sailfish_runner = SailfishSimulation(self.config, self.sim_dir + '/sailfish', self.script_name)
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

  def input_boundary(self, input_subdomain):
    h = np.mgrid[input_subdomain.pos[0]:input_subdomain.pos[0] + input_subdomain.size[0],
                 input_subdomain.pos[1]:input_subdomain.pos[1] + input_subdomain.size[1]]
    hx = np.mod(h[1], self.sim_shape[0])
    hy = np.mod(h[0], self.sim_shape[1])
    where_boundary = self.geometry_boundary_conditions(hx, hy, self.sim_shape)
    where_velocity, velocity = self.velocity_boundary_conditions(hx, hy, self.sim_shape)
    where_density, density = self.density_boundary_conditions(hx, hy, self.sim_shape)
    input_geometry = self.make_geometry_input(where_boundary, velocity, where_velocity, density, where_density)
    return input_geometry

  def mapping(self, mapping, shape_converter, input_generator, output_shape, run_output_shape):
    nr_subdomains = [int(math.ceil(x/float(y))) for x, y in zip(output_shape, run_output_shape)]
    output = []
    for ijk in itertools.product(xrange(nr_subdomains[0]), xrange(nr_subdomains[1])):
      # make input and output subdomains
      pos = [x * y for x, y in zip(ijk, max_output_shape)]
      subdomain = SubDomain(pos, max_output_shape)
      input_subdomain = shape_converter.out_in_subdomain(copy(subdomain))
      output_subdomain = shape_converter.in_out_subdomain(copy(input_subdomain))

      # generate input with input generator
      sub_input = input_generator(input_subdomain)

      # perform mapping function and extract out if needed
      if not (type(sub_input) is list):
        sub_output = [sub_input]
      sub_output = mapping(*sub_input)
      output_subdomain.zero_pos()
      sub_output = numpy_utils.extract(sub_output)

      # append to list of sub outputs
      output.append(sub_output)

    # stack back together to form one output
    output = numpy_utils.stack_grid(output, nr_subdomains, has_batch=True)
    return output

  def state_to_cstate(self, encoder, encoder_shape_converter):

    def input_generator(subdomain):
      if self.start_state is not None:
        start_state = numpy_utils.mobius_extract(self.start_state, subdomain, has_batch=False)
        start_state = np.expand_dims(start_state, axis=0)
      else:
        vel = self.domain.velocity_initial_conditions(0,0,None)
        feq = self.DxQy.vel_to_feq(vel).reshape([1] + self.DxQy.dims*[1] + [self.DxQy.Q])
        start_state = np.zeros([1] + subdomain.size + [self.DxQy.Q]) + feq
      return start_state 

    cstate = self.mapping(encoder, encoder_shape_converter, 
                          input_generator, self.sim_cshape, 
                          self.input_cshape)
    return cstate

  def boundary_to_cboundary(self, encoder, encoder_shape_converter):

    def input_generator(subdomain):
      if self.start_boundary is not None:
        input_geometry = numpy_utils.mobius_extract(self.start_boundary, input_subdomain, has_batch=False)
      else:
        input_geometry = self.input_boundary(input_subdomain)
      return input_geometry

    cboundar = self.mapping(encoder, encoder_shape_converter, 
                            input_generator, self.sim_cshape, 
                            self.input_cshape)
    return cboundary

  def cstate_to_cstate(self, cmapping, cmapping_shape_converter, cstate, cboundary):

    def input_generator(subdomain):
      sub_cstate =    numpy_utils.mobius_extract(cstate,    subdomain, has_batch=True)
      sub_cboundary = numpy_utils.mobius_extract(cboundary, subdomain, has_batch=True)
      return sub_cstate, sub_cboundary

    cstate = self.mapping(cmapping, cmapping_shape_converter, 
                          input_generator, self.sim_cshape, 
                          self.input_cshape)
    return cstate

  def cstate_to_state(self, decoder, decoder_shape_converter, cstate):

    def input_generator(subdomain):
      sub_cstate = numpy_utils.mobius_extract(cstate, subdomain, has_batch=True)
      return sub_cstate

    vel_rho = self.mapping(decoder, decoder_shape_converter, 
                            input_generator, self.sim_shape, 
                            self.input_shape)
    vel, rho = vel_rho[...,:-1], vel_rho[...,-1:]
    return vel, rho

