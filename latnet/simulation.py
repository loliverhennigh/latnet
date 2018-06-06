
import sys
import os
import time
from termcolor import colored, cprint

import matplotlib.pyplot as plt


# import latnet files
import utils.numpy_utils as numpy_utils
from utils.python_utils import *
from shape_converter import SubDomain
from vis import Visualizations
from sim_saver import SimSaver
from sailfish_simulation import SailfishSimulation
from john_hopkins_simulation import JohnHopkinsSimulation
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

  def __init__(self, config):

    # domain
    self._domain = self.domain(config)

    # shapes
    self.sim_shape = self.domain.sim_shape
    self.sim_cshape = [x/pow(2,config.nr_downsamples) for x in self.sim_shape]
    self.input_shape = str2shape(config.input_shape)
    self.input_cshape = str2shape(config.input_cshape)

    # needed configs
    self.dataset         = config.dataset
    self.config = config
    self.DxQy = lattice.TYPES[config.DxQy]()
    self.sim_dir = config.sim_dir
    self.num_iters = config.num_iters
    self.sim_save_every = config.sim_save_every
    self.sim_restore_iter = config.sim_restore_iter
    self.compare = config.compare
    self.domain.script_name = self.script_name

    # run time holders
    self.print_stats_every = 5
    self.time_stats = {}
    self.start_time = time.time()
    self.tic = time.time()
    self.toc = time.time()

    # make visualizer
    self.vis = Visualizations(self.config, self.sim_shape)

    # make saver
    self.saver = SimSaver(self.config)

    # set padding 
    self.padding_type = ['zero', 'zero']
    if self.domain.periodic_x:
      self.padding_type[0] = 'periodic'
    if self.domain.periodic_y:
      self.padding_type[1] = 'periodic'
    if self.domain.periodic_z:
      self.padding_type.append('periodic')

  def run(self):

    # unroll network
    self._network = self.network(self.config)
    (state_encoder, boundary_encoder, cmapping, cmapping_first, decoder_vel_rho,
      decoder_state, encoder_shape_converter, cmapping_shape_converter, 
      decoder_shape_converter) = self._network.eval_unroll()

    # possibly generate start state and boundary (really just used for testing)
    if self.sim_restore_iter > 0:
      if self.dataset == 'sailfish':
        self.sailfish_runner = SailfishSimulation(self.config, self.domain, self.sim_dir + '/sailfish')
        self.sailfish_runner.new_sim(self.sim_restore_iter)
        self.start_state = self.sailfish_runner.read_state(self.sim_restore_iter)
        self.start_boundary = self.sailfish_runner.read_boundary()
      elif self.dataset == 'JHTDB':
        self.jhtdb_runner = JohnHopkinsSimulation(self.config, self.sim_dir + '/JHTDB')
        subdomain = SubDomain([0, 0, 0], self.domain.sim_shape)
        self.jhtdb_runner.download_datapoint(subdomain, 0)
        self.start_state = self.jhtdb_runner.read_state(0, subdomain=subdomain, return_padding=False)
        self.start_boundary = self.jhtdb_runner.read_boundary(subdomain=subdomain, return_padding=False)
        
    else:
      self.sailfish_runner = None
      self.start_state = None
      self.start_boundary = None

    # generate compressed state
    print("computeing compressed state")
    cstate    = self.state_to_cstate(state_encoder, 
                                     encoder_shape_converter)
    self.start_state = None
    cboundary = self.boundary_to_cboundary(boundary_encoder, 
                                           encoder_shape_converter)
    self.start_boundary = None

    # run simulation
    for i in xrange(self.num_iters):

      if i % self.print_stats_every == 0:
        # print time states
        self.update_time_stats()
        self.print_stats(self.time_stats, i)

      if (i+1) % self.sim_save_every == 0:
        # decode state
        vel, rho = self.cstate_to_vel_rho(decoder_vel_rho, 
                                          decoder_shape_converter, 
                                          cstate,
                                          cboundary)

        # vis and save
        self.vis.update_vel_rho(i, vel, rho)
        self.saver.save(i, vel, rho, cstate)
        vel = None
        rho = None

      if i == 0:
        cstate = self.cstate_to_cstate(cmapping_first, 
                                       cmapping_shape_converter, 
                                       cstate, cboundary)
      else:
        cstate = self.cstate_to_cstate(cmapping, 
                                       cmapping_shape_converter, 
                                       cstate, cboundary)

    # generate comparision simulation
    if self.compare:
      if self.dataset == 'sailfish':
        if self.sailfish_runner is not None:
          self.sailfish_runner.restart_sim(self.num_iters)
        else:
          self.sailfish_runner = SailfishSimulation(self.config, self.domain, self.sim_dir + '/sailfish')
          self.sailfish_runner.new_sim(self.num_iters)
      elif self.dataset == 'JHTDB':
        print("Downloading compare data of JHTDB")
        for i in tqdm(xrange(self.num_iters+3)):
          subdomain = SubDomain([0, 0, 0], self.domain.sim_shape)
          self.jhtdb_runner.download_datapoint(subdomain, i)

      # run comparision function
      print("Comparing network generated and true")
      for i in tqdm(xrange(self.num_iters)):
        if i % self.sim_save_every == 0:
          # this functionality will probably be changed TODO
          if self.dataset == 'sailfish':
            true_vel, true_rho = self.sailfish_runner.read_vel_rho(i + self.sim_restore_iter + 1, add_batch=True)
            generated_vel, generated_rho = self.saver.read_vel_rho(i, add_batch=True)
            self.vis.update_compare_vel_rho(i, true_vel, true_rho, generated_vel, generated_rho)
          elif self.dataset == 'JHTDB':
            subdomain = SubDomain([0, 0, 0], self.domain.sim_shape)
            true_vel, true_rho = self.jhtdb_runner.read_vel_rho(i + self.sim_restore_iter + 1, subdomain=subdomain, add_batch=True)
            generated_vel, generated_rho = self.saver.read_vel_rho(i, add_batch=True)
            self.vis.update_compare_vel_rho(i, true_vel, true_rho, generated_vel, generated_rho)

  def input_boundary(self, input_subdomain):
    h = np.mgrid[input_subdomain.pos[0]:input_subdomain.pos[0] + input_subdomain.size[0],
                 input_subdomain.pos[1]:input_subdomain.pos[1] + input_subdomain.size[1]]
    hx = np.mod(h[1], self.sim_shape[0])
    hy = np.mod(h[0], self.sim_shape[1])
    where_boundary = self._domain.geometry_boundary_conditions(hx, hy, self.sim_shape)
    where_velocity, velocity = self._domain.velocity_boundary_conditions(hx, hy, self.sim_shape)
    where_density, density = self._domain.density_boundary_conditions(hx, hy, self.sim_shape)
    input_geometry = self._domain.make_geometry_input(where_boundary, velocity, where_velocity, density, where_density)
    return input_geometry

  def mapping(self, mapping, shape_converter, input_generator, output_shape, run_output_shape):
    nr_subdomains = [int(math.ceil(x/float(y))) for x, y in zip(output_shape, run_output_shape)]
    output = []
    iter_list = [xrange(x) for x in nr_subdomains]
    for ijk in itertools.product(*iter_list):
      print(str(ijk) + " out of " + str(nr_subdomains))
      # make input and output subdomains
      if not(type(shape_converter) is list):
        shape_converter = [shape_converter]
      input_subdomain = []
      output_subdomain = []
      for converter in shape_converter:
        pos = [x * y for x, y in zip(ijk, run_output_shape)]
        subdomain = SubDomain(pos, run_output_shape)
        input_subdomain.append(converter.out_in_subdomain(copy(subdomain)))
        output_subdomain.append(converter.in_out_subdomain(copy(input_subdomain[-1])))
      output_subdomain = output_subdomain[0]
      output_subdomain.zero_pos()

      # generate input with input generator
      sub_input = input_generator(*input_subdomain)

      # perform mapping function and extract out if needed
      if not (type(sub_input) is list):
        sub_input = [sub_input]
      sub_output = mapping(*sub_input)
      if not (type(sub_output) is list):
        sub_output = [sub_output]

      for i in xrange(len(sub_output)):
        sub_output[i] = numpy_utils.mobius_extract_2(sub_output[i], output_subdomain, 
                                                     has_batch=True)

      # append to list of sub outputs
      output.append(sub_output)

    # make total output shape
    total_subdomain = SubDomain(len(output_shape)*[0], output_shape)
    ctotal_subdomain = shape_converter[0].out_in_subdomain(copy(total_subdomain))
    total_subdomain  = shape_converter[0].in_out_subdomain(copy(ctotal_subdomain))
    total_subdomain = SubDomain([-x for x in total_subdomain.pos], output_shape)

    # stack back together to form one output
    output = llist2list(output)
    for i in xrange(len(output)):
      output[i] = numpy_utils.stack_grid(output[i], nr_subdomains, has_batch=True)
      output[i] = numpy_utils.mobius_extract_2(output[i], total_subdomain, 
                                               has_batch=True)
    if len(output) == 1:
      output = output[0]
    return output

  def state_to_cstate(self, encoder, encoder_shape_converter):

    def input_generator(subdomain):
      if self.start_state is not None:
        tic = time.time()
        start_state, pad_start_state = numpy_utils.mobius_extract_2(self.start_state, 
                                                                    subdomain, 
                                                                    has_batch=False, 
                                                                    padding_type=self.padding_type,
                                                                    return_padding=True)
      else:
        vel = self._domain.velocity_initial_conditions(0,0,None)
        feq = self.DxQy.vel_to_feq(vel).reshape([1] + self.DxQy.dims*[1] + [self.DxQy.Q])
        start_state = np.zeros([1] + subdomain.size + [self.DxQy.Q]) + feq
      start_state     = np.expand_dims(start_state, axis=0)
      pad_start_state = np.expand_dims(pad_start_state, axis=0)
      return (start_state, pad_start_state)

    cstate = self.mapping(encoder, encoder_shape_converter, 
                          input_generator, self.sim_cshape, 
                          self.input_cshape)
    return cstate

  def boundary_to_cboundary(self, encoder, encoder_shape_converter):

    def input_generator(subdomain):
      if self.start_boundary is not None:
        tic = time.time()
        input_geometry, pad_input_geometry = numpy_utils.mobius_extract_2(self.start_boundary, 
                                                                        subdomain,
                                                                        has_batch=False, 
                                                                        padding_type=self.padding_type,
                                                                        return_padding=True)
      else:
        input_geometry = self.input_boundary(subdomain)
      input_geometry     = np.expand_dims(input_geometry, axis=0)
      pad_input_geometry = np.expand_dims(pad_input_geometry, axis=0)
      return (input_geometry, pad_input_geometry)

    cboundary = self.mapping(encoder, encoder_shape_converter, 
                            input_generator, self.sim_cshape, 
                            self.input_cshape)
    return cboundary

  def cstate_to_cstate(self, cmapping, cmapping_shape_converter, cstate, cboundary):

    def input_generator(cstate_subdomain, cboundary_subdomain):
      sub_cstate =    numpy_utils.mobius_extract_2(cstate, cstate_subdomain, 
                                                 has_batch=True, 
                                                 padding_type=self.padding_type,
                                                 return_padding=True)
      sub_cboundary = numpy_utils.mobius_extract_2(cboundary, cboundary_subdomain, 
                                                 has_batch=True, 
                                                 padding_type=self.padding_type,
                                                 return_padding=True)
      return [sub_cstate, sub_cboundary]

    cstate = self.mapping(cmapping, [cmapping_shape_converter, cmapping_shape_converter],
                          input_generator, self.sim_cshape, 
                          self.input_cshape)
    return cstate

  def cstate_to_state(self, decoder, decoder_shape_converter, cstate, cboundary):

    def input_generator(cstate_subdomain, cboundary_subdomain):
      sub_cstate    = numpy_utils.mobius_extract_2(cstate,    cstate_subdomain,    
                                                 has_batch=True, 
                                                 padding_type=self.padding_type,
                                                 return_padding=True)
      sub_cboundary = numpy_utils.mobius_extract_2(cboundary, cboundary_subdomain, 
                                                 has_batch=True,
                                                 padding_type=self.padding_type,
                                                 return_padding=True)
      return [sub_cstate, sub_cboundary]

    state = self.mapping(decoder, [decoder_shape_converter, decoder_shape_converter], 
                              input_generator, self.sim_shape, 
                              self.input_shape)
    return state

  def cstate_to_vel_rho(self, decoder, decoder_shape_converter, cstate, cboundary):

    def input_generator(cstate_subdomain, cboundary_subdomain):
      sub_cstate    = numpy_utils.mobius_extract_2(cstate,    cstate_subdomain,    
                                                 has_batch=True, 
                                                 padding_type=self.padding_type,
                                                 return_padding=True)
      sub_cboundary = numpy_utils.mobius_extract_2(cboundary, cboundary_subdomain, 
                                                 has_batch=True,
                                                 padding_type=self.padding_type,
                                                 return_padding=True)
      return [sub_cstate, sub_cboundary]

    [vel, rho] = self.mapping(decoder, [decoder_shape_converter, decoder_shape_converter], 
                              input_generator, self.sim_shape, 
                              self.input_shape)
    return vel, rho

  def update_time_stats(self):
    # stop timer
    self.toc = time.time()
    # update total run time
    self.time_stats['run_time'] = int(time.time() - self.start_time)
    # update total step time
    self.time_stats['MLOPS'] = (self.config.lb_to_ln*self.print_stats_every*float(np.prod(np.array(self.sim_shape))) / 
                               (1000000*(self.toc - self.tic)))
    # start timer
    self.tic = time.time()


  def print_stats(self, time_stats, step):
    time_string = print_dict('TIME STATS', time_stats, 'magenta')
    print_string = time_string
    os.system('clear')
    print("EVAL INFO - step " + str(step))
    print(print_string)

