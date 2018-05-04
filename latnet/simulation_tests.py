
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
from simulation import Simulation
from sailfish_simulation import SailfishSimulation
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
import matplotlib.pyplot as plt

# this class is used to compare the compression error in various ways. It is built ontop of the simulation class.
class CompressionError(Simulation):
  simulation_length = 100

  def run(self):

    # unroll network
    self._network = self.network(self.config)
    (state_encoder, boundary_encoder, cmapping, cmapping_first, decoder_vel_rho,
      decoder_state, encoder_shape_converter, cmapping_shape_converter, 
      decoder_shape_converter) = self._network.eval_unroll()

    # run real simulation to get data for compression
    self.sailfish_runner = SailfishSimulation(self.config, self.domain, self.sim_dir + '/sailfish')
    self.sailfish_runner.new_sim(self.simulation_length)

    state_l2 = np.zeros((self.simulation_length-1))
    vel_l2 = np.zeros((self.simulation_length-1))
    rho_l2 = np.zeros((self.simulation_length-1))
    state_normalized = np.zeros((self.simulation_length-1))
    vel_normalized = np.zeros((self.simulation_length-1))
    rho_normalized = np.zeros((self.simulation_length-1))
    for i in tqdm(range(1, self.simulation_length)):
      self.start_state = self.sailfish_runner.read_state(i)
      self.start_boundary = self.sailfish_runner.read_boundary()

      # generate compressed state
      cstate    = self.state_to_cstate(state_encoder, 
                                       encoder_shape_converter)
      cboundary = self.boundary_to_cboundary(boundary_encoder, 
                                             encoder_shape_converter)

      # iterate 1 step
      cstate_new = self.cstate_to_cstate(cmapping_first, 
                                     cmapping_shape_converter, 
                                     cstate, cboundary)
      # decode state
      pred_state = self.cstate_to_state(decoder_state, 
                                        decoder_shape_converter, 
                                        cstate_new,
                                        cboundary)[0]

      # compare true state ves compressed and decoded state
      true_state = self.sailfish_runner.read_state(i+1)
      pred_vel = self.DxQy.lattice_to_vel(pred_state)
      pred_rho = self.DxQy.lattice_to_rho(pred_state)
      true_vel = self.DxQy.lattice_to_vel(true_state)
      true_rho = self.DxQy.lattice_to_rho(true_state)
      
      # store values
      state_l2[i-1] = numpy_l2_loss(true_state, pred_state)
      vel_l2[i-1] =  numpy_l2_loss(true_vel, pred_vel)
      rho_l2[i-1] =  numpy_l2_loss(true_rho, pred_rho)
      state_normalized[i-1] =  numpy_normalized_loss(true_state, pred_state)
      vel_normalized[i-1] = numpy_normalized_loss(true_vel, pred_vel)
      rho_normalized[i-1] = numpy_normalized_loss(true_rho, pred_rho)

    plt.plot(state_l2, label='state l2')
    plt.plot(vel_l2, label='vel l2')
    plt.plot(rho_l2, label='rho l2')
    plt.legend()
    plt.savefig("figs/error_plot_l2.pdf")

    plt.plot(state_normalized, label='state normalized l2')
    plt.plot(vel_normalized, label='vel normalized l2')
    plt.plot(rho_normalized, label='rho normalized l2')
    plt.legend()
    plt.savefig("figs/error_plot_normalize.pdf")
 
def numpy_l2_loss(true, generated):
  return np.square(np.linalg.norm(true-generated))

def numpy_normalized_loss(true, generated, norm="var"):

  if norm == "zero_one":
    max_true = np.max(true, axis=(0,1))
    min_true = np.min(true, axis=(0,1))
    true      = (true - min_true)/(max_true - min_true)
    generated = (generated - min_true)/(max_true - min_true)

  elif norm == "var":
    std = true.std(axis=(0,1))
    true      = true/std
    generated = generated/std

  return numpy_l2_loss(true, generated)



