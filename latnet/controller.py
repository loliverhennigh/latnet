
import sys

import tensorflow as tf

from latnetwork import LatNet
from data_queue import DataQueue
from config import LatNetConfigParser
from domain import Domain
from sim_saver import SimSaver
from sailfish_simulation import SailfishSimulation
from utils.python_utils import *

import matplotlib.pyplot as plt

class LatNetController(object):
    """Controls the execution of a LN simulation."""

    def __init__(self, trainer=None, simulation=None):

      self._config_parser = LatNetConfigParser()
      self._trainer = trainer
      self._simulation = simulation
     
      group = self._config_parser.add_group('Controller Details')
      group.add_argument('--mode', help='runtime mode', type=str,
            choices=['batch', 'visualization', 'benchmark'], default='batch'),
      group.add_argument('--run_mode', help='all modes', type=str,
            choices=['generate_data', 'train', 'eval'], default='train')

      group = self._config_parser.add_group('Network Details')
      group.add_argument('--latnet_network_dir', help='all mode', type=str,
                        default='./network_checkpoint')

      group = self._config_parser.add_group('Network Input Details')
      group.add_argument('--input_shape', help='all mode', type=str,
                         default='256x256')
      group.add_argument('--input_cshape', help='all mode', type=str,
                         default='32x32')
      group.add_argument('--lb_to_ln', help='all mode', type=int,
                        default=60)

      group = self._config_parser.add_group('Network Saver Details')
      group.add_argument('--save_network_freq', help='all mode', type=int, 
                        default=100)

      group = self._config_parser.add_group('Network Train Details')
      group.add_argument('--seq_length', help='all mode', type=int, 
                        default=5)
      group.add_argument('--batch_size', help='all mode', type=int,
                        default=4)
      group.add_argument('--gpus', help='all mode', type=str,
                        default='0')
      group.add_argument('--optimizer', help='all mode', type=str,
                        default='adam')
      group.add_argument('--gan', help='all mode', type=str2bool,
                        default=True)
      group.add_argument('--lr', help='all mode', type=float,
                        default=0.0002)
      group.add_argument('--decay_steps', help='all mode', type=int,
                        default=10000)
      group.add_argument('--decay_rate', help='all mode', type=float,
                        default=0.5)
      group.add_argument('--beta1', help='all mode', type=float,
                        default=0.9)
      group.add_argument('--l1_factor', help='all mode', type=float,
                        default=5.0)
      group.add_argument('--moving_average', help='all mode', type=str2bool,
                        default=False)
      group.add_argument('--train_iters', help='all mode', type=int,
                        default=500000)

      group = self._config_parser.add_group('Data Queue Details')
      group.add_argument('--train_sim_dir', help='train mode', type=str,
                        default='./train_data/sailfish_sim')
      group.add_argument('--gpu_fraction', help='all mode', type=float,
                        default=0.9)
      group.add_argument('--max_queue', help='all mode', type=int,
                        default=25)

      group = self._config_parser.add_group('Simulation Details')
      group.add_argument('--sim_shape', help='all mode', type=str,
                        default='512x512')
      group.add_argument('--DxQy', help='all mode', type=str,
            choices=['D2Q9'], default='D2Q9')
      group.add_argument('--num_iters', help='eval mode', type=int,
                        default=15)
      group.add_argument('--sim_restore_iter', help='if 0 then it will not restore', type=int,
                        default=1)

      group = self._config_parser.add_group('Simulation Saver Details')
      group.add_argument('--sim_dir', help='eval mode', type=str,
                        default='./simulation')
      group.add_argument('--sim_save_every', help='eval mode', type=int,
                        default=1)

      group = self._config_parser.add_group('Simulation Process Details')
      group.add_argument('--compare', help='compares to sailfish simulation', type=str2bool,
                        default=True)
      group.add_argument('--save_format', help='eval mode', type=str,
                        default='npy')
      group.add_argument('--save_cstate', help='eval mode', type=str2bool,
                        default=False)

      # TODO this group will be removed when the sailfish configs are integrated
      group = self._config_parser.add_group('Sailfish Helper Details')
      group.add_argument('--checkpoint_from', help='all mode', type=int,
                        default=100)
      group.add_argument('--restore_from', help='all mode', type=str,
                        default='')
      group.add_argument('--max_sim_iters', help='all mode', type=int,
                        default=50)
      group.add_argument('--visc', help='all mode', type=float,
                        default=0.1)
      group.add_argument('--restore_geometry', help='all mode', type=str2bool,
                        default=False)
      group.add_argument('--scr_scale', help='all mode', type=float,
                        default=.5)
      group.add_argument('--debug_sailfish', help='all mode', type=str2bool,
                        default=False)
      group.add_argument('--every', help='all mode', type=int,
                        default=100)
      group.add_argument('--subgrid', help='all mode', type=str,
                        default='les-smagorinsky')
      group.add_argument('--domain_name', help='all mode', type=str,
                        default='channel')

      group = self._config_parser.add_group('Network Configs')
      if self._trainer is not None:
        # add network specific configs
        if self._trainer.network:
          for base in self._trainer.network.mro():
            if 'add_options' in base.__dict__:
              base.add_options(group)
      elif self._simulation is not None:
        # add network specific configs
        if self._simulation.network is not None:
          for base in self._simulation.network.mro():
            if 'add_options' in base.__dict__:
              base.add_options(group)

      # update default configs based on simulation-specific class and network.
      defaults = {}
      if self._trainer is not None:
        self._trainer.update_defaults(defaults)
      if self._simulation is not None:
        self._simulation.update_defaults(defaults)
      self._config_parser.set_defaults(defaults)

    def run(self):

      # parse config
      args = sys.argv[1:]
      self.config = self._config_parser.parse(args)
     
      if self.config.run_mode == "train":
        self.train(self.config)
      elif self.config.run_mode == "generate_data":
        self.generate_data(self.config)
      elif self.config.run_mode == "eval":
        self.eval(self.config)

    def train(self, config):

      # train
      self.trainer = self._trainer(self.config)
      self.trainer.init_network()
      self.trainer.make_data_queue()
      self.trainer.train()

    def generate_data(self, config):

      if self._trainer is not None:
        for domain in self._trainer.domains:
          if config.domain_name == domain.name:
            sailfish_ctrl = SailfishSimulation(config, domain).create_sailfish_simulation()
            sailfish_ctrl.run()
            break
      elif self._simulation is not None:
        if config.domain_name == self._simulation.domain.name:
          sailfish_ctrl = SailfishSimulation(config, self._simulation.domain).create_sailfish_simulation()
          sailfish_ctrl.run()

    def eval(self, config):

      self.simulation = self._simulation(self.config)
      self.simulation.run()




