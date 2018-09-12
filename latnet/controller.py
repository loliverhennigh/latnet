
import sys

import tensorflow as tf

from latnetwork import LatNet
from data_queue import DataQueue
from config import LatNetConfigParser
from domain import Domain
from sim_saver import SimSaver
from optimizer import Optimizer
from data_queue import DataQueue
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
      group.add_argument('--dataset', help='all modes', type=str,
            choices=['sailfish', 'JHTDB', 'spectralDNS'], default='JHTDB')

      # TODO this group will be removed when the sailfish configs are integrated
      group = self._config_parser.add_group('Sailfish Helper Details')
      group.add_argument('--checkpoint_from', help='all mode', type=int,
                        default=100)
      group.add_argument('--restore_from', help='all mode', type=str,
                        default='')
      group.add_argument('--max_sim_iters', help='all mode', type=int,
                        default=50)
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

      group = self._config_parser.add_group('Domain Configs')
      for base in Domain.mro():
        if 'add_options' in base.__dict__:
          base.add_options(group)
      group = self._config_parser.add_group('Data Queue Configs')
      for base in DataQueue.mro():
        if 'add_options' in base.__dict__:
          base.add_options(group)
      # add optimizer specific configs
      group = self._config_parser.add_group('Optimizer Configs')
      for base in Optimizer.mro():
        if 'add_options' in base.__dict__:
          base.add_options(group)

      if self._trainer is not None:
        # add network specific configs
        group = self._config_parser.add_group('Network Configs')
        if self._trainer:
          for base in self._trainer.mro():
            if 'add_options' in base.__dict__:
              base.add_options(group)

      elif self._simulation is not None:
        # add network specific configs
        group = self._config_parser.add_group('Network Generated Simulation Configs')
        if self._simulation is not None:
          for base in self._simulation.mro():
            if 'add_options' in base.__dict__:
              base.add_options(group)
        # add network specific configs
        group = self._config_parser.add_group('Network Generated Simulation Saver Configs')
        for base in SimSaver.mro():
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
      elif self.config.run_mode == "decode":
        self.decode(self.config)

    def train(self, config):
      self.trainer = self._trainer(self.config)
      self.trainer.train()

    def generate_data(self, config):
      print("generating data")
      if self._trainer is not None:
        for domain in self._trainer.domains:
          if config.domain_name == domain.name:
            if domain.wrapper_name == 'sailfish':
              sailfish_ctrl = domain(config, config.train_data_dir).create_sailfish_simulation(config)
              sailfish_ctrl.run()
              break
      elif self._simulation is not None:
        if self._simulation.domain.wrapper_name == 'sailfish':
          sailfish_ctrl = self._simulation.domain(config, config.train_data_dir).create_sailfish_simulation(config)
          sailfish_ctrl.run()

    def eval(self, config):
      self.simulation = self._simulation(self.config)
      self.simulation.eval()

    def decode(self, config):
      self.simulation = self._simulation(self.config)
      self.simulation.decode()





