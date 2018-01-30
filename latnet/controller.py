
import sys

import tensorflow as tf

from latnetwork import LatNet
from data_queue import DataQueue
from config import LatNetConfigParser
from domain import Domain
from sim_saver import SimSaver

import matplotlib.pyplot as plt

class LatNetController(object):
    """Controls the execution of a LN simulation."""

    def __init__(self, _sim=None):

      self._config_parser = LatNetConfigParser()
      self._sim = _sim
     
      group = self._config_parser.add_group('Controller Details')
      group.add_argument('--mode', help='runtime mode', type=str,
            choices=['batch', 'visualization', 'benchmark'], default='batch'),
      group.add_argument('--run_mode', help='all modes', type=str,
            choices=['generate_data', 'train', 'eval'], default='train')

      group = self._config_parser.add_group('Network Details')
      group.add_argument('--latnet_network_dir', help='all mode', type=str,
                        default='./network_checkpoint')
      group.add_argument('--network_name', help='all mode', type=str,
                        default='advanced_network')

      group = self._config_parser.add_group('Network Input Details')
      group.add_argument('--input_shape', help='all mode', type=str,
                         default='512x512')
      group.add_argument('--input_cshape', help='all mode', type=str,
                         default='128x128')
      group.add_argument('--lb_to_ln', help='all mode', type=int,
                        default=120)

      group = self._config_parser.add_group('Network Saver Details')
      group.add_argument('--save_network_freq', help='all mode', type=int, 
                        default=100)

      group = self._config_parser.add_group('Network Train Details')
      group.add_argument('--seq_length', help='all mode', type=int, 
                        default=5)
      group.add_argument('--batch_size', help='all mode', type=int,
                        default=4)
      group.add_argument('--optimizer', help='all mode', type=str,
                        default='adam')
      group.add_argument('--lr', help='all mode', type=float,
                        default=0.0010)
      group.add_argument('--train_iterations', help='all mode', type=int,
                        default=1000000)

      group = self._config_parser.add_group('Data Queue Details')
      group.add_argument('--train_sim_dir', help='train mode', type=str,
                        default='./data_train/sailfish_sim/')
      group.add_argument('--gpu_fraction', help='all mode', type=float,
                        default=0.3)
      group.add_argument('--num_simulations', help='all mode', type=int,
                        default=10)
      group.add_argument('--max_queue', help='all mode', type=int,
                        default=30)

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
                        default='./simulation/')
      group.add_argument('--sim_save_every', help='eval mode', type=int,
                        default=1)

      group = self._config_parser.add_group('Simulation Process Details')
      group.add_argument('--compare', help='compares to sailfish simulation', type=bool,
                        default=True)
      group.add_argument('--save_format', help='eval mode', type=str,
                        default='npy')
      group.add_argument('--save_cstate', help='eval mode', type=bool,
                        default=False)

      # TODO this group will be removed when the sailfish configs are integrated
      group = self._config_parser.add_group('Sailfish Helper Details')
      group.add_argument('--checkpoint_from', help='all mode', type=int,
                        default=100)
      group.add_argument('--restore_from', help='all mode', type=str,
                        default='')
      group.add_argument('--max_sim_iters', help='all mode', type=int,
                        default=50000)
      group.add_argument('--visc', help='all mode', type=float,
                        default=0.1)
      group.add_argument('--restore_geometry', help='all mode', type=bool,
                        default=False)

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

      # make network
      self.network = LatNet(self.config)

      # unroll train_unroll
      self.network.train_unroll()
 
      # construct dataset
      self.dataset = DataQueue(self.config, self._sim, self.network.train_shape_converter())

      while True:
        feed_dict = self.dataset.minibatch()
        self.network.train_step(feed_dict)
        #if finished:
        #  print("finished training")
        #  break

    def generate_data(self, config):

      sailfish_ctrl = self._sim(config).create_sailfish_simulation()
      sailfish_ctrl.run()

    def eval(self, config):

      self.network = LatNet(self.config)

      with tf.Graph().as_default():

        # unroll network
        (state_encoder, boundary_encoder, cmapping, decoder,
         encoder_shape_converter, cmapping_shape_converter, 
         decoder_shape_converter) = self.network.eval_unroll()

        # run simulation
        self.domain = self._sim(config, self.network.network_config['nr_downsamples'])

        self.domain.run(state_encoder, 
                        boundary_encoder, 
                        cmapping, decoder,
                        encoder_shape_converter, 
                        cmapping_shape_converter, 
                        decoder_shape_converter)

