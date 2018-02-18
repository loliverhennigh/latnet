
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

      group = self._config_parser.add_group('Network Input Details')
      group.add_argument('--input_shape', help='all mode', type=str,
                         default='256x256')
      group.add_argument('--input_cshape', help='all mode', type=str,
                         default='32x32')
      group.add_argument('--lb_to_ln', help='all mode', type=int,
                        default=60)
      group.add_argument('--boundary_mask', help='all mode', type=str2bool,
                        default=False)

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
                        default=50000)

      group = self._config_parser.add_group('Data Queue Details')
      group.add_argument('--train_sim_dir', help='train mode', type=str,
                        default='./data_train/sailfish_sim')
      group.add_argument('--gpu_fraction', help='all mode', type=float,
                        default=0.9)
      group.add_argument('--num_simulations', help='all mode', type=int,
                        default=10)
      group.add_argument('--max_queue', help='all mode', type=int,
                        default=50)

      group = self._config_parser.add_group('Simulation Details')
      group.add_argument('--sim_shape', help='all mode', type=str,
                        default='512x512')
      group.add_argument('--periodic_x', help='all mode', type=str2bool,
                        default=True)
      group.add_argument('--periodic_y', help='all mode', type=str2bool,
                        default=True)
      group.add_argument('--DxQy', help='all mode', type=str,
            choices=['D2Q9'], default='D2Q9')
      group.add_argument('--num_iters', help='eval mode', type=int,
                        default=15)
      group.add_argument('--sim_restore_iter', help='if 0 then it will not restore', type=int,
                        default=10)

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

      self.network_name = _sim.network_name
      group = self._config_parser.add_group('Network Configs')
      # add network specific configs
      for base in LatNet.mro():
          if 'add_options' in base.__dict__:
              base.add_options(group, self.network_name)

      # update default configs based on simulation-specific class.
      if self._sim is not None:
        defaults = {}
        self._sim.update_defaults(defaults)
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

      # make network
      self.network = LatNet(self.config, self.network_name, self._sim.script_name)

      # unroll train_unroll
      self.network.train_unroll()
 
      # construct dataset
      self.dataset = DataQueue(self.config, self._sim, self.network.train_shape_converter())

      # train network
      self.network.train(self.dataset)

    def generate_data(self, config):

      sailfish_ctrl = self._sim(config).create_sailfish_simulation()
      sailfish_ctrl.run()

    def eval(self, config):

      self.network = LatNet(self.config, self.network_name, self._sim.script_name)

      with tf.Graph().as_default():

        # unroll network
        (state_encoder, boundary_encoder, cmapping, decoder,
         encoder_shape_converter, cmapping_shape_converter, 
         decoder_shape_converter) = self.network.eval_unroll()

        # run simulation
        self.domain = self._sim(config)

        self.domain.run(state_encoder, 
                        boundary_encoder, 
                        cmapping, decoder,
                        encoder_shape_converter, 
                        cmapping_shape_converter, 
                        decoder_shape_converter)
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

