
import sys

import tensorflow as tf

from latnetwork import LatNet
from data_queue import DataQueue
from config import LatNetConfigParser
from domain import Domain
#from lattice import *

import matplotlib.pyplot as plt

class LatNetController(object):
    """Controls the execution of a LN simulation."""

    def __init__(self, eval_sim=None, train_sim=None):

      self._config_parser = LatNetConfigParser()
      self._train_sim = train_sim
      self._eval_sim = eval_sim
     
      group = self._config_parser.add_group('Basic Stuff')
      group.add_argument('--mode', help='runtime mode', type=str,
            choices=['batch', 'visualization', 'benchmark'], default='batch'),
      group.add_argument('--run_mode', help='all modes', type=str,
            choices=['generate_data', 'train', 'eval'], default='train')
      group.add_argument('--sailfish_sim_dir', help='train mode', type=str,
                        default='/data/sailfish_sim/')
      group.add_argument('--latnet_network_dir', help='all mode', type=str,
                        default='./network_checkpoint')
      group.add_argument('--latnet_sim_dir', help='eval mode', type=str,
                        default='./latnet_simulation')

      
      group = self._config_parser.add_group('Network Details')
      group.add_argument('--network_name', help='all mode', type=str,
                        default='advanced_network')

      group = self._config_parser.add_group('Simulation Details')
      group.add_argument('--sim_shape', help='all mode', type=str,
                        default='512x512')

      group.add_argument('--visc', help='all mode', type=float,
                        default=0.1)
      group.add_argument('--max_sim_iters', help='all mode', type=int,
                        default=50000)
      group.add_argument('--lb_to_ln', help='all mode', type=int,
                        default=120)
      group.add_argument('--restore_geometry', help='all mode', type=bool,
                        default=False)


      group = self._config_parser.add_group('Saver Details')
      group.add_argument('--save_freq', help='all mode', type=int, 
                        default=100)

      group = self._config_parser.add_group('Train Details')
      group.add_argument('--seq_length', help='all mode', type=int, 
                        default=5)
      group.add_argument('--batch_size', help='all mode', type=int,
                        default=2)
      group.add_argument('--optimizer', help='all mode', type=str,
                        default='adam')
      group.add_argument('--lr', help='all mode', type=float,
                        default=0.0004)
      group.add_argument('--train_iterations', help='all mode', type=int,
                        default=1000000)

      group = self._config_parser.add_group('Data Queue Details')
      group.add_argument('--gpu_fraction', help='all mode', type=float,
                        default=0.3)
      group.add_argument('--num_simulations', help='all mode', type=int,
                        default=10)
      group.add_argument('--max_queue', help='all mode', type=int,
                        default=30)
      group.add_argument('--nr_threads', help='all mode', type=int,
                        default=1)
      group.add_argument('--checkpoint_from', help='all mode', type=int,
                        default=100)
      group.add_argument('--restore_from', help='all mode', type=str,
                        default='')

      group = self._config_parser.add_group('Input Details')
      group.add_argument('--input_shape', help='all mode', type=str,
                         default='512x512')
      group.add_argument('--compressed_shape', help='all mode', type=str,
                         default='64x64')
      group.add_argument('--lattice_q', help='all mode', type=int,
            choices=[9], default=9)


    def _finish_simulation(self, subdomains, summary_receiver):
      pass

    def _load_sim(self):
      pass

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
      self.dataset = DataQueue(self.config, self._train_sim, self.network.train_shape_converter())

      while True:
        feed_dict = self.dataset.minibatch()
        self.network.train_step(feed_dict)
        #if finished:
        #  print("finished training")
        #  break

    def generate_data(self, config):

      sailfish_ctrl = self._train_sim(config).create_sailfish_simulation()
      sailfish_ctrl.run()

    def eval(self, config):

      self.network = LatNet(self.config)

      with tf.Graph().as_default():

        # unroll network
        (state_encoder, boundary_encoder, cmapping, decoder,
         encoder_shape_converter, cmapping_shape_converter, 
         decoder_shape_converter) = self.network.eval_unroll()

        # run simulation
        self.domain = self._eval_sim(config, self.network.network_config['nr_downsamples'])

        # compute compressed state
        cstate    = self.domain.state_to_cstate(state_encoder, encoder_shape_converter)
        cboundary = self.domain.boundary_to_cboundary(boundary_encoder, encoder_shape_converter)
        print(cboundary.shape)
        print(cstate.shape)

        for i in xrange(1000):

          cstate = self.domain.cstate_to_cstate(cmapping, cmapping_shape_converter, cstate, cboundary)
          if i % 10 == 0:
            # decode state
            state = self.domain.cstate_to_state(decoder, decoder_shape_converter, cstate)
            plt.imshow(state[0,:,:,0])
            plt.savefig('figs/out_state_' + str(i) + '.png')

        """
        print(self.state_from_compressed_state.get_shape())
        print(decompressed_state.shape)
        plt.imshow(decompressed_state[0,:,:,0])
        plt.show()
        exit()

        # perform simulation on compressed state
        for i in xrange(1000):
          compressed_state = self.domain.compute_compressed_mapping(sess, 
                                      self.compressed_state_from_compressed_state, 
                                      self.compressed_state, self.compressed_boundary,
                                      compressed_state, compressed_boundary, 
                                      self.network.compressed_state_padding_decrease())

          if i % 10 == 0:
            decompressed_state = self.domain.compute_decompressed_state(sess,
                                      self.state_from_compressed_state,
                                      self.decoder_compressed_state, self.decoder_compressed_boundary,
                                      compressed_state, compressed_boundary, 
                                      self.network.decompressed_state_padding_decrease())

            print(decompressed_state.shape)
            plt.imshow(decompressed_state[0,:,:,2])
            plt.show()
        """



