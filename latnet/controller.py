
import sys

import tensorflow as tf

from latnetwork import LatNet
from data_queue import DataQueue
from config import LatNetConfigParser
from loss import Loss
from inputs import Inputs
from optimizer import Optimizer
from saver import Saver
from domain import Domain
from lattice import *

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
                        default='basic_network')

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
                        default=500)

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
                         default='256x256')
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

      self.network = LatNet(self.config)

      with tf.Graph().as_default():

        # global step counter
        global_step = tf.get_variable('global_step', [], 
                          initializer=tf.constant_initializer(0), trainable=False)

        # make inputs
        self.inputs = Inputs(self.config)
        self.state_in, self.state_out = self.inputs.state_seq(self.network.state_padding_decrease_seq())
        self.boundary = self.inputs.boundary()

        # make network pipe
        self.pred_state_out = self.network.unroll(self.state_in, self.boundary)

        # make loss
        self.loss = Loss(self.config)
        self.total_loss = self.loss.mse(self.state_out, self.pred_state_out)

        # make train op
        all_params = tf.trainable_variables()
        self.optimizer = Optimizer(self.config)
        self.optimizer.compute_gradients(self.total_loss, all_params)
        self.train_op = self.optimizer.train_op(all_params, global_step)

        # start session 
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.8)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        init = tf.global_variables_initializer()
        sess.run(init)

        # make saver
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        self.saver = Saver(self.config, self.network.network_config, graph_def)
        self.saver.load_checkpoint(sess)

        # construct dataset
        self.dataset = DataQueue(self.config, self._train_sim)

        # train
        for i in xrange(sess.run(global_step), self.config.train_iterations):
          _, l = sess.run([self.train_op, self.total_loss], 
                          feed_dict=self.dataset.minibatch(self.state_in, self.state_out, self.boundary, self.network.state_padding_decrease_seq()))
          if i % 100 == 0:
            print("current loss is " + str(l))
            print("current step is " + str(i))

          if i % self.config.save_freq == 0:
            print("saving...")
            self.saver.save_summary(sess, self.dataset.minibatch(self.state_in, self.state_out, self.boundary, self.network.state_padding_decrease_seq()), sess.run(global_step))
            self.saver.save_checkpoint(sess, int(sess.run(global_step)))

    def generate_data(self, config):

      ctrl = self._train_sim(config).create_sailfish_simulation()
      ctrl.run()

    def eval(self, config):

      self.network = LatNet(self.config)

      with tf.Graph().as_default():

        # make inputs
        self.inputs = Inputs(self.config)
        self.state = self.inputs.state(self.network.state_padding_decrease())
        self.compressed_state = self.inputs.compressed_state(self.network.network_config['filter_size_compression'],
                                           self.network.compressed_state_padding_decrease())
        self.decoder_compressed_state = self.inputs.compressed_state(self.network.network_config['filter_size_compression'],
                                           self.network.decompressed_state_padding_decrease())
        self.boundary = self.inputs.boundary(self.network.state_padding_decrease())
        self.compressed_boundary = self.inputs.compressed_boundary(2*self.network.network_config['filter_size_compression'],
                                           self.network.compressed_state_padding_decrease())
        self.decoder_compressed_boundary = self.inputs.compressed_boundary(2*self.network.network_config['filter_size_compression'],
                                           self.network.decompressed_state_padding_decrease())

        # make network pipe
        ( self.compressed_state_from_state, self.compressed_boundary_from_boundary, 
          self.compressed_state_from_compressed_state,
          self.state_from_compressed_state) = self.network.single_unroll(self.state,
                                              self.boundary, self.compressed_state,
                                              self.compressed_boundary,
                                              self.decoder_compressed_state,
                                              self.decoder_compressed_boundary)
        print(self.state_from_compressed_state.get_shape())

        # start session 
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        init = tf.global_variables_initializer()
        sess.run(init)

        # make saver
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        self.saver = Saver(self.config, self.network.network_config, graph_def)
        self.saver.load_checkpoint(sess, maybe_remove_prev=False)

        # run simulation
        self.domain = self._eval_sim(config, self.network.network_config['nr_downsamples'])

        # compute compressed state
        compressed_state = self.domain.compute_compressed_state(sess, 
                                self.compressed_state_from_state, 
                                self.state, self.network.state_padding_decrease())

        # compute compressed boundary
        compressed_boundary = self.domain.compute_compressed_boundary(sess, 
                                   self.compressed_boundary_from_boundary, 
                                   self.boundary, self.network.state_padding_decrease())

        decompressed_state = self.domain.compute_decompressed_state(sess,
                                  self.state_from_compressed_state,
                                  self.decoder_compressed_state, self.decoder_compressed_boundary,
                                  compressed_state, compressed_boundary, 
                                  self.network.decompressed_state_padding_decrease())

        plt.imshow(decompressed_state[0,:,:,0])
        plt.show()

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

            plt.imshow(decompressed_state[0,:,:,2])
            plt.show()



