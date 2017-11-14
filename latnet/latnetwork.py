
import tensorflow as tf
import numpy as np

from saver import Saver
import lattice as lat


class LatNet:

  def __init__(self, config):

    self.network_dir  = config.latnet_network_dir
    self.network_name = config.network_name
    self.seq_length = config.seq_length

    if self.network_name == "basic_network":
      import network_architectures.basic_network as net
    else:
      print("network name not found")
      exit()

    # piecese of network
    self.encoder_state                = tf.make_template('encoder_state', net.encoder_state)
    self.encoder_boundary             = tf.make_template('encoder_boundary', net.encoder_boundary)
    self.compression_mapping_boundary = tf.make_template('compression_mapping_boundary', net.compression_mapping_boundary)
    self.compression_mapping          = tf.make_template('compression_mapping', net.compression_mapping)
    self.decoder_state                = tf.make_template('decoder_state', net.decoder_state)
    self.network_config               = net.CONFIGS
    self.padding                      = net.PADDING

    self.unroll                       = tf.make_template('unroll', self._unroll)
    self.continual_unroll             = tf.make_template('continual_unroll', self._continual_unroll)

  def _unroll(self, state_in, boundary):
    # assumes state has seq length in second dim
    # store all out
    x_out = []

    # encode
    y_1 = self.encoder_state(state_in)
    compressed_boundary = self.encoder_boundary(boundary)

    # apply boundary
    y_1 = self.compression_mapping_boundary(y_1, compressed_boundary)

    # unroll all
    for i in xrange(self.seq_length):
      # decode and add to list
      x_2 = self.decoder_state(y_1)
      x_out.append(x_2)

      # compression mapping
      y_1 = self.compression_mapping(y_1)

      # apply boundary
      y_1 = self.compression_mapping_boundary(y_1, compressed_boundary)

    # make image summary
    for i in xrange(self.seq_length):
      tf.summary.image('predicted_state_out_vel_', lat.vel_to_norm(lat.lattice_to_vel(x_out[i])))

    return x_out

  def _continual_unroll(self, state, boundary):
    # encode
    y_1 = self.encoder_state(state)
    compressed_boundary = self.encode_boundary(boundary)

    # apply boundary
    y_1_boundary = self.compression_mapping_boundary(y_1, compressed_boundary)

    # decode and add to list
    x_2 = self.decoder_state(y_1_boundary)

    # compression mapping
    y_2 = self.compression_mapping(y_1_boundary)

    return y_1, compressed_boundary, x_2, y_2

  def state_padding_decrease_seq(self):
    # calculates the decrease in state size after unrolling the network
    decrease = []
    for i in xrange(self.seq_length):
      decrease.append((self.padding['encoder_state_padding']
              + (i+1)*self.padding['compression_mapping_boundary_padding']
                  + i*self.padding['compression_mapping_padding']
                    + self.padding['decoder_state_padding']))
    return decrease



