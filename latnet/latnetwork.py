
import tensorflow as tf
import numpy as np

from saver import Saver
import lattice as lat

from network_architectures.shape_converter import ShapeConverter


class LatNet:

  def __init__(self, config):

    self.network_dir  = config.latnet_network_dir
    self.network_name = config.network_name
    self.seq_length = config.seq_length

    if self.network_name == "basic_network":
      import network_architectures.basic_network as net
    elif self.network_name == "advanced_network":
      import network_architectures.advanced_network as net
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
    self.single_unroll             = tf.make_template('unroll', self._single_unroll)

  def _unroll(self, pipe):
    # encode
    self.encoder_state(pipe, in_name="state", out_name="cstate_0")
    self.encoder_boundary(pipe, in_name="boundary", out_name="cboundary")

    # unroll all
    for i in xrange(self.seq_length):
      # decode and add to list
      self.decoder_state(pipe, in_name="cstate_" + str(i), out_name="state_" + str(i))

      # compression mapping
      self.compression_mapping(pipe, in_name="cstate_" + str(i), out_name="cstate_" + str(i))

      # apply boundary
      self.compression_mapping_boundary(pipe, in_name="cstate_" + str(i), in_name="cstate_" + str(i+1))

      # make image summary
      tf.summary.image('predicted_state_vel_', lat.lattice_to_norm(pipe.out_tensors['state_' + str(i)]))

  def _single_unroll(self, state, boundary, 
                compressed_state, compressed_boundary, 
                decoder_compressed_state, decoder_compressed_boundary):
    # encode
    compressed_state_from_state = self.encoder_state(state)
    compressed_boundary_from_boundary = self.encoder_boundary(boundary)

    # compression mapping
    compressed_state_from_compressed_state = self.compression_mapping_boundary(compressed_state, compressed_boundary)
    compressed_state_from_compressed_state = self.compression_mapping(compressed_state_from_compressed_state)

    # decode and add to list
    compressed_state_store = self.compression_mapping_boundary(decoder_compressed_state, decoder_compressed_boundary)
    state_from_compressed_state = self.decoder_state(compressed_state_store)

    return (compressed_state_from_state, compressed_boundary_from_boundary, 
           compressed_state_from_compressed_state,
           state_from_compressed_state)

  def state_padding_decrease_seq(self):
    # calculates the decrease in state size after unrolling the network
    decrease = []
    for i in xrange(self.seq_length):
      decrease.append((self.padding['encoder_state_padding']
              + (i+1)*self.padding['compression_mapping_boundary_padding']
                  + i*self.padding['compression_mapping_padding']
                    + self.padding['decoder_state_padding']))
    return decrease

  def state_padding_decrease(self):
    return self.padding['encoder_state_padding']

  def compressed_state_padding_decrease(self):
    print(self.padding['compression_mapping_padding'],pow(2,self.network_config['nr_downsamples']))
    return int(self.padding['compression_mapping_padding']/pow(2,self.network_config['nr_downsamples']))

  def decompressed_state_padding_decrease(self):
    print(self.padding['decoder_state_padding'],pow(2,self.network_config['nr_downsamples']))
    return int(self.padding['decoder_state_padding']/pow(2,self.network_config['nr_downsamples']))

  def compressed_filter_size(self):
    return self.network_config['filter_size_compression']





