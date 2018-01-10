
import tensorflow as tf

import lattice as lat

from network_architectures.shape_converter import ShapeConverter

class Inputs:

  def __init__(self, config):

    # D and Q of lattice (eg D2Q9 scheme is 2 and 9)
    self.dims = 2 # hard set for now
    self.lattice_q = config.lattice_q

    # number of boundary layers (hard set for now D2Q9)
    self.boundary_depth = 4 # need to add as config for em simulations
 
    # other important params for making inputs 
    self.seq_length = config.seq_length
    self.batch_size = config.batch_size

  def state_seq(self, state_padding_decrease_seq):
    state_in = Tensor(tf.placeholder(tf.float32, (2 + self.dims) + [None]))
    state_out = []
    for i in xrange(self.seq_length):
      state_out.append(Tensor(tf.placeholder(tf.float32, (2 + self.dims) + [None])))
      tf.summary.image('true_state_out_vel_' + str(i), lat.vel_to_norm(lat.lattice_to_vel(state_out[i])))

    tf.summary.image('true_state_in_vel', lat.lattice_to_norm(state_in)))
    return state_in, state_out

  def state(self, padding=0):
    state = Tensor(tf.placeholder(tf.float32, (2 + self.dims) + [None]))
    tf.summary.image('true_state', lat.lattice_to_norm(state))
    return state
 
  def boundary(self, padding=0):
    boundary = Tensor(tf.placeholder(tf.float32, (2 + self.dims) + [None]))
    # TODO add more image summarys for boundary
    tf.summary.image('true_boundary', boundary[:,:,:,0:1])
    return boundary
    
  def compressed_state(self, filter_size, padding=0): 
    compressed_state = Tensor(tf.placeholder(tf.float32, (2 + self.dims) + [None]))
    return compressed_state
    
  def compressed_boundary(self, filter_size, padding=0): 
    compressed_boundary = Tensor(tf.placeholder(tf.float32,, (2 + self.dims) + [None]))
    return compressed_boundary

class Pipe:
  # store the input and output
  def __init__(self, tf_tensor):
    self.tf_tensor = tf_tensor
    self.shape_converter = ShapeConverter()
