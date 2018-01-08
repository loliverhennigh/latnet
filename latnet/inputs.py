
import tensorflow as tf

import lattice as lat

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

    # make state, boundary, and compresed state shapes
    self.state_shape = self.dims * [None] + [self.lattice_q]
    self.boundary_shape = self.dims * [None] + [self.boundary_depth]
    self.cstate_shape = self.dims * [None] + [self.boundary_depth]

  def state_seq(self, state_padding_decrease_seq, with_batch=True):
    if with_batch == True:
      shape =  
    else:
      shape = [1] + 
      state_in = tf.placeholder(tf.float32, [self.batch_size] + self.input_shape + [self.lattice_q])
    state_out = []
    for i in xrange(self.seq_length):
      input_shape = [None, None]
      state_out.append(tf.placeholder(tf.float32, [self.batch_size] + input_shape + [self.lattice_q]))
      tf.summary.image('true_state_out_vel_' + str(i), lat.vel_to_norm(lat.lattice_to_vel(state_out[i])))

    tf.summary.image('true_state_in_vel', lat.vel_to_norm(lat.lattice_to_vel(state_in)))
    return state_in, state_out

  def state(self, padding=0):
    input_shape = [x + 2*padding for x in self.input_shape]
    state = tf.placeholder(tf.float32, [self.batch_size] + input_shape + [self.lattice_q])
    tf.summary.image('true_state', lat.vel_to_norm(state))
    return state
 
  def boundary(self, padding=0):
    input_shape = [x + 2*padding for x in self.input_shape]
    boundary = tf.placeholder(tf.float32, [self.batch_size] + input_shape + [self.boundary_depth])
    tf.summary.image('true_boundary', boundary[:,:,:,0:1])
    return boundary
    
  def compressed_state(self, filter_size, padding=0): 
    compressed_shape = [x + 2*padding for x in self.compressed_shape]
    compressed_state = tf.placeholder(tf.float32, [self.batch_size] + compressed_shape + [filter_size])
    return compressed_state
    
  def compressed_boundary(self, filter_size, padding=0): 
    compressed_shape = [x + 2*padding for x in self.compressed_shape]
    compressed_boundary = tf.placeholder(tf.float32, [self.batch_size] + compressed_shape + [filter_size])
    return compressed_boundary

