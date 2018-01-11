
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

  def state_seq(self):
    states = {}
    for i in xrange(self.seq_length):
      states['state_' + str(i)] = tf.placeholder(tf.float32, (2 + self.dims) + [None])
      tf.summary.image('true_states_vel_' + str(i), lat.lattice_to_norm(states['state_' + str(i)]))

    return states

  def state(self):
    state = {'state': tf.placeholder(tf.float32, (2 + self.dims) + [None])}
    tf.summary.image('true_state', lat.lattice_to_norm(state['state']))
    return state
 
  def boundary(self):
    boundary = {'boundary': tf.placeholder(tf.float32, (2 + self.dims) + [None])}
    # TODO add more image summarys for boundary
    tf.summary.image('true_boundary', boundary['boundary'][...,0:1])
    return boundary
    
  def compressed_state(self): 
    compressed_state = {'cstate':, tf.placeholder(tf.float32, (2 + self.dims) + [None])}
    return compressed_state
    
  def compressed_boundary(self): 
    compressed_boundary = {'cboundary': tf.placeholder(tf.float32,, (2 + self.dims) + [None])}
    return compressed_boundary
