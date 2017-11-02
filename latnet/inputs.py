
import tensorflow as tf

class Inputs:

  def __init__(self, config):

    # determine shape
    shape = config.shape.split('x')
    shape = map(int, shape)
    self.shape = shape
    self.lattice_q = config.lattice_q
    self.boundary_depth = 4 # need to add as config for em simulations
    self.batch_size = config.batch_size
  
    if config.mode == "train":
      self.seq_length = config.seq_length
    elif config.mode == "eval":
      self.seq_length = 1

  def state(self):
    state = tf.placeholder(tf.float32, [self.batch_size, self.seq_length] + self.shape + [self.lattice_q])
    tf.summary.image('true_state_1', state[:,0,:,:,0:1])
    tf.summary.image('true_state_2', state[:,0,:,:,1:2])
    tf.summary.image('true_state_3', state[:,0,:,:,2:3])
    return state

  def boundary(self):
    boundary = tf.placeholder(tf.float32, [self.batch_size, 1] + self.shape + [self.boundary_depth])
    tf.summary.image('true_boundary', boundary[:,0,:,:,0:1])
    return boundary
    
    
