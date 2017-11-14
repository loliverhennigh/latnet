
import tensorflow as tf

import lattice as lat

class Inputs:

  def __init__(self, config):

    # determine shape
    input_shape = config.input_shape.split('x')
    input_shape = map(int, input_shape)
    self.input_shape = input_shape
    self.lattice_q = config.lattice_q
    self.boundary_depth = 4 # need to add as config for em simulations
    self.batch_size = config.batch_size
  
    if config.run_mode == "train":
      self.seq_length = config.seq_length
    elif config.run_mode == "eval":
      self.seq_length = 1

  def state_seq(self, state_padding_decrease_seq):
    state_in = tf.placeholder(tf.float32, [self.batch_size] + self.input_shape + [self.lattice_q])
    state_out = []
    for i in xrange(self.seq_length):
      input_shape = [x - 2*state_padding_decrease_seq[i] for x in self.input_shape]
      state_out.append(tf.placeholder(tf.float32, [self.batch_size] + input_shape + [self.lattice_q]))
      tf.summary.image('true_state_out_vel_' + str(i), lat.vel_to_norm(lat.lattice_to_vel(state_out[i])))

    tf.summary.image('true_state_in_vel', lat.vel_to_norm(lat.lattice_to_vel(state_in)))
    return state_in, state_out

  def boundary(self):
    boundary = tf.placeholder(tf.float32, [self.batch_size] + self.input_shape + [self.boundary_depth])
    tf.summary.image('true_boundary', boundary[:,:,:,0:1])
    return boundary
    
    
