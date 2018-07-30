
"""functions used to construct different optimizers

Function borrowed and modified from https://github.com/openai/pixel-cnn
"""

import tensorflow as tf
import numpy as np

from utils.python_utils import *

FLAGS = tf.app.flags.FLAGS

class Optimizer(object):

  def __init__(self, config, name):
    self.lr = config.lr
    self.decay_steps = config.decay_steps
    self.decay_rate = config.decay_rate
    self.moving_average = config.moving_average
    self.name = name
    if config.optimizer == "adam":
      self.train_op = self.adam_updates
    elif config.optimizer == "gradient_descent":
      self.train_op = self.gradient_updates

  @classmethod
  def add_options(cls, group):
    group.add_argument('--optimizer', 
                   help='optimizer to use during training', 
                   type=str,
                   choices=['adam', 'gradient_descent'],
                   default='adam')
    group.add_argument('--lr', 
                   help='learning rate', 
                   type=float,
                   default=0.0004)
    group.add_argument('--decay_steps', 
                   help=' decay steps for exponential decay lr', 
                   type=int,
                   default=10000)
    group.add_argument('--decay_rate', 
                   help=' decay rate for exponential decay lr', 
                   type=float,
                   default=1.0)
    group.add_argument('--moving_average', 
                   help='moving average of weights', 
                   type=str2bool,
                   default=True)

  def get_lr(self, global_step):
    learning_rate = tf.train.exponential_decay(self.lr, global_step, 
                                               self.decay_steps, self.decay_rate)
    tf.summary.scalar('learning_rate_' + self.name, learning_rate)
    return learning_rate

  def get_moving_average(self, params):
    ema = tf.train.ExponentialMovingAverage(decay=.9995)
    return tf.group(ema.apply(params))
 
  def gradient_updates(self, params, gradients, global_step, other_update=None):
    ''' gradient optimizer '''
    updates = []

    # make moving average
    if self.moving_average:
      updates.append(self.get_moving_average(params))

    learning_rate = 1000.0 * self.get_lr(global_step)

    for p, g in zip(params, gradients):
      if g is None:
        continue
      p_t = p - learning_rate * g
      updates.append(p.assign(p_t))
    updates.append(global_step.assign_add(1))
    if other_update is not None:
      updates += other_update

    return tf.group(*updates)

  def adam_updates(self, params, gradients, global_step, mom1=0.9, mom2=0.999, other_update=None):
    ''' Adam optimizer '''
    updates = []

    # make moving average
    if self.moving_average:
      updates.append(self.get_moving_average(params))

    learning_rate = self.get_lr(global_step)
 
    t = tf.Variable(1., self.name + '_adam_t')
    for p, g in zip(params, gradients):
      if g is None:
        continue
      mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
      if mom1>0:
        v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
        v_t = mom1*v + (1. - mom1)*g
        v_hat = v_t / (1. - tf.pow(mom1,t))
        updates.append(v.assign(v_t))
      else:
        v_hat = g
      mg_t = mom2*mg + (1. - mom2)*tf.square(g)
      mg_hat = mg_t / (1. - tf.pow(mom2,t))
      g_t = v_hat / tf.sqrt(mg_hat + 1e-8)
      p_t = p - learning_rate * g_t
      updates.append(mg.assign(mg_t))
      updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    updates.append(global_step.assign_add(1))
    if other_update is not None:
      updates += other_update

    return tf.group(*updates)

