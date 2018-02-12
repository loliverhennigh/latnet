
"""functions used to construct different optimizers

Function borrowed and modified from https://github.com/openai/pixel-cnn
"""

import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

class Optimizer:

  def __init__(self, config, name):
    self.lr = config.lr
    self.decay_steps = config.decay_steps
    self.decay_rate = config.decay_rate
    self.beta1 = config.beta1
    self.moving_average = config.moving_average
    self.name = name
    if config.optimizer == "adam":
      self.train_op = self.adam_updates

  def adam_updates(self, params, gradients, global_step, mom2=0.999, other_update=None):
    ''' Adam optimizer '''
    updates = []

    # make moving average
    if self.moving_average:
      ema = tf.train.ExponentialMovingAverage(decay=.9995)
      updates.append(tf.group(ema.apply(params)))
    learning_rate = tf.train.exponential_decay(self.lr, global_step, 
                                               self.decay_steps, self.decay_rate)
    tf.summary.scalar('learning_rate_' + self.name, learning_rate)
 
    t = tf.Variable(1., self.name + '_adam_t')
    for p, g in zip(params, gradients):
      if g is None:
        continue
      mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
      if self.beta1>0:
        v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
        v_t = self.beta1*v + (1. - self.beta1)*g
        v_hat = v_t / (1. - tf.pow(self.beta1,t))
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
      print(other_update)
      updates += other_update

    return tf.group(*updates)

