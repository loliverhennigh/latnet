
import tensorflow as tf
import numpy as np

class Loss:

  def __init__(self, config):
    pass

  def mse(self, true, generated):
    if isinstance(true, list):
      loss = 0.0
      for i in xrange(len(true)):
        loss += tf.nn.l2_loss(true[i] - generated[i])
    else:
      loss = tf.nn.l2_loss(true - generated)
    tf.summary.scalar('mse_loss', loss)
    return loss




