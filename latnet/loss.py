
import tensorflow as tf
import numpy as np

class Loss:

  def __init__(self, config):
    pass

  def mse(self, true, generated):
    loss = tf.nn.l2_loss(true - generated)
    tf.summary.scalar('mse_loss', loss)
    return loss




