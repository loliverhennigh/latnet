
import tensorflow as tf
import fnmatch
import os

class NetworkSaver:

  def __init__(self, config, network_name, graph_def):

    self.network_dir = config.latnet_network_dir
    self.config = config
    self.gan = config.gan
    self.network_name = network_name

    # a bit messy this way but oh well

    self.checkpoint_path = self._make_checkpoint_path()
    self._make_saver()
    self.summary_writer = self._make_summary_writer(graph_def)
    self.summary_op = self._make_summary_op()

  def _make_checkpoint_path(self):
    # make checkpoint path with all the flags specifing different directories
 
    # run through all params and add them to the base path
    base_path = self.network_dir + '/' + self.network_name
    for k, v in self.config.__dict__.items():
      if k not in self.none_save_args:
        base_path += '/' + k + '.' + str(v)
    return base_path

  def _make_saver(self):
    variables = tf.global_variables()
    variables_autoencoder = [v for i, v in enumerate(variables) if ("coder" in v.name[:v.name.index(':')]) or ('global' in v.name[:v.name.index(':')])]
    variables_compression = [v for i, v in enumerate(variables) if "compression_mapping" in v.name[:v.name.index(':')]]
    variables_discriminator = [v for i, v in enumerate(variables) if "discriminator" in v.name[:v.name.index(':')]]
    self.saver_all = tf.train.Saver(variables, max_to_keep=1)
    self.saver_autoencoder = tf.train.Saver(variables_autoencoder)
    self.saver_compression = tf.train.Saver(variables_compression)
    if self.gan:
      self.saver_discriminator = tf.train.Saver(variables_discriminator)

  def _make_summary_writer(self, graph_def):
    summary_writer = tf.summary.FileWriter(self.checkpoint_path, graph_def=graph_def)
    return summary_writer

  def _list_all_checkpoints(self):
    # get a list off all the checkpoint directorys

    # run through all params and add them to the base path
    paths = []
    for root, dirnames, filenames in os.walk(self.network_dir):
      for filename in fnmatch.filter(filenames, 'checkpoint'):
        paths.append(root[len(self.network_dir)+1:])
    return paths

  def load_checkpoint(self, sess, maybe_remove_prev=False):
    ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
    print("looking for checkpoint in " + self.checkpoint_path) 
    if ckpt is not None:
      print("trying init autoencoder from " + ckpt.model_checkpoint_path)
      try:
        self.saver_autoencoder.restore(sess, ckpt.model_checkpoint_path)
      except:
        if maybe_remove_prev:
          tf.gfile.DeleteRecursively(self.checkpoint_path)
          tf.gfile.MakeDirs(self.checkpoint_path)
        print("there was a problem using variables for autoencoder in checkpoint, random init will be used instead")
      print("trying init compression mapping from " + ckpt.model_checkpoint_path)
      try:
        self.saver_compression.restore(sess, ckpt.model_checkpoint_path)
      except:
        if maybe_remove_prev:
          tf.gfile.DeleteRecursively(self.checkpoint_path)
          tf.gfile.MakeDirs(self.checkpoint_path)
        print("there was a problem using variables for compression mapping in checkpoint, random init will be used instead")
      if self.gan:
        print("trying init discriminator from " + ckpt.model_checkpoint_path)
        try:
          self.saver_discriminator.restore(sess, ckpt.model_checkpoint_path)
        except:
          if maybe_remove_prev:
            tf.gfile.DeleteRecursively(self.checkpoint_path)
            tf.gfile.MakeDirs(self.checkpoint_path)
          print("there was a problem using variables for autoencoder in checkpoint, random init will be used instead")
    else:
      print("using rand init")

  def save_checkpoint(self, sess, global_step):
    save_path = os.path.join(self.checkpoint_path, 'model.ckpt')
    self.saver_all.save(sess, save_path, global_step=global_step)  

  def save_summary(self, sess, feed_dict, global_step):
    summary_str = sess.run(self.summary_op, feed_dict=feed_dict)
    self.summary_writer.add_summary(summary_str, global_step) 

    



