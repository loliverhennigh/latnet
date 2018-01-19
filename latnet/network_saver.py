
import tensorflow as tf
import fnmatch
import os

class NetworkSaver:

  def __init__(self, config, network_config, graph_def):

    self.network_dir = config.latnet_network_dir
    self.network_name = config.network_name
    self.network_config = network_config
    self.checkpoint_path = self._make_checkpoint_path()
    self.saver = self._make_saver()
    self.summary_writer = self._make_summary_writer(graph_def)
    self.summary_op = self._make_summary_op()

  def _make_checkpoint_path(self):
    # make checkpoint path with all the flags specifing different directories
 
    # run through all params and add them to the base path
    base_path = self.network_dir + '/' + self.network_name
    for k, v in self.network_config.items():
      base_path += '/' + k + '.' + str(v)
    return base_path

  def _make_saver(self):
    variables = tf.global_variables()
    saver = tf.train.Saver(variables, max_to_keep=1)
    return saver

  def _make_summary_writer(self, graph_def):
    summary_writer = tf.summary.FileWriter(self.checkpoint_path, graph_def=graph_def)
    return summary_writer

  def _make_summary_op(self):
    summary_op = tf.summary.merge_all()
    return summary_op

  def _list_all_checkpoints(self):
    # get a list off all the checkpoint directorys

    # run through all params and add them to the base path
    paths = []
    for root, dirnames, filenames in os.walk(self.network_dir):
      for filename in fnmatch.filter(filenames, 'checkpoint'):
        paths.append(root[len(self.network_dir)+1:])
    return paths

  def load_checkpoint(self, sess, maybe_remove_prev=True):
    ckpt = tf.train.get_checkpoint_state(self.checkpoint_path)
    if ckpt is not None:
      print("init from " + ckpt.model_checkpoint_path)
      try:
        self.saver.restore(sess, ckpt.model_checkpoint_path)
      except:
        if maybe_remove_prev:
          tf.gfile.DeleteRecursively(self.checkpoint_path)
          tf.gfile.MakeDirs(self.checkpoint_path)
        print("there was a problem using variables in checkpoint, random init will be used instead")

  def save_checkpoint(self, sess, global_step):
    save_path = os.path.join(self.checkpoint_path, 'model.ckpt')
    self.saver.save(sess, save_path, global_step=global_step)  

  def save_summary(self, sess, feed_dict, global_step):
    summary_str = sess.run(self.summary_op, feed_dict=feed_dict)
    self.summary_writer.add_summary(summary_str, global_step) 

    



