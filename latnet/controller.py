
from latnetwork import LatNet
from data_queue import DataQueue
from config import LatNetConfigParser
from loss import Loss

class LatNetController(object):
    """Controls the execution of a LN simulation."""

    def __init__(self, latnet_sim=None, sailfish_sim=None):

      self._config_parser = config.LatNetConfigParser()
      self._sailfish_sim = sailfish_sim
      self._latnet_sim = latnet_sim

     
      group = self._config_parser.add_group('Basic stuff')
      group.add_argument('--sailfish_sim_dir', help='train mode', type=str,
            choices=[], default='/data/sailfish_sim')
      group.add_argument('--latnet_network_dir', help='all mode', type=str,
            choices=[], default='./network_checkpoint')
      group.add_argument('--latnet_sim_dir', help='eval mode', type=str,
            choices=[], default='./latnet_simulation')
      group.add_argument('--network_name', help='all mode', type=str,
            choices=[], default='basic_network')
      group.add_argument('--mode', help='all mode', type=str,
            choices=['train', 'eval'], default='train')
      group.add_argument('--shape', help='all mode', type=str,
            choices=['train', 'eval'], default='train')
      group.add_argument('--lattice_q', help='all mode', type=int,
            choices=[9], default=9)

      self.network = LatNet(self.config)

    def _finish_simulation(self, subdomains, summary_receiver):
      pass

    def save_config(self, subdomains):
      pass

    def _load_sim(self):
      pass

    def train(self):

      # parse config
      self.config = self._config_parser.parse(
          args, internal_defaults={'quiet': True} if hasattr(
              builtins, '__IPYTHON__') else None)

      with tf.Graph().as_default():

        # global step counter
        global_step = tf.get_variable('global_step', [], 
                          initializer=tf.constant_initializer(0), trainable=False)

        # make inputs
        self.inputs = Inputs(self.config)
        self.state = self.inputs.state()
        self.boundary = self.inputs.boundary()

        # make network pipe
        self.state_out = self.network.unroll(self.state, self.boundary)

        # make loss
        self.loss = Loss(config)
        self.total_loss = self.loss.mse(self.state, self.state_out)

        # make train op
        all_params = tf.trainable_variables()
        self.optimizer = Optimizer(self.config)
        self.optimizer.compute_gradients(self.total_loss, all_params)
        self.train_op = self.optimizer.train_op(all_params)

        # start session 
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.config.gpu_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        init = tf.global_variables_initializer()
        sess.run(init)

        # make saver
        self.saver = Saver(self.config, self.network.config)
        self.saver.load_checkpoint()

        # construct dataset
        self.dataset = DataQueue(self.config, sailfish_sim)
        self.dataset.create_dataset()

        # train
        for i in xrange(sess.run(global_step), self.config.train_iterations):
          _, l = sess.run([self.train_op self.total_loss], 
                          feed_dict=self.dataset.minibatch(self.state, self.boundary))
          if i % 100:
            print("current loss is " + str(l))
            print("current step is " + str(i))

          if i % 1000:
            self.saver.save_summary(sess, self.dataset.minibatch(self.state, self.boundary), global_step)
            self.save_checkpoint(global_step)

    def run(self, ignore_cmdline=False):
      pass


