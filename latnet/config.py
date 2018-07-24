
import ConfigParser as configparser
import argparse
import os
import re


class LatNetConfig(argparse.Namespace):
  pass

class LatNetConfigParser(object):
    def __init__(self):
        self._parser = argparse.ArgumentParser(description=desc)
        self.config = LatNetConfig()

    def add_group(self, name):
        return self._parser.add_argument_group(name)

    def set_defaults(self, defaults):
        for option in defaults.keys():
            assert self._parser.get_default(option) is not None,\
                    'Unknown option "{0}" specified in update_defaults()'.format(option)
        return self._parser.set_defaults(**defaults)

    def parse(self, args, internal_defaults=None):
        self._parser.parse_args(args=args, namespace=self.config)
        return self.config

NONSAVE_CONFIGS = ['mode', 'run_mode', 'latnet_network_dir', 'input_shape',
                           'input_cshape', 'save_network_freq', 'seq_length',
                           'batch_size', 'gpus', 'train_iters', 'train_sim_dir',
                           'gpu_fraction', 'num_simulations', 'max_queue', 'sim_shape',
                           'num_iters', 'sim_restore_iter', 'sim_dir', 'sim_save_every',
                           'compare', 'save_format', 'save_cstate', 'checkpoint_from',
                           'restore_from', 'max_sim_iters', 'restore_geometry', 'scr_scale',
                           'debug_sailfish', 'every', 'unit_test', 'propagation_enabled',
                           'time_dependence', 'space_dependence', 'incompressible', 
                           'relaxation_enabled', 'quiet', 'periodic_x', 'domain_name',
                           'periodic_y', 'periodic_z', 'start_num_data_points_train', 
                           'start_num_data_points_test', 'train_autoencoder']



