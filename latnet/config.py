
import ConfigParser as configparser
import argparse
import os
import re


class LatNetConfig(argparse.Namespace):
    """Specifies the configuration of a LatNet simulation.

    This class carries all settings, specified programmatically from a script
    or manually via command line parameters.
    """
    @property
    def output_required(self):
        return self.output or self.mode == 'visualization'

    @property
    def needs_iteration_num(self):
        return self.time_dependence or self.access_pattern == 'AA'


class LatNetConfigParser(object):
    def __init__(self, description=None):
        desc = "LatNet simulator."
        if description is not None:
            desc += " " + description

        self._parser = argparse.ArgumentParser(description=desc)
        self._parser.add_argument('-q', '--quiet',
                help='reduce verbosity', action='store_true', default=False)
        self.config = LatNetConfig()

    def add_group(self, name):
        return self._parser.add_argument_group(name)

    def set_defaults(self, defaults):
        for option in defaults.keys():
            assert self._parser.get_default(option) is not None,\
                    'Unknown option "{0}" specified in update_defaults()'.format(option)
        return self._parser.set_defaults(**defaults)

    def parse(self, args, internal_defaults=None):
        config = configparser.ConfigParser()
        config.read(['/etc/sailfishrc', os.path.expanduser('~/.sailfishrc'),
                '.sailfishrc'])

        # Located here for convenience, so that this attribute can be referenced in
        # the symbolic expressions module even for LatNet models where this option is not
        # supported.
        self.config.incompressible = False
        try:
            self._parser.set_defaults(**dict(config.items('main')))
        except configparser.NoSectionError:
            pass

        if internal_defaults is not None:
            self._parser.set_defaults(**internal_defaults)

        self._parser.parse_args(args=args, namespace=self.config)

        # Additional internal config options, not settable via
        # command line parameters.
        self.config.relaxation_enabled = True
        self.config.propagation_enabled = True

        # Indicates whether the simulation has any DynamicValues which are
        # time-dependent.
        self.config.time_dependence = False

        # Indicates whether the simulation has any DynamicValues which are
        # location-dependent.
        self.config.space_dependence = False
        self.config.unit_test = False
        return self.config



