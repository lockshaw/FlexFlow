import argparse
from configlib.cmake_bool import CMakeBool
import os

class EnvStore(argparse.Action):
    def __init__(self, *args, **kwargs):
        self.env_sources = kwargs.pop('env_sources', [])
        self._has_logged = False
        super().__init__(*args, **kwargs)

    @property
    def default(self):
        for env_source in self.env_sources:
            env_value = os.environ.get(env_source, '')
            if env_value != '':
                if not self._has_logged:
                    _l.info('Pulling args.%s from environment variable %s', self.dest, env_source)
                    self._has_logged = True
                if self.type is not None:
                    env_value = self.type(env_value)
                return env_value
        default = self._env_store_default
        if default is not None and self.type is not None:
            default = self.type(default)
        return default

    @default.setter
    def default(self, value):
        self._env_store_default = value  # can't use self._default because it is used in the parent class

    #@property
    #def help(self):
    #    return self._help

    #@help.setter
    #def help(self, value):
    #    self._default = value

class EnvStoreConst(EnvStore):
    def __init__(self, *args, **kwargs):
        self._const = kwargs.pop('const')
        kwargs['nargs'] = 0
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, self._const)

def env_store(*args, **kwargs):
    return EnvStore(*args, **kwargs)

def env_store_false(*args, **kwargs):
    kwargs['default'] = True
    kwargs['const'] = CMakeBool(False)
    return EnvStoreConst(*args, **kwargs)

def env_store_true(*args, **kwargs):
    kwargs['default'] = False
    kwargs['const'] = CMakeBool(True)
    return EnvStoreConst(*args, **kwargs)

