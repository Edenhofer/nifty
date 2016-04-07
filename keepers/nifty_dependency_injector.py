# -*- coding: utf-8 -*-

import sys

class dependency_injector(object):
    def __init__(self, modules=[]):
        self.registry = {}
        map(self.register, modules)

    def get(self, x):
        return self.registry.get(x)

    def __getitem__(self, x):
        return self.registry.__getitem__(x)

    def __contains__(self, x):
        return self.registry.__contains__(x)

    def __iter__(self):
        return self.registry.__iter__()

    def __getattr__(self, x):
        return self.registry.__getattribute__(x)

    def register(self, module_name, check=None):
        if isinstance(module_name, tuple):
            module_name, key_name = (str(module_name[0]), str(module_name[1]))
        else:
            module_name = str(module_name)
            key_name = module_name

        try:
            loaded_module = sys.modules[module_name]
        except KeyError:
            try:
                loaded_module = recursive_import(module_name)
            except ImportError:
                pass

        if loaded_module is not None:
            if check is not None:
                check_passed = check(loaded_module)
            else:
                check_passed = True

            if check_passed is True:
                self.registry[key_name] = loaded_module

    def unregister(self, module_name):
        try:
            del self.registry['module_name']
        except KeyError:
            pass

def recursive_import(name):
    m = __import__(name)
    for n in name.split(".")[1:]:
        m = getattr(m, n)
    return m

