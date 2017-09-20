"""
Utility functions for keeping track of the version of CleverHans.

These functions provide a finer level of granularity than the
manually specified version string attached to each release.
"""
import hashlib
from cleverhans.devtools.list_files import list_files


def dev_version():
    """
    Returns a hexdigest of all the python files in the module.
    """

    m = hashlib.md5()
    py_files = sorted(list_files(suffix=".py"))
    for filename in py_files:
        with open(filename, 'rb') as f:
            content = f.read()
        m.update(content)
    return m.hexdigest()
