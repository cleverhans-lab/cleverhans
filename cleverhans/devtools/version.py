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

  md5_hash = hashlib.md5()
  py_files = sorted(list_files(suffix=".py"))
  if not py_files:
    return ''
  for filename in py_files:
    with open(filename, 'rb') as fobj:
      content = fobj.read()
    md5_hash.update(content)
  return md5_hash.hexdigest()


def append_dev_version(release_version):
  """
  If dev version is not empty appends it to release_version.
  """

  dev_version_value = dev_version()
  if dev_version_value:
    return release_version + '-' + dev_version_value
  else:
    return release_version
