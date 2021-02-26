"""The CleverHans adversarial example library"""
from cleverhans.devtools.version import append_dev_version

# If possible attach a hex digest to the version string to keep track of
# changes in the development branch
__version__ = append_dev_version('3.0.1')
