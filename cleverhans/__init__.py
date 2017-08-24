from cleverhans.devtools.version import dev_version

# Attach a hex digest to the version string to keep track of changes
# in the development branch
__version__ = '2.0.0-' + dev_version()
