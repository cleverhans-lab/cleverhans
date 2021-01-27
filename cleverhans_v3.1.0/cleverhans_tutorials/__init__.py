"""
The CleverHans tutorials.
While mostly designed to be run as standalone scripts, the tutorials together also form an importable module.
Module importation is mostly intended to support writing unit tests of the tutorials themselves, etc.
The tutorial code is not part of our API contract and can change rapidly without warning.
"""
import os
import warnings

import cleverhans


def check_installation(cur_file):
  """Warn user if running cleverhans from a different directory than tutorial."""
  cur_dir = os.path.split(os.path.dirname(os.path.abspath(cur_file)))[0]
  ch_dir = os.path.split(cleverhans.__path__[0])[0]
  if cur_dir != ch_dir:
    warnings.warn("It appears that you have at least two versions of "
                  "cleverhans installed, one at %s and one at"
                  " %s. You are running the tutorial script from the "
                  "former but python imported the library module from the "
                  "latter. This may cause errors, for example if the tutorial"
                  " version is newer than the library version and attempts to"
                  " call new features." % (cur_dir, ch_dir))
