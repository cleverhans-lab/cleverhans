"""
Unit tests for format checking
"""

from __future__ import print_function


import os
import subprocess

import cleverhans
from cleverhans.devtools.list_files import list_files
from cleverhans.utils import shell_call

# Enter a manual list of files that are allowed to violate PEP8 here
whitelist_pep8 = [
    # This file is broken but could be fixed
    "../examples/multigpu_advtrain/test_attack_multigpu.py"
]

all_py_files = list_files('.py')

def update_whitelist():
  """Add files to the whitelist"""
  global whitelist_pep8
  # We don't want to test RL-attack because it has so many dependencies
  # not used elsewhere, and pylint wants to import them all
  whitelist_pep8.extend([os.path.relpath(path, cleverhans.__path__[0])
                         for path in all_py_files if "RL-attack" in path])
  # Similarly, we don't want to require robust_vision_benchmark installed
  whitelist_pep8.extend([os.path.relpath(path, cleverhans.__path__[0])
                         for path in all_py_files
                         if "robust_vision_benchmark" in path])
  # Similarly, we don't want to require that cloud be installed
  whitelist_pep8.extend([os.path.relpath(path, cleverhans.__path__[0])
                         for path in all_py_files
                         if "cloud_client" in path])
  # This example has more dependencies too
  whitelist_pep8.extend([os.path.relpath(path, cleverhans.__path__[0])
                         for path in all_py_files
                         if "facenet_adversarial_faces" in path])
  # This too
  whitelist_pep8.extend([os.path.relpath(path, cleverhans.__path__[0])
                         for path in all_py_files
                         if "madry_lab_challenges" in path])
  # This code is no longer compatible with new versions of cleverhans / scipy and will be removed soon
  whitelist_pep8.extend([os.path.relpath(path, cleverhans.__path__[0])
                         for path in all_py_files
                         if "nips17_adversarial_competition" in path])


update_whitelist()


whitelist_docstrings = [
]


def test_format_pep8():
  """
  Test if pep8 is respected.
  """
  files_to_check = []
  module_dir = cleverhans.__path__[0]
  for path in all_py_files:
    rel_path = os.path.relpath(path, module_dir)
    if rel_path in whitelist_pep8:
      continue
    else:
      files_to_check.append(path)
  repo_dir = os.path.join(module_dir, os.pardir)
  rcpath = os.path.join(repo_dir, '.pylintrc')
  assert os.path.exists(rcpath)

  # We must run pylint via the command line and subprocess because of
  # problems with the pylint module.
  # The documentation claims you can run it as a python module, but
  # the documentation is wrong: https://github.com/PyCQA/pylint/issues/1870
  # If you run the version described in the linked issue, pylint
  # calls sys.exit once it is done, so it kills the test.

  # Running all files in one pylint command is important for 2 reasons:
  # 1) Correctness: pylint can detect issues that require access to multiple
  #    files, such as cyclic imports
  # 2) Speed: pylint imports modules for deep analysis, so if you run
  #    multiple subprocesses each needs to re-import tensorflow.
  # On Ian's laptop, pylint takes about 10s per file to run on the repo,
  # and there are about 90 files as of the writing of this comment.
  # Running pylint on all files simultaneously takes about 70s, so it
  # is a little better than a 10X speedup.

  # Running multiple jobs in parallel helps but far less than linearly.
  # On Ian's 4-core laptop, running 4 jobs drops the runtime from 70s
  # to 45s.
  # Some of the work is I/O, so it actually makes some sense to run
  # more jobs than cores. On Ian's 4-core laptop, running 8 jobs drops
  # the runtime to 40s.
  # There's a further complication though: I think each job needs to
  # redo imports, so the total amount of work to do increases with
  # the number of jobs. On Ian's laptop, using 64 jobs causes the
  # runtime to increase to 220s. There is not an obvious simple
  # formula like "use one job per CPU core" or "use way more jobs
  # than cores to saturate I/O". For now I'm hoping that 8 will be
  # a reasonable default: it gets good performance on my laptop,
  # and on machines with fewer than 4 cores there should still be
  # a benefit to not being blocked on I/O.

  try:
    shell_call(['pylint', '--rcfile', rcpath, '--jobs', '8'] + files_to_check)
  except subprocess.CalledProcessError as e:
    raise ValueError(e.output.decode("utf-8"))


if __name__ == "__main__":
  test_format_pep8()
