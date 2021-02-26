"""Code for listing files that belong to the library."""
import os
import cleverhans


def list_files(suffix=""):
  """
  Returns a list of all files in CleverHans with the given suffix.

  Parameters
  ----------
  suffix : str

  Returns
  -------

  file_list : list
      A list of all files in CleverHans whose filepath ends with `suffix`.
  """

  cleverhans_path = os.path.abspath(cleverhans.__path__[0])
  # In some environments cleverhans_path does not point to a real directory.
  # In such case return empty list.
  if not os.path.isdir(cleverhans_path):
    return []
  repo_path = os.path.abspath(os.path.join(cleverhans_path, os.pardir))
  file_list = _list_files(cleverhans_path, suffix)

  extra_dirs = ['cleverhans_tutorials', 'examples', 'scripts', 'tests_tf', 'tests_pytorch']

  for extra_dir in extra_dirs:
    extra_path = os.path.join(repo_path, extra_dir)
    if os.path.isdir(extra_path):
      extra_files = _list_files(extra_path, suffix)
      extra_files = [os.path.join(os.pardir, path) for path in extra_files]
      file_list = file_list + extra_files

  return file_list


def _list_files(path, suffix=""):
  """
  Returns a list of all files ending in `suffix` contained within `path`.

  Parameters
  ----------
  path : str
      a filepath
  suffix : str

  Returns
  -------
  l : list
      A list of all files ending in `suffix` contained within `path`.
      (If `path` is a file rather than a directory, it is considered
      to "contain" itself)
  """
  if os.path.isdir(path):
    incomplete = os.listdir(path)
    complete = [os.path.join(path, entry) for entry in incomplete]
    lists = [_list_files(subpath, suffix) for subpath in complete]
    flattened = []
    for one_list in lists:
      for elem in one_list:
        flattened.append(elem)
    return flattened
  else:
    assert os.path.exists(path), "couldn't find file '%s'" % path
    if path.endswith(suffix):
      return [path]
    return []
