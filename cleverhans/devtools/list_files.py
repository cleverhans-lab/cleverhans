"""Code for listing files that belong to the library."""
import logging
import cleverhans
import os
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"

logger = logging.getLogger(__name__)


def list_files(suffix=""):
    """
    Returns a list of all files in pylearn2 with the given suffix.

    Parameters
    ----------
    suffix : str

    Returns
    -------

    file_list : list
        A list of all files in pylearn2 whose filepath ends with `suffix`
    """

    cleverhans_path, = cleverhans.__path__

    file_list = _list_files(cleverhans_path, suffix)

    return file_list


def _list_files(path, suffix=""):
    """
    .. todo::

        WRITEME

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
        for l in lists:
            for elem in l:
                flattened.append(elem)
        return flattened
    else:
        assert os.path.exists(path), "couldn't find file '%s'" % path
        if path.endswith(suffix):
            return [path]
        return []


if __name__ == '__main__':
    # Print all .py files in the library
    result = list_files('.py')
    for path in result:
        logger.info(path)
