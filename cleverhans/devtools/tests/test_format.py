"""
Unit tests for format checking
"""

from __future__ import print_function

from nose.plugins.skip import SkipTest

import os

import cleverhans
from cleverhans.devtools.tests.docscrape import docstring_errors
from cleverhans.devtools.list_files import list_files
from pycodestyle import StyleGuide

whitelist_pep8 = [
]

whitelist_docstrings = [
]


def test_format_pep8():
    """
    Test if pep8 is respected.
    """
    pep8_checker = StyleGuide()
    files_to_check = []
    for path in list_files(".py"):
        rel_path = os.path.relpath(path, cleverhans.__path__[0])
        if rel_path in whitelist_pep8:
            continue
        else:
            files_to_check.append(path)
    report = pep8_checker.check_files(files_to_check)
    if report.total_errors > 0:
        raise AssertionError("PEP8 Format not respected")


def print_files_information_pep8():
    """
    Print the list of files which can be removed from the whitelist and the
    list of files which do not respect PEP8 formatting that aren't in the
    whitelist
    """
    infracting_files = []
    non_infracting_files = []
    pep8_checker = StyleGuide(quiet=True)
    for path in list_files(".py"):
        number_of_infractions = pep8_checker.input_file(path)
        rel_path = os.path.relpath(path, cleverhans.__path__[0])
        if number_of_infractions > 0:
            if rel_path not in whitelist_pep8:
                infracting_files.append(path)
        else:
            if rel_path in whitelist_pep8:
                non_infracting_files.append(path)
    print("Files that must be corrected or added to whitelist:")
    for file in infracting_files:
        print(file)
    print("Files that can be removed from whitelist:")
    for file in non_infracting_files:
        print(file)


def test_format_docstrings():
    """
    Test if docstrings are well formatted.
    """
    # Disabled for now
    return True

    try:
        verify_format_docstrings()
    except SkipTest as e:
        import traceback
        traceback.print_exc(e)
        raise AssertionError(
            "Some file raised SkipTest on import, and inadvertently"
            " canceled the documentation testing."
        )


def verify_format_docstrings():
    """
    Implementation of `test_format_docstrings`. The implementation is
    factored out so it can be placed inside a guard against SkipTest.
    """
    format_infractions = []

    for path in list_files(".py"):
        rel_path = os.path.relpath(path, cleverhans.__path__[0])
        if rel_path in whitelist_docstrings:
            continue
        try:
            format_infractions.extend(docstring_errors(path))
        except Exception as e:
            format_infractions.append(["%s failed to run so format cannot "
                                       "be checked. Error message:\n %s" %
                                       (rel_path, e)])

    if len(format_infractions) > 0:
        msg = "\n".join(':'.join(line) for line in format_infractions)
        raise AssertionError("Docstring format not respected:\n%s" % msg)


if __name__ == "__main__":
    print_files_information_pep8()
