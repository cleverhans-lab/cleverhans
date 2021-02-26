"""Defaults for pyplot

Import this file to set some default parameters for pyplot.
These are designed to make the plot look good in the context of a LaTeX
document.
If you have installed the Computer Modern fonts
( ftp://canopus.iacp.dvo.ru/pub/Font/cm_unicode/cm-unicode-0.6.3a-otf.tar.gz )
these defaults will use them, so that text in your pyplot figures will
match text in the rest of your document.
If you do not have those fonts installed, pyplot commands will still work.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
from matplotlib import pyplot
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
matplotlib.rcParams['text.usetex'] = True
pyplot.rcParams['pdf.fonttype'] = 42
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.serif'] = 'CMU Serif'
pyplot.rcParams['font.size'] = 8
# Note: if you get an error, delete fontList.cache
