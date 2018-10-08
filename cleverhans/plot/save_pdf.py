"""
The save_pdf function.
"""
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot


def save_pdf(path):
  """
  Saves a pdf of the current matplotlib figure.

  :param path: str, filepath to save to
  """

  pp = PdfPages(path)
  pp.savefig(pyplot.gcf())
  pp.close()
