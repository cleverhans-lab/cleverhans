"""Script which downloads dataset images.

Usage:
  python download_images.py --input_file=INPUT_FILE --output_dir=IMAGES_DIR

where:
  INPUT_FILE is input csv file with dataset description, i.e. dev_dataset.csv
  IMAGES_DIR is output directory where all images should be downloaded

Example:
  # create directory for images
  mkdir images
  # download images declared in dev_dataset.csv
  python download_images.py --input_file=dev_dataset.csv --output_dir=images


Dependencies:
  Python 2.7 or higher.
  Pillow library: https://python-pillow.org/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cStringIO
import csv
import os
import sys
import urllib2

from PIL import Image


def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser(
      description='Tool to download dataset images.')
  parser.add_argument('--input_file', required=True,
                      help='Location of dataset.csv')
  parser.add_argument('--output_dir', required=True,
                      help='Output path to download images')
  args = parser.parse_args()
  return args.input_file, args.output_dir


def download_image(image_id, url, x1, y1, x2, y2, output_dir):
  """Downloads one image, crops it, resizes it and saves it locally."""
  output_filename = os.path.join(output_dir, image_id + '.png')
  if os.path.exists(output_filename):
    # Don't download image if it's already there
    return True
  try:
    # Download image
    url_file = urllib2.urlopen(url)
    if url_file.getcode() != 200:
      return False
    image_buffer = url_file.read()
    # Crop, resize and save image
    image = Image.open(cStringIO.StringIO(image_buffer)).convert('RGB')
    w = image.size[0]
    h = image.size[1]
    image = image.crop((int(x1*w), int(y1*h), int(x2*w), int(y2*h)))
    image = image.resize((299, 299), resample=Image.ANTIALIAS)
    image.save(output_filename)
  except IOError:
    return False
  return True


def main():
  input_filename, output_dir = parse_args()
  failed_to_download = set()
  with open(input_filename) as input_file:
    reader = csv.reader(input_file)
    header_row = next(reader)
    try:
      row_idx_image_id = header_row.index('ImageId')
      row_idx_url = header_row.index('URL')
      row_idx_x1 = header_row.index('x1')
      row_idx_y1 = header_row.index('y1')
      row_idx_x2 = header_row.index('x2')
      row_idx_y2 = header_row.index('y2')
    except ValueError as e:
      print('One of the columns was not found in the source file: ', e.message)

    for idx, row in enumerate(reader):
      if len(row) < len(header_row):
        # skip partial or empty lines
        continue
      if not download_image(image_id=row[row_idx_image_id],
                            url=row[row_idx_url],
                            x1=float(row[row_idx_x1]),
                            y1=float(row[row_idx_y1]),
                            x2=float(row[row_idx_x2]),
                            y2=float(row[row_idx_y2]),
                            output_dir=output_dir):
        failed_to_download.add(row[row_idx_image_id])
      sys.stdout.write('\rDownloaded {0} images'.format(idx + 1))
      sys.stdout.flush()

    print()
    if failed_to_download:
      print('\nUnable to download images with the following IDs:')
      for image_id in failed_to_download:
        print(image_id)


if __name__ == '__main__':
  main()
