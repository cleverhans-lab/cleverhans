"""Script which downloads dataset images using multiple processes.

Usage:
  python download_images_multiprocessed.py --input_file=INPUT_FILE --output_dir=IMAGES_DIR

where:
  INPUT_FILE is input csv file with dataset description, i.e. dev_dataset.csv
  IMAGES_DIR is output directory where all images should be downloaded

Example:
  # create directory for images
  mkdir images
  # download images declared in dev_dataset.csv using multiple processes
  python download_images_multiprocessed.py --input_file=dev_dataset.csv --output_dir=images


Dependencies:
  Python 2.7 or higher.
  Pillow library: https://python-pillow.org/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import csv
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial

from PIL import Image
from io import BytesIO

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen


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
        url_file = urlopen(url)
        if url_file.getcode() != 200:
            return False
        image_buffer = url_file.read()
        # Crop, resize and save image
        image = Image.open(BytesIO(image_buffer)).convert('RGB')
        w = image.size[0]
        h = image.size[1]
        image = image.crop(
            (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
        image = image.resize((299, 299), resample=Image.ANTIALIAS)
        image.save(output_filename)
    except IOError:
        return False
    return True


def get_images(row, output_dir):
    if not download_image(image_id=row[0],
                          url=row[1],
                          x1=float(row[2]),
                          y1=float(row[3]),
                          x2=float(row[4]),
                          y2=float(row[5]),
                          output_dir=output_dir):
        print("Download failed: " + str(row[0]))


def main():
    input_filename, output_dir = parse_args()

    if not os.path.isdir(os.path.join(os.getcwd(), output_dir)):
        print("Output directory {} does not exist".format(output_dir))
        sys.exit()

    with open(input_filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)[1:]
    pool = ThreadPool(8)
    partial_get_images = partial(get_images, output_dir=output_dir)
    for i, _ in enumerate(pool.imap_unordered(partial_get_images, rows), 1):
        sys.stderr.write('\rDownloaded {0} images'.format(i + 1))
    pool.close()
    pool.join()


if __name__ == '__main__':
    main()
