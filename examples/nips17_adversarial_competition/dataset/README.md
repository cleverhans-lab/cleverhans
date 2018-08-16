
# Dataset for adversarial competition.

Two datasets were used for the competition:

* **DEV** dataset which was available for the development and experimenting
  during the competition.
* **FINAL** dataset which was kept secret until after the competition
  and was used for final scoring.

Both datasets are composed from publicly available images which were posted
online under CC-BY license.

## Dataset format

Each dataset is described by
[CSV](https://en.wikipedia.org/wiki/Comma-separated_values)
file with URLs of the images, their bounding boxes,
classification labels and some other metadata.

`dev_dataset.csv` contains description of DEV dataset and
`final_dataset.csv` contains description of FINAL dataset.

File with the dataset description contains following columns:

* **ImageId** - id of the image.
* **URL** - URL of the image.
* **x1**, **y1**, **x2**, **y2** - bounding box of the area of interest in
  the image. Bounding box is relative, which means that all coordinates are
  between 0 and 1.
* **TrueLabel** - true label of the image.
* **TargetClass** - label for targeted adversarial attack.
* **OriginalLandingURL** - original landing page where this image was found.
* **License** - licence under which image was distributed by author.
* **Author** - author of the image.
* **AuthorProfileURL** - URL of the author's profile.

Dataset is labelled with
[ImageNet](http://www.image-net.org/challenges/LSVRC/2012/) labels.
Specific values of labels are compatible with pre-trained Inception models,
which are available as a part of
[TF-Slim](https://github.com/tensorflow/models/tree/master/slim).
In particular pre-trained Inception v3 and InceptionResnet v2
could be used to classify dataset with high accuracy.
Note that TF-Slim ImageNet labels are from 1 to 1000 and label 0 is unused.

## Downloading images

File with the dataset description contains only URLs of the images.
Actual images have to be downloaded before being used for experiments.

`download_images.py` is a Python program which downloads images for all
records in the dataset file. Usage:

```
# Replace CSV_FILE with path to dev_dataset.csv or final_dataset.csv
CSV_FILE=dev_dataset.csv
# Replace OUTPUT_DIR with path to directory where all images should be stored
OUTPUT_DIR=images
# Download images
python download_images.py --input_file=${CSV_FILE} --output_dir=${OUTPUT_DIR}
```
To accelerate the image downloading, multiple threads are spawned (the default is CPU_COUNT + 1).
You can change this setting using --threads flag to any other positive value.

All downloaded images will be cropped according to the bounding boxes specified in
the dataset file and resized to 299x299 pixels.
Each image will be saved in PNG format with filename `IMAGE_ID.png`
where `IMAGE_ID` is the id of the image from the dataset file.
