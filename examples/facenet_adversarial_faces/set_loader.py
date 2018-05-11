import lfw
import facenet

import numpy as np


pairs_path = "datasets/lfw/pairs.txt"
testset_path = "datasets/lfw/lfw_mtcnnpy_160"
file_extension = 'png'
image_size = 160


def load_testset(size):
    # Load images paths and labels
    pairs = lfw.read_pairs(pairs_path)
    paths, labels = lfw.get_paths(testset_path, pairs, file_extension)

    # Random choice
    permutation = np.random.choice(len(labels), size, replace=False)
    paths_batch_1 = []
    paths_batch_2 = []

    for index in permutation:
        paths_batch_1.append(paths[index * 2])
        paths_batch_2.append(paths[index * 2 + 1])

    labels = np.asarray(labels)[permutation]
    paths_batch_1 = np.asarray(paths_batch_1)
    paths_batch_2 = np.asarray(paths_batch_2)

    # Load images
    faces1 = facenet.load_data(paths_batch_1, False, False, image_size)
    faces2 = facenet.load_data(paths_batch_2, False, False, image_size)

    # Change pixel values to 0 to 1 values
    min_pixel = min(np.min(faces1), np.min(faces2))
    max_pixel = max(np.max(faces1), np.max(faces2))
    faces1 = (faces1 - min_pixel) / (max_pixel - min_pixel)
    faces2 = (faces2 - min_pixel) / (max_pixel - min_pixel)

    # Convert labels to one-hot vectors
    onehot_labels = []
    for index in range(len(labels)):
        if labels[index]:
            onehot_labels.append([1, 0])
        else:
            onehot_labels.append([0, 1])

    return faces1, faces2, np.array(onehot_labels)
