from tutorials import mnist_blackbox
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS


if __name__ == '__main__':
    # Reduce number of data augmentations for faster runtime
    FLAGS.data_aug = 4

    # Run the MNIST tutorial on a dataset of reduced size
    test_dataset_indices = {'train_start': 0,
                            'train_end': 10000,
                            'test_start': 0,
                            'test_end': 1666}
    report = mnist_blackbox.mnist_blackbox(**test_dataset_indices)

    # Check accuracy values contained in the AccuracyReport object
    assert report['bbox'] > 0.85
    assert report['sub'] > 0.55
    assert report['bbox_on_sub_adv_ex'] < 0.65
