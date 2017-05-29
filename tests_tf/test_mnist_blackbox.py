from tutorials import mnist_blackbox


def test_mnist_blackbox():
    # Run the MNIST tutorial on a dataset of reduced size
    # Reduce number of data augmentations for faster runtime
    mnist_blackbox_args = {'train_start': 0,
                           'train_end': 10000,
                           'test_start': 0,
                           'test_end': 1666,
                           'data_aug': 4}
    report = mnist_blackbox.mnist_blackbox(**mnist_blackbox_args)

    # Check accuracy values contained in the AccuracyReport object
    assert report['bbox'] > 0.85
    assert report['sub'] > 0.55
    assert report['bbox_on_sub_adv_ex'] < 0.65


if __name__ == '__main__':
    test_mnist_blackbox()
