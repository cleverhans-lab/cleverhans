from tutorials import mnist_tutorial_tf


if __name__ == '__main__':
    # Run the MNIST tutorial on a dataset of reduced size
    test_dataset_indices = {'train_start': 0,
                            'train_end': 10000,
                            'test_start': 0,
                            'test_end': 1666}
    report = mnist_tutorial_tf.mnist_tutorial(**test_dataset_indices)

    # Check accuracy values contained in the AccuracyReport object
    assert report.clean_train_clean_eval > 0.85
    assert report.clean_train_adv_eval < 0.06
    assert report.adv_train_clean_eval > 0.85
    assert report.adv_train_adv_eval > 0.190
