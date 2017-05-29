from tutorials import mnist_tutorial_tf


if __name__ == '__main__':
    # Run the MNIST tutorial on a dataset of reduced size
    test_dataset_indices = {'train_start': 0,
                            'train_end': 10000,
                            'test_start': 0,
                            'test_end': 1666}
    test_results = mnist_tutorial_tf.mnist_tutorial(**test_dataset_indices)

    # Check accuracy values
    assert test_results[0] > 0.85
    assert test_results[1] < 0.06
    assert test_results[2] > 0.85
    assert test_results[3] > 0.190
