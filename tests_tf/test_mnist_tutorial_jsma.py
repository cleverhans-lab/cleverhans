from tutorials import mnist_tutorial_jsma


def test_mnist_tutorial_jsma():
    # Run the MNIST tutorial on a dataset of reduced size
    # and disable visualization.
    jsma_tutorial_fnargs = {'train_start': 0,
                            'train_end': 10000,
                            'test_start': 0,
                            'test_end': 1666,
                            'viz_enabled': False}
    report = mnist_tutorial_jsma.mnist_tutorial_jsma(**jsma_tutorial_fnargs)

    # Check accuracy values contained in the AccuracyReport object
    assert report.clean_train_clean_eval > 0.85
    assert report.clean_train_adv_eval < 0.15

    # There is no adversarial training in the JSMA tutorial
    assert report.adv_train_clean_eval == 0.
    assert report.adv_train_adv_eval == 0.


if __name__ == '__main__':
    test_mnist_tutorial_jsma()
