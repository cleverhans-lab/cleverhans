import unittest


class TestMNISTTutorialKerasTF(unittest.TestCase):
    def test_mnist_tutorial_keras_tf(self):
        from tutorials import mnist_tutorial_keras_tf

        # Run the MNIST tutorial on a dataset of reduced size
        test_dataset_indices = {'train_start': 0,
                                'train_end': 5000,
                                'test_start': 0,
                                'test_end': 333,
                                'nb_epochs': 3,
                                'train_dir': '/tmp',
                                'filename': 'mnist.ckpt',
                                'load_model': False}
        report = mnist_tutorial_keras_tf.mnist_tutorial(**test_dataset_indices)

        # Check accuracy values contained in the AccuracyReport object
        self.assertTrue(report.clean_train_clean_eval > 0.90)
        self.assertTrue(report.clean_train_adv_eval < 0.05)
        self.assertTrue(report.adv_train_clean_eval > 0.90)
        self.assertTrue(report.adv_train_adv_eval > 0.30)


if __name__ == '__main__':
    unittest.main()
