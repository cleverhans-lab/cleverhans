import unittest


class TestMNISTTutorialTF(unittest.TestCase):
    def test_mnist_tutorial_tf(self):
        from tutorials import mnist_tutorial_pure_tf

        # Run the MNIST tutorial on a dataset of reduced size
        test_dataset_indices = {'train_start': 0,
                                'train_end': 5000,
                                'test_start': 0,
                                'test_end': 333}
        report = mnist_tutorial_pure_tf.mnist_tutorial(**test_dataset_indices)

        # Check accuracy values contained in the AccuracyReport object
        self.assertTrue(report.clean_train_clean_eval > 0.85)
        self.assertTrue(report.clean_train_adv_eval < 0.07)
        self.assertTrue(report.adv_train_clean_eval > 0.8)
        self.assertTrue(report.adv_train_adv_eval > 0.14)


if __name__ == '__main__':
    unittest.main()
