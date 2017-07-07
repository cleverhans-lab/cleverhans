import unittest


class TestMNISTTutorialJSMA(unittest.TestCase):
    def test_mnist_tutorial_jsma(self):
        from tutorials import mnist_tutorial_jsma

        # Run the MNIST tutorial on a dataset of reduced size
        # and disable visualization.
        jsma_tutorial_args = {'train_start': 0,
                              'train_end': 10000,
                              'test_start': 0,
                              'test_end': 1666,
                              'viz_enabled': False,
                              'source_samples': 1,
                              'nb_epochs': 2}
        report = mnist_tutorial_jsma.mnist_tutorial_jsma(**jsma_tutorial_args)

        # Check accuracy values contained in the AccuracyReport object
        self.assertTrue(report.clean_train_clean_eval > 0.75)
        self.assertTrue(report.clean_train_adv_eval < 0.05)

        # There is no adversarial training in the JSMA tutorial
        self.assertTrue(report.adv_train_clean_eval == 0.)
        self.assertTrue(report.adv_train_adv_eval == 0.)


if __name__ == '__main__':
    unittest.main()
