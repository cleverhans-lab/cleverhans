import unittest


class TestMNISTTutorialCW(unittest.TestCase):
    def test_mnist_tutorial_cw(self):
        from cleverhans_tutorials import mnist_tutorial_cw

        # Run the MNIST tutorial on a dataset of reduced size
        # and disable visualization.
        cw_tutorial_args = {'train_start': 0,
                            'train_end': 10000,
                            'test_start': 0,
                            'test_end': 1666,
                            'viz_enabled': False}
        report = mnist_tutorial_cw.mnist_tutorial_cw(**cw_tutorial_args)

        # Check accuracy values contained in the AccuracyReport object
        self.assertTrue(report.clean_train_clean_eval > 0.85)
        self.assertTrue(report.clean_train_adv_eval == 0.00)

        # There is no adversarial training in the CW tutorial
        self.assertTrue(report.adv_train_clean_eval == 0.)
        self.assertTrue(report.adv_train_adv_eval == 0.)


if __name__ == '__main__':
    unittest.main()
