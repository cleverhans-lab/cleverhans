import unittest


class TestMNISTBlackboxF(unittest.TestCase):
    def test_mnist_blackbox(self):
        from tutorials import mnist_blackbox

        # Run the MNIST tutorial on a dataset of reduced size, reduced number
        # of data augmentations, increased substitute holdout for faster runtime.
        mnist_blackbox_args = {'train_start': 0,
                               'train_end': 5000,
                               'test_start': 0,
                               'test_end': 2000,
                               'data_aug': 1,
                               'holdout': 1000,
                               'nb_epochs': 2,
                               'nb_epochs_s': 6}
        report = mnist_blackbox.mnist_blackbox(**mnist_blackbox_args)

        # Check accuracy values contained in the AccuracyReport object
        self.assertTrue(report['bbox'] > 0.7, report['bbox'])
        self.assertTrue(report['sub'] > 0.2, report['sub'])
        self.assertTrue(report['bbox_on_sub_adv_ex'] < 0.7, report['bbox_on_sub_adv_ex'])

if __name__ == '__main__':
    unittest.main()
