import unittest


class TestMNISTTutorialCW(unittest.TestCase):
    def test_mnist_tutorial_cw(self):
        from tutorials import mnist_tutorial_cw

        # Run the MNIST tutorial on a dataset of reduced size
        # and disable visualization.
        cw_tutorial_args = {'train_start': 0,
                            'train_end': 100,
                            'test_start': 0,
                            'test_end': 166,
                            'viz_enabled': False,
                            'attack_iterations': 10}
        report = mnist_tutorial_cw.mnist_tutorial_cw(**cw_tutorial_args)

        cw_tutorial_args['targeted'] = False
        report = mnist_tutorial_cw.mnist_tutorial_cw(**cw_tutorial_args)

        # just verify that it doesn't crash; the results are being tested
        # in test_attacks.py


if __name__ == '__main__':
    unittest.main()
