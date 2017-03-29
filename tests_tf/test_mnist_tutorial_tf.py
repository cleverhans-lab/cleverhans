from inspect import getsourcefile
import os
import sys

# Import the tutorial after evaluating its relative path
current_dir = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
current_dir = current_dir[:current_dir.rfind(os.path.sep)]
tutorial_dir = os.path.join(current_dir, 'tutorials')
sys.path.insert(0, tutorial_dir)
import mnist_tutorial_tf
sys.path.pop(0)


if __name__ == '__main__':

    test_results = mnist_tutorial_tf.mnist_tutorial(testing=True)

    assert test_results[0] > 0.85
    assert test_results[1] < 0.05
    assert test_results[2] > 0.85
    assert test_results[3] > 0.25
