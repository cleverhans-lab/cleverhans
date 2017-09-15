import cleverhans.model
import tensorflow as tf
import numpy as np


def cleverhans_attack_wrapper(cleverhans_attack_fn, reset=True):
    def attack(a):
        session = tf.Session()
        with session.as_default():
            model = RVBCleverhansModel(a)
            adversarial_image = cleverhans_attack_fn(model, session, a)
            adversarial_image = np.squeeze(adversarial_image, axis=0)
            if reset:
                # optionally, reset to ignore other adversarials
                # found during the search
                a._reset()
            # run predictions to make sure the returned adversarial
            # is taken into account
            min_, max_ = a.bounds()
            adversarial_image = np.clip(adversarial_image, min_, max_)
            a.predictions(adversarial_image)
    return attack


def py_func_grad(func, inp, Tout, stateful=True, name=None, grad=None):
    """Custom py_func with gradient support

    """
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({
            "PyFunc": rnd_name,
            "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


class RVBCleverhansModel(cleverhans.model.Model):
    """This is a cleverhans model that wraps a robust vision benchmark model.

    """

    def __init__(self, adversarial):
        self.adversarial = adversarial

    def get_layer_names(self):
        return ['logits']

    def fprop(self, x):
        return {'logits': self._logits_op(x)}

    def _logits_op(self, x, name=None):
        with tf.name_scope(name, "logits", [x]) as name:

            num_classes = self.adversarial.num_classes()

            def _backward_py(gradient_y, x):
                x = np.squeeze(x, axis=0)
                gradient_y = np.squeeze(gradient_y, axis=0)
                gradient_x = self.adversarial.backward(gradient_y, x)
                gradient_x = gradient_x.astype(np.float32)
                return gradient_x[np.newaxis]

            def _backward_tf(op, grad):
                images = op.inputs[0]
                gradient_x = tf.py_func(
                    _backward_py, [grad, images], tf.float32)
                gradient_x.set_shape(images.shape)
                return gradient_x

            def _forward_py(x):
                predictions = self.adversarial.batch_predictions(
                    x, strict=False)[0]
                predictions = predictions.astype(np.float32)
                return predictions

            op = py_func_grad(
                _forward_py,
                [x],
                [tf.float32],
                name=name,
                grad=_backward_tf)

            logits = op[0]
            logits.set_shape((x.shape[0], num_classes))

        return logits
