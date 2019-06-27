import jax.numpy as np
from jax import grad, vmap
from jax.experimental.stax import logsoftmax

from cleverhans.future.jax.utils import one_hot


def fast_gradient_method(model_fn, x, eps, ord, clip_min=None, clip_max=None, y=None,
	targeted=False):
  """
  JAX implementation of the Fast Gradient Method.
  :param model_fn: a callable that takes an input tensor and returns the model logits.
  :param x: input tensor.
  :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
  :param ord: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
  :param clip_min: (optional) float. Minimum float value for adversarial example components.
  :param clip_max: (optional) float. Maximum float value for adversarial example components.
  :param y: (optional) Tensor with true labels. If targeted is true, then provide the
            target label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting adversarial samples. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect (explained in this paper:
            https://arxiv.org/abs/1611.01236). Default is None.
  :param targeted: (optional) bool. Is the attack targeted or untargeted?
            Untargeted, the default, will try to make the label incorrect.
            Targeted will instead try to move in the direction of being more like y.
  :return: a tensor for the adversarial example
  """
  if y is None:
    # Using model predictions as ground truth to avoid label leaking
    x_labels = np.argmax(model_fn(x), 1)
    y = one_hot(x_labels, 10)

  def loss_adv(image, label):
    pred = model_fn(np.expand_dims(image, axis=0)) #predict(params, image)
    loss = - np.sum(logsoftmax(pred) * label)
    if targeted:
    	loss = -loss
    return loss

  grads_fn = vmap(grad(loss_adv), in_axes=(0, 0), out_axes=0)
  grads = grads_fn(x, y)

  if ord == np.inf:
    perturbations = eps * np.sign(grads)
  else:
    raise NotImplementedError("Only L-inf are currently implemented.")

  adv_x = x + perturbations

  # If clipping is needed, reset all values outside of [clip_min, clip_max]
  if (clip_min is not None) or (clip_max is not None):
    # We don't currently support one-sided clipping
    assert clip_min is not None and clip_max is not None
    adv_x = np.clip(adv_x, a_min=clip_min, a_max=clip_max)

  return adv_x
