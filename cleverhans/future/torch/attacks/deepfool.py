"""The Deepfool attack."""
import numpy as np
import torch
from torch.autograd.gradcheck import zero_gradients
from cleverhans.future.torch.utils import clip_eta


def deepfool(model_fn, x, clip_min=-np.inf, clip_max=np.inf,
             y=None, targeted=False, eps=None, norm=None,
             num_classes=10, overshoot=0.02, max_iter=50,
             is_debug=False, sanity_checks=False):
  """
  PyTorch implementation of DeepFool (https://arxiv.org/pdf/1511.04599.pdf).
  :param model_fn: A callable that takes an input tensor and returns the model logits.
  :param x: Input tensor.
  :param clip_min: If specified, the minimum input value.
  :param clip_max: If specified, the maximum input value.
  :param y: (optional) Tensor with true labels. If targeted is true, then provide the
            target label. Otherwise, only provide this parameter if you'd like to use true
            labels when crafting adversarial samples. Otherwise, model predictions are used
            as labels to avoid the "label leaking" effect (explained in this paper:
            https://arxiv.org/abs/1611.01236). Default is None.
  :param targeted: (optional) bool. Is the attack targeted or untargeted? Untargeted, the
            default, will try to make the label incorrect. Targeted will instead try to
            move in the direction of being more like y.
  :param eps: The size of the maximum perturbation, or None if the perturbation
            should not be constrained.
  :param norm: Order of the norm used for eps (mimics NumPy). Possible values: np.inf, 1 or 2.
  :param num_classes: the attack targets this many of the closest classes in the untargeted
            version.
  :param overshoot: used as a termination criterion to prevent vanishing updates.
  :param max_iter: maximum number of iterations for DeepFool.
  :param is_debug: If True, print the success rate after each iteration.
  :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
            memory or for unit tests that intentionally pass strange input)
  :return: a tensor for the adversarial example
  """

  if y is not None and len(x) != len(y):
    raise ValueError('number of inputs {} is different from number of labels {}'
                     .format(len(x), len(y)))
  if y is None:
    if targeted:
      raise ValueError('cannot perform a targeted attack without specifying targets y')
    y = torch.argmax(model_fn(x), dim=1)

  if eps is not None:
    if eps < 0:
      raise ValueError(
          "eps must be greater than or equal to 0, got {} instead".format(eps))
    if norm not in [np.inf, 1, 2]:
      raise ValueError('invalid norm')
    if eps == 0:
      return x

  if clip_min is not None and clip_max is not None:
    if clip_min > clip_max:
      raise ValueError(
          "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}"
          .format(clip_min, clip_max))

  asserts = []

  # If a data range was specified, check that the input was in that range
  
  asserts.append(torch.all(x >= clip_min))
  asserts.append(torch.all(x <= clip_max))

  # Determine classes to target
  if targeted:
    target_classes = y[:, None]
    y = torch.argmax(model_fn(x), dim=1)
  else:
    logits = model_fn(x)
    logit_indices = torch.arange(
      logits.size()[1],
      dtype=y.dtype,
      device=y.device,
    )[None, :].expand(y.size()[0], -1)
    # Number of target classes should be at most number of classes minus 1
    num_classes = min(num_classes, logits.size()[1] - 1)
    incorrect_logits = torch.where(
      logit_indices == y[:, None],
      torch.full_like(logits, -np.inf),
      logits,
    )
    target_classes = incorrect_logits.argsort(
      dim=1, descending=True)[:, :num_classes]

  x = x.clone().detach().to(torch.float)
  perturbations = torch.zeros_like(x)

  if is_debug:
    print("Starting DeepFool attack")

  for i in range(max_iter):
    x_adv = x + (1 + overshoot) * perturbations
    x_adv.requires_grad_(True)
    zero_gradients(x_adv)
    logits = model_fn(x_adv)

    # "Live" inputs are still being attacked; others have already achieved misclassification
    if targeted:
      live = torch.argmax(logits, dim=1) != target_classes[:, 0]
    else:
      live = torch.argmax(logits, dim=1) == y
    if is_debug:
      print('Iteration {}: {:.1f}% success'.format(
        i, 100 * (1 - live.sum().float() / len(live)).item()))
    if torch.all(~live):
      # Stop early if all inputs are already misclassified
      break

    smallest_magnitudes = torch.full((int(live.sum()),), np.inf,
                                     dtype=torch.float, device=perturbations.device)
    smallest_perturbation_updates = torch.zeros_like(perturbations[live])

    logits[live, y[live]].sum().backward(retain_graph=True)
    grads_correct = x_adv.grad.data[live].clone().detach()

    for k in range(target_classes.size()[1]):
      zero_gradients(x_adv)

      logits_target = logits[live, target_classes[live, k]]
      logits_target.sum().backward(retain_graph=True)
      grads_target = x_adv.grad.data[live].clone().detach()

      grads_diff = (grads_target - grads_correct).detach()
      logits_margin = (logits_target - logits[live, y[live]]).detach()

      grads_norm = grads_diff.norm(p=1 if norm == np.inf else 2,
                                   dim=list(range(1, len(grads_diff.size()))))
      magnitudes = logits_margin.abs() / grads_norm

      magnitudes_expanded = magnitudes
      for _ in range(len(grads_diff.size()) - 1):
        grads_norm = grads_norm.unsqueeze(-1)
        magnitudes_expanded = magnitudes_expanded.unsqueeze(-1)

      if norm == np.inf:
        perturbation_updates = ((magnitudes_expanded + 1e-4) *
                                torch.sign(grads_diff))
      else:
        perturbation_updates = ((magnitudes_expanded + 1e-4) * grads_diff /
                                 grads_norm)

      smaller = magnitudes < smallest_magnitudes
      smallest_perturbation_updates[smaller] = perturbation_updates[smaller]
      smallest_magnitudes[smaller] = magnitudes[smaller]

    all_perturbation_updates = torch.zeros_like(perturbations)
    all_perturbation_updates[live] = smallest_perturbation_updates
    perturbations.add_(all_perturbation_updates)

    perturbations *= (1 + overshoot)
    if eps is not None:
      perturbations = clip_eta(perturbations, norm, eps)
    perturbations = torch.clamp(x + perturbations, clip_min, clip_max) - x
    perturbations /= (1 + overshoot)

#   perturbations *= (1 + overshoot)
#   if eps is not None:
#     perturbations = clip_eta(perturbations, norm, eps)
# 
  x_adv = x + perturbations * (1 + overshoot)

  asserts.append(torch.all(x_adv >= clip_min))
  asserts.append(torch.all(x_adv <= clip_max))

  if sanity_checks:
    assert np.all(asserts)

  return x_adv
