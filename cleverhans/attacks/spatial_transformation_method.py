"""The SpatialTransformationMethod attack
"""
import warnings

from cleverhans.attacks.attack import Attack


class SpatialTransformationMethod(Attack):
  """
  Spatial transformation attack
  """

  def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
    """
    Create a SpatialTransformationMethod instance.
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.

  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param kwargs: passed through to super constructor
    """

    super(SpatialTransformationMethod, self).__init__(
        model, sess, dtypestr, **kwargs)
    self.feedable_kwargs = ('n_samples', 'dx_min', 'dx_max', 'n_dxs', 'dy_min',
                            'dy_max', 'n_dys', 'angle_min', 'angle_max',
                            'n_angles', 'black_border_size')

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.
    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    from cleverhans.attacks_tf import spm

    labels, _ = self.get_or_guess_labels(x, kwargs)

    return spm(
        x,
        self.model,
        y=labels,
        n_samples=self.n_samples,
        dx_min=self.dx_min, dx_max=self.dx_max, n_dxs=self.n_dxs,
        dy_min=self.dy_min, dy_max=self.dy_max, n_dys=self.n_dys,
        angle_min=self.angle_min, angle_max=self.angle_max,
        n_angles=self.n_angles, black_border_size=self.black_border_size)

  def parse_params(self,
                   n_samples=None,
                   dx_min=-0.1,
                   dx_max=0.1,
                   n_dxs=2,
                   dy_min=-0.1,
                   dy_max=0.1,
                   n_dys=2,
                   angle_min=-30,
                   angle_max=30,
                   n_angles=6,
                   black_border_size=0,
                   **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.
    :param n_samples: (optional) The number of transformations sampled to
                      construct the attack. Set it to None to run
                      full grid attack.
    :param dx_min: (optional float) Minimum translation ratio along x-axis.
    :param dx_max: (optional float) Maximum translation ratio along x-axis.
    :param n_dxs: (optional int) Number of discretized translation ratios
                  along x-axis.
    :param dy_min: (optional float) Minimum translation ratio along y-axis.
    :param dy_max: (optional float) Maximum translation ratio along y-axis.
    :param n_dys: (optional int) Number of discretized translation ratios
                  along y-axis.
    :param angle_min: (optional float) Largest counter-clockwise rotation
                      angle.
    :param angle_max: (optional float) Largest clockwise rotation angle.
    :param n_angles: (optional int) Number of discretized angles.
    :param black_border_size: (optional int) size of the black border in pixels.
    """
    self.n_samples = n_samples
    self.dx_min = dx_min
    self.dx_max = dx_max
    self.n_dxs = n_dxs
    self.dy_min = dy_min
    self.dy_max = dy_max
    self.n_dys = n_dys
    self.angle_min = angle_min
    self.angle_max = angle_max
    self.n_angles = n_angles
    self.black_border_size = black_border_size

    if self.dx_min < -1 or self.dy_min < -1 or \
       self.dx_max > 1 or self.dy_max > 1:
      raise ValueError("The value of translation must be bounded "
                       "within [-1, 1]")
    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")
    return True
