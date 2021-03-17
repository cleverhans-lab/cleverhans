"""
Boundary Attack++
"""
import numpy as np
import torch


def hop_skip_jump_attack(
    model_fn,
    x,
    norm,
    y_target=None,
    image_target=None,
    initial_num_evals=100,
    max_num_evals=10000,
    stepsize_search="geometric_progression",
    num_iterations=64,
    gamma=1.0,
    constraint=2,
    batch_size=128,
    verbose=True,
    clip_min=0,
    clip_max=1,
):
    """
    PyTorch implementation of HopSkipJumpAttack.
    HopSkipJumpAttack was originally proposed by Chen, Jordan and Wainwright.
    It is a decision-based attack that requires access to output
    labels of a model alone.
    Paper link: https://arxiv.org/abs/1904.02144
    At a high level, this attack is an iterative attack composed of three
    steps: Binary search to approach the boundary; gradient estimation;
    stepsize search. HopSkipJumpAttack requires fewer model queries than
    Boundary Attack which was based on rejective sampling.

    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor with n samples.
    :param norm: The distance to optimize. Possible values: 2 or np.inf.
    :param y_target:  A tensor of shape (n, nb_classes) for target labels.
    Required for targeted attack.
    :param image_target: A tensor of shape (n, **image shape) for initial
    target images. Required for targeted attack.
    :param initial_num_evals: initial number of evaluations for
                              gradient estimation.
    :param max_num_evals: maximum number of evaluations for gradient estimation.
    :param stepsize_search: How to search for stepsize; choices are
                            'geometric_progression', 'grid_search'.
                            'geometric progression' initializes the stepsize
                             by ||x_t - x||_p / sqrt(iteration), and keep
                             decreasing by half until reaching the target
                             side of the boundary. 'grid_search' chooses the
                             optimal epsilon over a grid, in the scale of
                             ||x_t - x||_p.
    :param num_iterations: The number of iterations.
    :param gamma: The binary search threshold theta is gamma / d^{3/2} for
                   l2 attack and gamma / d^2 for linf attack.
    :param batch_size: batch_size for model prediction.
    :param verbose: (boolean) Whether distance at each step is printed.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    """
    shape = (1,) + x.shape[1:]
    if y_target is not None:
        assert image_target is not None, "Require a target image for targeted attack."
    if clip_min is not None and clip_max is not None:
        if clip_min > clip_max:
            raise ValueError(
                "clip_min must be less than or equal to clip_max, got clip_min={} and clip_max={}".format(
                    clip_min, clip_max
                )
            )

    d = int(np.prod(shape))

    if constraint == 2:
        theta = gamma / (np.sqrt(d) * d)
    else:
        theta = gamma / (d * d)

    def hsja(sample, target_label, target_image):
        if target_label is None:
            _, original_label = torch.max(model_fn(sample), 1)

        def decision_function(images):
            """
            Decision function output 1 on the desired side of the boundary,
            0 otherwise.
            """
            images = torch.clamp(images, clip_min, clip_max)
            prob = []
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                prob_i = model_fn(batch)
                prob.append(prob_i)
            prob = torch.cat(prob, dim=0)
            if target_label is None:
                return torch.max(prob, dim=1)[1] != original_label
            else:
                return torch.max(prob, dim=1)[1] == target_label

        # Initialize.
        if target_image is None:
            perturbed = initialize(decision_function, sample, shape, clip_min, clip_max)
        else:
            perturbed = target_image.to(sample.device)

        # Project the initialization to the boundary.
        perturbed, dist_post_update = binary_search_batch(
            sample, perturbed, decision_function, shape, constraint, theta
        )
        dist = compute_distance(perturbed, sample, constraint)

        for j in np.arange(num_iterations):
            current_iteration = j + 1

            # Choose delta.
            delta = select_delta(
                dist_post_update,
                current_iteration,
                clip_max,
                clip_min,
                d,
                theta,
                constraint,
            )

            # Choose number of evaluations.
            num_evals = int(min([initial_num_evals * np.sqrt(j + 1), max_num_evals]))

            # approximate gradient.
            gradf = approximate_gradient(
                decision_function,
                perturbed,
                num_evals,
                delta,
                constraint,
                shape[1:],
                clip_min,
                clip_max,
            )
            if constraint == np.inf:
                update = torch.sign(gradf)
            else:
                update = gradf

            # search step size.
            if stepsize_search == "geometric_progression":
                # find step size.
                epsilon = geometric_progression_for_stepsize(
                    perturbed, update, dist, decision_function, current_iteration
                )

                # Update the sample.
                perturbed = torch.clamp(
                    perturbed + epsilon * update, clip_min, clip_max
                )

                # Binary search to return to the boundary.
                perturbed, dist_post_update = binary_search_batch(
                    sample, perturbed, decision_function, shape, constraint, theta
                )

            elif stepsize_search == "grid_search":
                # Grid search for stepsize.
                epsilons = (
                    torch.from_numpy(np.logspace(-4, 0, num=20, endpoint=True))
                    .to(perturbed.device)
                    .float()
                    * dist
                )
                perturbeds = (
                    perturbed + epsilons.view((20,) + (1,) * (len(shape) - 1)) * update
                )
                perturbeds = torch.clamp(perturbeds, clip_min, clip_max)
                idx_perturbed = decision_function(perturbeds)

                if torch.sum(idx_perturbed) > 0:
                    # Select the perturbation that yields the minimum distance # after binary search.
                    perturbed, dist_post_update = binary_search_batch(
                        sample,
                        perturbeds[idx_perturbed],
                        decision_function,
                        shape,
                        constraint,
                        theta,
                    )

            # compute new distance.
            dist = compute_distance(perturbed, sample, constraint)
            if verbose:
                print(
                    "iteration: {:d}, {:s} distance {:.4E}".format(
                        j + 1, str(constraint), dist
                    )
                )

        return perturbed

    # Perform attack on one instance at a time
    adv_x = []
    for i, x_ in enumerate(x):
        x_ = x_.unsqueeze(0)
        if y_target is not None:
            # targeted attack that requires target label and image.
            pert = hsja(x_, y_target[i], image_target[i])
        else:
            if image_target is not None:
                pert = hsja(x_, None, image_target[i])
            else:
                # untargeted attack without an initialized image.
                pert = hsja(x_, None, None)
        adv_x.append(pert)
    return torch.cat(adv_x, 0)


def compute_distance(x_ori, x_pert, constraint=2):
    """ Compute the distance between two images. """
    if constraint == 2:
        dist = torch.norm(x_ori - x_pert, p=2)
    elif constraint == np.inf:
        dist = torch.max(torch.abs(x_ori - x_pert))
    return dist


def approximate_gradient(
    decision_function, sample, num_evals, delta, constraint, shape, clip_min, clip_max
):
    """ Gradient direction estimation """
    # Generate random vectors.
    noise_shape = [num_evals] + list(shape)
    if constraint == 2:
        rv = torch.randn(noise_shape)
    elif constraint == np.inf:
        rv = -1 + torch.rand(noise_shape) * 2

    axis = tuple(range(1, 1 + len(shape)))
    rv = rv / torch.sqrt(torch.sum(rv ** 2, dim=axis, keepdim=True))
    perturbed = sample + delta * rv.to(sample.device)
    perturbed = torch.clamp(perturbed, clip_min, clip_max)
    rv = (perturbed - sample) / delta

    # query the model.
    decisions = decision_function(perturbed).float()
    fval = 2.0 * decisions.view((decisions.shape[0],) + (1,) * len(shape)) - 1.0

    # Baseline subtraction (when fval differs)
    fval_mean = torch.mean(fval)
    if fval_mean == 1.0:  # label changes.
        gradf = torch.mean(rv, dim=0)
    elif fval_mean == -1.0:  # label not change.
        gradf = -torch.mean(rv, dim=0)
    else:
        fval = fval - fval_mean
        gradf = torch.mean(fval * rv, dim=0)

    # Get the gradient direction.
    gradf = gradf / torch.norm(gradf, p=2)

    return gradf


def project(original_image, perturbed_images, alphas, shape, constraint):
    """ Projection onto given l2 / linf balls in a batch. """
    alphas = alphas.view((alphas.shape[0],) + (1,) * (len(shape) - 1))
    if constraint == 2:
        projected = (1 - alphas) * original_image + alphas * perturbed_images
    elif constraint == np.inf:
        projected = torch.clamp(
            perturbed_images, original_image - alphas, original_image + alphas
        )
    return projected


def binary_search_batch(
    original_image, perturbed_images, decision_function, shape, constraint, theta
):
    """ Binary search to approach the boundary. """

    # Compute distance between each of perturbed image and original image.
    dists_post_update = torch.stack(
        [
            compute_distance(original_image, perturbed_image, constraint)
            for perturbed_image in perturbed_images
        ]
    )

    # Choose upper thresholds in binary searchs based on constraint.
    if constraint == np.inf:
        highs = dists_post_update
        # Stopping criteria.
        thresholds = torch.min(dists_post_update * theta, theta)
    else:
        highs = torch.ones(len(perturbed_images)).to(original_image.device)
        thresholds = theta

    lows = torch.zeros(len(perturbed_images)).to(original_image.device)

    while torch.max((highs - lows) / thresholds) > 1:
        # projection to mids.
        mids = (highs + lows) / 2.0
        mid_images = project(original_image, perturbed_images, mids, shape, constraint)

        # Update highs and lows based on model decisions.
        decisions = decision_function(mid_images)
        lows = torch.where(decisions == 0, mids, lows)
        highs = torch.where(decisions == 1, mids, highs)

    out_images = project(original_image, perturbed_images, highs, shape, constraint)

    # Compute distance of the output image to select the best choice.
    # (only used when stepsize_search is grid_search.)
    dists = torch.stack(
        [
            compute_distance(original_image, out_image, constraint)
            for out_image in out_images
        ]
    )
    _, idx = torch.min(dists, 0)

    dist = dists_post_update[idx]
    out_image = out_images[idx].unsqueeze(0)
    return out_image, dist


def initialize(decision_function, sample, shape, clip_min, clip_max):
    """
    Efficient Implementation of BlendedUniformNoiseAttack in Foolbox.
    """
    success = 0
    num_evals = 0

    # Find a misclassified random noise.
    while True:
        random_noise = clip_min + torch.rand(shape).to(sample.device) * (
            clip_max - clip_min
        )
        success = decision_function(random_noise)[0]
        if success:
            break
        num_evals += 1
        message = (
            "Initialization failed! Try to use a misclassified image as `target_image`"
        )
        assert num_evals < 1e4, message

    # Binary search to minimize l2 distance to original image.
    low = 0.0
    high = 1.0
    while high - low > 0.001:
        mid = (high + low) / 2.0
        blended = (1 - mid) * sample + mid * random_noise
        success = decision_function(blended)[0]
        if success:
            high = mid
        else:
            low = mid

    initialization = (1 - high) * sample + high * random_noise
    return initialization


def geometric_progression_for_stepsize(
    x, update, dist, decision_function, current_iteration
):
    """Geometric progression to search for stepsize.
    Keep decreasing stepsize by half until reaching
    the desired side of the boundary.
    """
    epsilon = dist / np.sqrt(current_iteration)
    while True:
        updated = x + epsilon * update
        success = decision_function(updated)[0]
        if success:
            break
        else:
            epsilon = epsilon / 2.0

    return epsilon


def select_delta(
    dist_post_update, current_iteration, clip_max, clip_min, d, theta, constraint
):
    """
    Choose the delta at the scale of distance
     between x and perturbed sample.
    """
    if current_iteration == 1:
        delta = 0.1 * (clip_max - clip_min)
    else:
        if constraint == 2:
            delta = np.sqrt(d) * theta * dist_post_update
        elif constraint == np.inf:
            delta = d * theta * dist_post_update

    return delta
