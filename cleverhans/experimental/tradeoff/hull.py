
def make_hull(points):
  # Don't edit the client's memory
  points = list(points)

  def key(elem):
    adv_acc, clean_acc = elem
    return adv_acc

  points = sorted(points, key=key)

  # Increase clean_acc to respect implied attainability
  for i in range(len(points)):
    points[i] = (points[i][0], max(point[1] for point in points[i:]))

  # Remove ties
  i = 0
  while i < len(points):
    if i + 1 < len(points):
      if points[i][0] == points[i + 1][0]:
        if points[i][1] > points[i + 1][1]:
          del points[i + 1]
          i += 1
        else:
          del points[i]
      elif points[i][1] == points[i + 1][1]:
        del points[i]
      else:
        i += 1
    else:
      i += 1

  # Since this is only going to run on a small number of points, use the
  # super naive algorithm / prioritize readability over performance.
  i = 0
  while i < len(points):
    j = i + 1
    while j < len(points):
      removed = False
      for k in range(j + 1, len(points)):
        # Read out adv_acc
        left_adv_acc = points[i][0]
        mid_adv_acc = points[j][0]
        right_adv_acc = points[k][0]
        # Compute convex combination coefficient
        coeff = (mid_adv_acc - left_adv_acc) / (right_adv_acc - left_adv_acc)
        # Read out clean acc
        left_clean_acc = points[i][1]
        mid_clean_acc = points[j][1]
        right_clean_acc = points[k][1]
        # Clean acc requirement
        clean_acc_requirement = coeff * right_clean_acc + (1. - coeff) * left_clean_acc
        # Use <= so that redundant points are removed too, it is not enough to
        # be on the surface, the point must *define* the surface
        if mid_clean_acc <= clean_acc_requirement:
          del points[j]
          removed = True
          break
      if not removed:
        j += 1
    i += 1

  return points

def area_below(points):

    # Don't edit the client's memory
  points = list(points)

  # Zero robustness is always attainable
  # Note that this preprocessing step shouldn't go in the `make_hull` function
  # because it violates the "strictly decreasing" property
  if points[0][0] != 0:
    points = [(0., points[0][1])] + points

  area = 0.

  for i in range(1, len(points)):
    w = points[i][0] - points[i - 1][0]
    assert w > 0.
    hl = points[i - 1][1]
    hr = points[i][1]
    if i == 1:
      assert hr <= hl
    else:
      assert hr < hl, (i, hl, hr, points)
    # Rectangle
    area += w * hr
    # Triangle
    area += 0.5 * w * (hl - hr)

  return area
