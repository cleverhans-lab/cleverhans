def ordered_union(l1, l2):
    out = []
    for e in l1 + l2:
        if e not in out:
            out.append(e)
    return out
