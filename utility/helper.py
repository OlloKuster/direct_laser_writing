def f2param(x, lims):
    """
    Scales x from [0, 1] to [lims[0], lims[1]].
    :param x: Density distribution to be scaled.
    :param lims: Limits of the scaled distribution.
    :return: Rescaled array ranging from lims[0] to lims[1].
    """
    (a, b) = lims
    return (b - a) * x + a

def split_int(a):
    """
    Splits a into two parts as close as possible to the middle.
    :param a: Value to be split in the middle.
    :return: Tuple of both halves of a.
    """
    return a // 2, a // 2 + a % 2
