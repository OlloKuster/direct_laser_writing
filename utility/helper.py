import jax.numpy as jnp
import jax
import torch
import numpy as np


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


# def softplus(x, beta=50):
#     """
#     Softplus function to balance out the optimisation.
#     :param x: Array of figure of merits which are evaluated.
#     :param beta: Steepness of the curve.
#     :return: Array of softplus of all figure of merits.
#     """
#     mask = x * beta > 20
#     return jnp.where(mask, x, 1 / beta * jnp.log(1 + jnp.exp(jnp.where(mask, 0, x * beta))))


def relu(x):
    return jax.nn.relu6(x)


def convert_to(x, package):
    """
    Conversions of variables during steps (e.g. torch tensor to numpy array).
    :param x: Input.
    :param package: Chooses which format x will be converted to.
    :return:
    """
    if package == "torch":
        return torch.tensor(np.array(x), device='cuda')
    if package == "torch2np":
        return x.detach().cpu().numpy()
    else:
        return x


def draw_circle(rho, x_cen, y_cen, radius):
    x = np.linspace(0, rho.shape[0], rho.shape[0])
    y = np.linspace(0, rho.shape[1], rho.shape[0])
    xx, yy = np.meshgrid(x, y)
    for x_c, y_c in zip(x_cen, y_cen):
        rho = rho + np.round(np.exp(-((xx-x_c)**2 + (yy-y_c)**2) / (2 * radius)))
    return np.exp(-((xx-x_cen)**2 + (yy-y_cen)**2) / (2 * radius))