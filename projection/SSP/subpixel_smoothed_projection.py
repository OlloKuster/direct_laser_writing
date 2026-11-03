import jax.numpy as jnp
import numpy as np

from projection.tanh.tanh_projection import tanh_filter_jax_f


def f2bin_smooth(rho, alpha, resolution, f2bin):
    """
    Implements the subpixel-smoothed projection.
    https://doi.org/10.1364/OE.563512
    The smoothing kernel is an ellipsoid with the semi-axes 0.55 * (dx, dy, ...), i.e. 0.55 voxels along every axis.
    For a cubic grid this is the ball of radius 0.55 * dx of the paper. Since the radius scales with the voxel size,
    the result only depends on the differences between neighbouring voxels and not on the resolution itself, also
    for anisotropic voxels like the writing lattice.
    :param rho: Input density (design variable), any number of dimensions.
    :param alpha: Threshold value.
    :param resolution: Resolution of the problem [px/um], a scalar or one value per axis.
    :param f2bin: Binarization function.
    :return: Subpixel-smoothed density.
    """
    spacing = 1 / np.broadcast_to(np.asarray(resolution, dtype=float), (rho.ndim,))
    rho_proj = f2bin(rho)
    rho_grad = jnp.gradient(rho, *spacing)
    if rho.ndim == 1:
        rho_grad = [rho_grad]
    # Change of rho across the smoothing radius along the gradient: R * |grad rho| with R = 0.55 * |n * spacing|
    delta2 = sum((0.55 * g * dx) ** 2 for g, dx in zip(rho_grad, spacing))
    nonzero_norm = delta2 > 0
    delta = jnp.sqrt(jnp.where(nonzero_norm, delta2, 1))
    d_R = (alpha - rho) / delta
    needs_smoothing = nonzero_norm & (jnp.abs(d_R) < 1)
    F = jnp.where(needs_smoothing,
                  0.5 - 15 / 16 * d_R + 5 / 8 * d_R ** 3 - 3 / 16 * d_R ** 5,
                  1.0)
    F_minus = jnp.where(needs_smoothing,
                        0.5 + 15 / 16 * d_R - 5 / 8 * d_R ** 3 + 3 / 16 * d_R ** 5,
                        1.0)
    rho_minus = rho - delta * F
    rho_plus = rho + delta * F_minus
    rho_minus_eff_proj = f2bin(rho_minus)
    rho_plus_eff_proj = f2bin(rho_plus)
    rho_proj_smoothed = (1 - F) * rho_minus_eff_proj + F * rho_plus_eff_proj
    return jnp.where(needs_smoothing, rho_proj_smoothed, rho_proj)


def ssp_proj_jax_f(alpha, beta, resolution):
    """
    Generates the SSP Projection function.
    :param alpha: Threshold value.
    :param beta: Binarization level.
    :param resolution: Resolution of the problem.
    :return: SSP Projection function.
    """
    f2bin = tanh_filter_jax_f(alpha, beta)

    return lambda x: f2bin_smooth(x, alpha, resolution, f2bin)


def ssp_robust_proj_jax_f(alphas, beta, resolution):
    """
    Generates the SSP Projection function for a dlw optimization.
    :param alphas: Tuple of threshold values.
    :param beta: Binarization level.
    :param resolution: Resolution of the problem.
    :return: SSP Projection function.
    """
    f2bin_eroded = tanh_filter_jax_f(alphas[0], beta)
    f2bin = tanh_filter_jax_f(alphas[1], beta)
    f2bin_dilated = tanh_filter_jax_f(alphas[2], beta)

    return (lambda x: (f2bin_smooth(x[0], alphas[0], resolution, f2bin_eroded),
                       f2bin_smooth(x[1], alphas[1], resolution, f2bin),
                       f2bin_smooth(x[2], alphas[2], resolution, f2bin_dilated)))
