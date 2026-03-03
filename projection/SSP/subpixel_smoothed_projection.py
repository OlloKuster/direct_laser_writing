import jax.numpy as jnp

from projection.tanh.tanh_projection import tanh_filter_jax_f


def f2bin_smooth(rho, alpha, resolution, f2bin):
    """
    Implements the subpixel-smoothed projection.
    https://doi.org/10.1364/OE.563512
    :param rho: Input density (design variable).
    :param alpha: Threshold value.
    :param resolution: Resolution of the problem [px/um].
    :param f2bin: Binarization function.
    :return: Subpixel-smoothed density.
    """
    dx = dy = dz = 1 / resolution
    R_smoothing = 0.55 * dx
    rho_proj = f2bin(rho)
    rho_grad = jnp.gradient(rho)
    rho_grad_norm2 = (rho_grad[0] / dx) ** 2 + (rho_grad[1] / dy) ** 2 + (rho_grad[2] / dz)**2
    nonzero_norm = jnp.abs(rho_grad_norm2) > 0
    rho_grad_norm = jnp.sqrt(jnp.where(nonzero_norm,
                                       rho_grad_norm2, 1))
    rho_grad_norm_eff = jnp.where(nonzero_norm, rho_grad_norm, 1)
    d = (alpha - rho) / rho_grad_norm_eff
    needs_smoothing = nonzero_norm & (jnp.abs(d) < R_smoothing)
    d_R = d / R_smoothing
    F = jnp.where(needs_smoothing,
                  0.5 - 15 / 16 * d_R + 5 / 8 * d_R ** 3 - 3 / 16 * d_R ** 5,
                  1.0)
    F_minus = jnp.where(needs_smoothing,
                        0.5 + 15 / 16 * d_R - 5 / 8 * d_R ** 3 + 3 / 16 * d_R ** 5,
                        1.0)
    rho_minus = rho - R_smoothing * rho_grad_norm_eff * F
    rho_plus = rho + R_smoothing * rho_grad_norm_eff * F_minus
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
    Generates the SSP Projection function for a robust optimization.
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
