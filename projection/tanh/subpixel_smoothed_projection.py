import jax.numpy as jnp

from projection.SSP.tanh_projection import tanh_filter_jax_f


def ssp_proj_jax_f(alpha, beta, resolution):
    f2bin = tanh_filter_jax_f(alpha, beta)

    def f2bin_smooth(rho):
        dx = dy = 1 / resolution
        R_smoothing = 0.55 * dx
        rho_proj = f2bin(rho)
        rho_grad = jnp.gradient(rho)
        rho_grad_norm2 = (rho_grad[0] / dx) ** 2 + (rho_grad[1] / dy) ** 2
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

    return f2bin_smooth
