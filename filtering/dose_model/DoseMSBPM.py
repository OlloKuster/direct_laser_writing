import numpy as np
import jax.numpy as jnp
from jax.lax import conv_general_dilated


def dose_msbpm_jax_f(rho_0_GT, psf, print_params, nonlinearity):
    """
    Creates the dlw dose model.
    :param rho_0_GT: Saturation density of the resist.
    :param psf: Point Spread Function intensity profile [kx, ky, kz].
    :param print_params: (sig_2_r, time_exposure, intensity_without_power, correction_factor) List the parameters
                         relevant to the print.
    :param nonlinearity: Power exponent of the nonlinearity.
    :return: (accumulated_dose, dose_to_rho) Convolution of the density with the PSF^n and the conversion of the
             convolved dose into a density for a given laser power.
    """
    sig_2_r, time_exposure, intensity_without_power, correction_factor = print_params
    kernel = jnp.asarray(np.asarray(psf, dtype=np.float64) ** nonlinearity)[None, None]  # [O, I, kx, ky, kz]
    # Computed on the host in float64, intensity_without_power ** nonlinearity exceeds the float32 range
    prefactor = float(np.float64(sig_2_r) * time_exposure * np.float64(intensity_without_power) ** nonlinearity
                      * correction_factor)

    def accumulated_dose(obj):
        """
        Convolution of the density with the PSF^n (cross-correlation with 'same' padding, like torch conv3d).
        :param obj: Input density [nx, ny, nz].
        :return: Convolved dose [nx, ny, nz].
        """
        obj = jnp.asarray(obj)
        return conv_general_dilated(obj[None, None], kernel.astype(obj.dtype),
                                    window_strides=(1, 1, 1), padding='SAME',
                                    dimension_numbers=('NCHWD', 'OIHWD', 'NCHWD'))[0, 0]

    def dose_to_rho(conv, lp):
        """
        Converts the convolved dose into the accumulated power in the resist as a density.
        :param conv: Output of accumulated_dose.
        :param lp: Laser Power.
        :return: Accumulated power in the resist as a density.
        """
        return rho_0_GT * (1. - jnp.exp(-prefactor * conv * lp ** nonlinearity))

    return accumulated_dose, dose_to_rho
