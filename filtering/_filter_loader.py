from filtering.dose_model._dose_filter import dose_filter_f, dose_filter_robust_f
from filtering.gaussian_filter._gaussian_filter import gaussian_filter_jax_f, conic_filter_jax_f


def filter_loader(filter: str, *args):
    """
    Loads the respective filter as a function.
    :param filter: Selects which filter function will be returned. The filter function should take the density
                   and additional parameters as an input.
                   Modes:
                    "None": Returns unity.
                    "dose_conv": Returns the dosage accumulation simulation by 3D convolution.
                                 args: (resolution, lp, psf), psf e.g. from utils_dose_sim.calc_psf_GT.
                    "dose_conv_robust": Same as "dose_conv" for an eroded, normal and dilated laser power.
                                        args: (resolution, deviation, psf).
                    "gauss_jax": Returns a Gaussian filter which uses Jax.
                    "conic_jax": Returns a cone filter which uses Jax.
    :return: The filter function with rho -> f(rho)
    """

    if filter == "None":
        return lambda x: x
    if filter == "dose_conv":
        resolution, lp, psf = args
        return dose_filter_f(resolution, psf, lp)
    if filter == "dose_conv_robust":
        resolution, deviation, psf = args
        return dose_filter_robust_f(resolution, psf, deviation)
    if filter == "gauss_jax":
        sigma = args[0]
        return gaussian_filter_jax_f(sigma)
    if filter == "conic_jax":
        radius = args[0]
        return conic_filter_jax_f(radius)
