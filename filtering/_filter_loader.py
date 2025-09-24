from filtering.dose_model._dose_filter import dose_filter_f
from filtering.gaussian_filter._gaussian_filter import gaussian_filter_jax_f


def filter_loader(filter: str, *args):
    """
    Loads the respective filter as a function.
    :param filter: Selects which filter function will be returned. The filter function should take the density
                   and additional parameters as an input.
                   Modes:
                    "None": Returns unity.
                    "dose_conv": Returns the dosage accumulation simulation by 3D convolution.
                    "gauss_jax": Returns a Gaussian filter which uses Jax.
    :return: The filter function with rho -> f(rho)
    """

    if filter == "None":
        return lambda x: x
    if filter == "dose_conv":
        resolution = args[0]
        return dose_filter_f(resolution)
    if filter == "gauss_jax":
        sigma = args[0]
        return gaussian_filter_jax_f(sigma)
