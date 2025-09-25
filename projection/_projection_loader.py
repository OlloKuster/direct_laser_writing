from projection.tanh.subpixel_smoothed_projection import ssp_proj_jax_f
from projection.SSP.tanh_projection import tanh_filter_jax_f


def projection_loader(projection: str, *args):
    """
    Loads the respective filter as a function.
    :param projection: Selects which projection function will be returned. The projection function should take the
                       density and additional parameters (alpha and beta) as an input.
                       Modes:
                        "None": Returns unity.
                        "tanh_jax": Returns the tanh-projection which uses Jax.
                        "ssp_jax": Returns the SSP-projection which uses Jax [https://doi.org/10.1364/OE.563512].
    :return: The projection function with rho -> p(rho)
    """
    if projection == "None":
        return lambda x: x
    if projection == "tanh_jax":
        return tanh_filter_jax_f(alpha, beta)
    if projection == "ssp_jax":
        alpha, beta, resolution = args
        return ssp_proj_jax_f(alpha, beta, resolution)
