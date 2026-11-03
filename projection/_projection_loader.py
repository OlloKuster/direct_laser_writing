from projection.SSP.subpixel_smoothed_projection import ssp_proj_jax_f, ssp_robust_proj_jax_f
from projection.lattice.lattice_projection import lattice_proj_jax_f
from projection.tanh.tanh_projection import tanh_filter_jax_f


def projection_loader(projection: str, *args):
    """
    Loads the respective filter as a function.
    :param projection: Selects which projection function will be returned. The projection function should take the
                       density and additional parameters (alpha and beta) as an input.
                       Modes:
                        "None": Returns unity.
                        "tanh_jax": Returns the tanh-projection which uses Jax.
                        "ssp_jax": Returns the SSP-projection which uses Jax [https://doi.org/10.1364/OE.563512].
                        "robust_ssp_jax": Returns the SSP-projection for multiple threshold values.
                        "lattice_tanh_jax": Initial projection onto the writing lattice (ConfigPrint.hatching,
                                            ConfigPrint.slicing) with a tanh binarization.
                        "lattice_ssp_jax": Same as "lattice_tanh_jax" with an SSP binarization.
    :return: The projection function with rho -> p(rho)
    """
    if projection == "None":
        alpha, beta, _ = args
        return lambda x: x
    if projection == "tanh_jax":
        alpha, beta, _ = args
        return tanh_filter_jax_f(alpha, beta)
    if projection == "ssp_jax":
        alpha, beta, resolution = args
        return ssp_proj_jax_f(alpha, beta, resolution)
    if projection == "robust_ssp_jax":
        alpha, beta, resolution = args
        return ssp_robust_proj_jax_f(alpha, beta, resolution)
    if projection in ("lattice_tanh_jax", "lattice_ssp_jax"):
        alpha, beta, resolution = args
        if projection == "lattice_tanh_jax":
            f2bin = tanh_filter_jax_f(alpha, beta)
        else:
            f2bin = ssp_proj_jax_f(alpha, beta, (resolution, resolution, resolution))
        return lattice_proj_jax_f(f2bin, resolution)
