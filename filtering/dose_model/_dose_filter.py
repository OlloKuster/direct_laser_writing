import jax
import jax.numpy as jnp

from .config_print import ConfigPrint
from filtering.dose_model.DoseMSBPM import dose_msbpm_jax_f


def _dose_model(resolution, psf):
    """
    Sets up the dlw dose model for a given resolution and PSF.
    :param resolution: Resolution of the simulation [px/um].
    :param psf: Point Spread Function intensity profile [kx, ky, kz], e.g. from utils_dose_sim.calc_psf_GT.
    :return: (accumulated_dose, dose_to_rho) see dose_msbpm_jax_f.
    """
    res_lat = 1 / resolution * 10 ** (-6)  # hatching
    time_exposure = ConfigPrint.w0 / (ConfigPrint.vs / res_lat)

    print_params = [ConfigPrint.sig_2_r,
                    time_exposure,
                    ConfigPrint.intensity_without_power,
                    ConfigPrint.correction_factor]

    return dose_msbpm_jax_f(ConfigPrint.rho_0_GT, psf, print_params, ConfigPrint.nonlinearity)


def dose_filter_f(resolution, psf, lp=ConfigPrint.lp):
    """
    Creases the function for the dlw-model filter, given a resolution.
    :param resolution: Resolution of the simulation [px/um].
    :param psf: Point Spread Function intensity profile [kx, ky, kz], e.g. from utils_dose_sim.calc_psf_GT.
    :param lp: Laser Power.
    :return: Filter function, with density as its input.
    """
    accumulated_dose, dose_to_rho = _dose_model(resolution, psf)

    @jax.jit
    def dose_filter(rho_0):
        """
        The dose filter function, converts the regular density into a power accumulated density which can be
        converted into a printed structure using a threshold value.
        :param rho_0: Regular TopOpt density.
        :return: Accumulated power as a "density".
        """
        return dose_to_rho(accumulated_dose(rho_0 * ConfigPrint.power), lp)

    return dose_filter


def dose_filter_robust_f(resolution, psf, deviation):
    """
    Creases the function for the robust dlw-model filter, given a resolution.
    :param resolution: Resolution of the simulation [px/um].
    :param psf: Point Spread Function intensity profile [kx, ky, kz], e.g. from utils_dose_sim.calc_psf_GT.
    :param deviation: Relative deviation of the laser power for the eroded and dilated structure.
    :return: Filter function, with density as its input.
    """
    accumulated_dose, dose_to_rho = _dose_model(resolution, psf)
    lps = ConfigPrint.lp * jnp.array([1. - deviation, 1., 1. + deviation])

    @jax.jit
    def dose_filter_robust(rho_0):
        """
        The dose filter function, converts the regular density into a power accumulated density which can be
        converted into a printed structure using a threshold value.
        :param rho_0: Regular TopOpt density.
        :return: Accumulated power as a "density" for the eroded, normal and dilated laser power [3, nx, ny, nz].
        """
        # The laser power only enters after the convolution, so it is shared by all three structures
        conv = accumulated_dose(rho_0 * ConfigPrint.power)
        return dose_to_rho(conv[None], lps[:, None, None, None])

    return dose_filter_robust
