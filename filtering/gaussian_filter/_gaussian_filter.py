import jax.numpy as jnp
from jax.scipy.signal import convolve


def gaussian_filter_jax_f(sigma):
    def f2gauss(rho_0):
        """
        Uses jax to apply a Gaussian filter to x with standard deviation sigma.
        :param rho_0: The array where the Gaussian blur will be applied to.
        :return: Gaussian blurred array.
        """
        def gkernel(sigma):
            l = int(2 * jnp.ceil(4.0 * sigma) + 1)
            ax = jnp.linspace(-(l - 1) / 2., (l - 1) / 2., l)
            xx, yy, zz = jnp.meshgrid(ax, ax, ax)

            kernel = jnp.exp(-0.5 * (xx ** 2 + yy ** 2 + zz ** 2) / sigma ** 2)
            return kernel / jnp.sum(kernel)

        kernel = gkernel(sigma)
        res = convolve(rho_0, kernel, mode='same')
        return res

    return f2gauss
