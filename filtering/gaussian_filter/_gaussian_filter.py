import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve
from jax.lax import conv_general_dilated as convolve_lax
from jax.lax import conv_dimension_numbers


def gaussian_filter_jax_f(sigma):
    '''
    Generates a Gauss filter with standard deviation sigma.
    :param sigma: Standard deviation of our Gaussian. Feature size is roughly sqrt(3)*sigma [px].
    :return: A gauss filter function.
    '''
    sigma = sigma

    def f2gauss_lax(rho_0):
        """
        Uses jax to apply a Gaussian filter to x with standard deviation sigma.
        :param rho_0: The array where the Gaussian blur will be applied to.
        :return: Gaussian blurred array.
        """

        def gkernel(sigma):
            l = int(jnp.ceil(4.0 * sigma) + 1)
            ax = jnp.linspace(-(l - 1) / 2., (l - 1) / 2., l)
            xx, yy, zz = jnp.meshgrid(ax, ax, ax)

            kernel = jnp.exp(-0.5 * (xx ** 2 + yy ** 2 + zz ** 2) / sigma ** 2)
            return kernel / jnp.sum(kernel)

        kernel = gkernel(sigma)[:, :, :, jnp.newaxis, jnp.newaxis]
        rho_0 = rho_0[jnp.newaxis, :, :, :, jnp.newaxis]
        dn = conv_dimension_numbers(rho_0.shape, kernel.shape,
                                    ('NHWDC', 'HWDIO', 'NHWDC'))
        res = convolve_lax(rho_0, kernel, (1, 1, 1), 'SAME', (1, 1, 1), (1, 1, 1), dn)
        return res.squeeze()

    return f2gauss_lax


def conic_filter_jax_f(radius):
    """
    Generates a conic filter with radius radius.
    :param radius: Radius of the cone. Feature size is roughly 2*sigma [px].
    :return: A cone filter function.
    """
    def f2conic_lax(rho_0):
        """
        Uses jax to apply a Gaussian filter to x with standard deviation sigma.
        :param rho_0: The array where the Gaussian blur will be applied to.
        :return: Gaussian blurred array.
        """

        def gkernel(radius):
            l = int(2 * radius + 1)
            ax = jnp.linspace(-(l - 1) / 2., (l - 1) / 2., l)
            xx, yy, zz = jnp.meshgrid(ax, ax, ax)

            kernel = jnp.maximum(radius - jnp.sqrt(xx ** 2 + yy ** 2 + zz ** 2), jnp.zeros_like(xx))
            return kernel / jnp.sum(kernel)

        kernel = gkernel(radius)[:, :, :, jnp.newaxis, jnp.newaxis]
        rho_0 = rho_0[jnp.newaxis, :, :, :, jnp.newaxis]
        dn = conv_dimension_numbers(rho_0.shape, kernel.shape,
                                    ('NHWDC', 'HWDIO', 'NHWDC'))
        res = convolve_lax(rho_0, kernel, (1, 1, 1), 'SAME', (1, 1, 1), (1, 1, 1), dn)
        res = jax.image.resize(res.squeeze(), rho_0.squeeze().shape, "cubic")
        return res

    return f2conic_lax
