import jax.numpy as jnp


def tanh_filter_jax_f(alpha=0.5, beta=30):
    """
    Regular tanh-projection function generator.
    :param alpha: Threshold value.
    :param beta: Binarization level.
    :return: Tanh-Projection function
    """
    def f2bin(rho):
        """
        Binarises the values of x with parameters alpha and beta.
        :param rho: Array which will be binarised.
        :param alpha: Steepness of the binarisation function.
        :param beta: Origin of the binarisation function.
        :return: Binarised array of x.
        """
        if beta == jnp.inf:
            return jnp.where(rho > alpha, 1.0, 0.0)
        else:
            num = jnp.tanh(alpha * beta) + jnp.tanh(beta * (rho - alpha))
            denom = jnp.tanh(alpha * beta) + jnp.tanh(beta * (1 - alpha))
            proj = num / denom
            return proj

    return f2bin

