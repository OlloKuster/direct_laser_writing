import jax.numpy as jnp
import autograd.numpy as anp


def tanh_filter_jax_f():
    def f2bin(rho, alpha=0.5, beta=30):
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


def tanh_filter_ag_f():
    def f2bin(rho_0, alpha=0.5, beta=30):
        """
        Binarises the values of x with parameters alpha and beta.
        :param rho_0: Array which will be binarised.
        :param alpha: Steepness of the binarisation function.
        :param beta: Origin of the binarisation function.
        :return: Binarised array of x.
        """
        if beta == anp.inf:
            return anp.where(rho_0 > alpha, 1.0, 0.0)
        else:
            num = anp.tanh(alpha * beta) + anp.tanh(beta * (rho_0 - alpha))
            denom = anp.tanh(alpha * beta) + anp.tanh(beta * (1 - alpha))
            proj = num / denom
            return proj

    return f2bin
