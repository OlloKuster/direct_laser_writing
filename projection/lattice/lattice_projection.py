import jax.numpy as jnp
import numpy as np

from filtering.dose_model.config_print import ConfigPrint


def lattice_steps(hatching, slicing, resolution):
    """
    Hatching and slicing distance in voxels of the simulation grid.
    :param hatching: Distance between the lines [um].
    :param slicing: Distance between the layers [um].
    :param resolution: Resolution of the simulation [px/um].
    :return: (m_y, m_z) Voxels from one line to the next and from one layer to the next.
    """
    m_y, m_z = hatching * resolution, slicing * resolution
    return round(m_y), round(m_z)


def lattice_proj_jax_f(f2bin, resolution, offset=(0, 0)):
    """
    Generates the initial projection onto the writing lattice (ConfigPrint.hatching, ConfigPrint.slicing). The scan
    runs along x and is sampled at every voxel, the lines are hatching apart in y and the layers slicing apart in z.
    :param f2bin: Binarization of the exposure, e.g. tanh_filter_jax_f(alpha, beta).
    :param resolution: Resolution of the simulation [px/um].
    :param offset: Grid index of the first line and the first layer.
    :return: Lattice projection function, design [nx, ny, nz] -> written pattern [nx, ny, nz], which is zero between
             the lines and layers.
    """
    m_y, m_z = lattice_steps(ConfigPrint.hatching, ConfigPrint.slicing, resolution)
    o_y, o_z = offset

    def lattice_proj(rho):
        """
        Binarizes the design and only keeps it at the exposure positions of the writing lattice.
        :param rho: Design variable [nx, ny, nz].
        :return: Written pattern [nx, ny, nz].
        """
        on_lattice = np.zeros(rho.shape, dtype=bool)  # Static, only depends on the shape
        on_lattice[:, o_y::m_y, o_z::m_z] = True
        return jnp.where(on_lattice, f2bin(rho), 0.)

    return lattice_proj


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    f2bin = lambda x: x

    lattice_proj = lattice_proj_jax_f(f2bin,  20)

    rho = jnp.array(np.random.rand(100, 100, 100))

    proj = lattice_proj(rho)

    plt.imshow(proj[:, :, 0].T, origin='lower', extent=(0, 5, 0, 5))
    plt.show()
