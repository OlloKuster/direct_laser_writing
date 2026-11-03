import jax.numpy as jnp


def lattice_steps(hatching, slicing, resolution):
    """
    Hatching and slicing distance in voxels of the simulation grid.
    :param hatching: Distance between the lines [um].
    :param slicing: Distance between the layers [um].
    :param resolution: Resolution of the simulation [px/um].
    :return: (m_y, m_z) Voxels from one line to the next and from one layer to the next.
    """
    m_y, m_z = hatching * resolution, slicing * resolution
    if abs(m_y - round(m_y)) > 1e-6 or abs(m_z - round(m_z)) > 1e-6:
        raise ValueError(f"Hatching ({hatching} um) and slicing ({slicing} um) must be multiples of the voxel size "
                         f"1/{resolution} um.")
    return round(m_y), round(m_z)


def lattice_shape(grid_shape, m_y, m_z, offset=(0, 0)):
    """
    Shape of the design variable on the writing lattice, one value per exposure position.
    :param grid_shape: Shape of the simulation grid.
    :param m_y: Voxels between two lines, see lattice_steps.
    :param m_z: Voxels between two layers, see lattice_steps.
    :param offset: Grid index of the first line and the first layer.
    :return: (nx, number of lines, number of layers).
    """
    return (grid_shape[0],
            len(range(offset[0], grid_shape[1], m_y)),
            len(range(offset[1], grid_shape[2], m_z)))


def lattice_proj_jax_f(f2bin, grid_shape, m_y, m_z, offset=(0, 0)):
    """
    Generates the initial projection onto the writing lattice. The scan runs along x and is sampled at every voxel,
    the lines are m_y voxels apart in y and the layers m_z voxels apart in z.
    :param f2bin: Binarization of the exposure on the lattice, e.g. tanh_filter_jax_f(alpha, beta).
    :param grid_shape: Shape of the simulation grid.
    :param m_y: Voxels between two lines, see lattice_steps.
    :param m_z: Voxels between two layers, see lattice_steps.
    :param offset: Grid index of the first line and the first layer.
    :return: Lattice projection function, design [nx, n_lines, n_layers] -> written pattern on the grid [nx, ny, nz],
             which is zero between the lines and layers.
    """
    o_y, o_z = offset

    def lattice_proj(rho):
        """
        Binarizes the exposure of every lattice point and places it on the simulation grid.
        :param rho: Design variable on the writing lattice [nx, n_lines, n_layers].
        :return: Written pattern on the simulation grid [nx, ny, nz].
        """
        written = f2bin(rho)
        return jnp.zeros(grid_shape, written.dtype).at[:, o_y::m_y, o_z::m_z].set(written)

    return lattice_proj
