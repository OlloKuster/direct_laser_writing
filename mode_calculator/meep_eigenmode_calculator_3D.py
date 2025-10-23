import meep as mp
import numpy as np
import matplotlib.pyplot as plt


def find_mode_profile(simulation_domain, resolution, width, epsilon, wavelength, d_sub=0, offset_wg=0,
                      mode=1, until=50, parity=mp.ODD_Y, field=mp.Ez):

    cell = mp.Vector3(10, simulation_domain[1], simulation_domain[2])
    geometry = [mp.Block(size=mp.Vector3(mp.inf, width[0], width[1]),
                         center=mp.Vector3(0, -simulation_domain[1] / 2 + d_sub + width[0] / 2, offset_wg),
                         material=mp.Medium(epsilon=epsilon[2])),
                mp.Block(size=mp.Vector3(mp.inf, d_sub, mp.inf),
                         center=mp.Vector3(0, (-simulation_domain[1]+d_sub)/2, 0),
                         material=mp.Medium(epsilon=epsilon[0]))]
    pml_layers = [mp.PML(0.5)]

    fsrc = 1 / wavelength

    sources = [mp.EigenModeSource(src=mp.GaussianSource(fsrc, fwidth=0.1*fsrc),
                                  center=mp.Vector3(-0.5, -simulation_domain[1] / 2 + d_sub + width[0] / 2, offset_wg),
                                  size=mp.Vector3(y=width[0]+0.5, z=width[1]+0.5),
                                  # direction=mp.X,
                                  # eig_kpoint=mp.Vector3(x=1),
                                  # eig_band=mode,
                                  eig_parity=parity,
                                  eig_match_freq=True)]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        sources=sources,
                        resolution=resolution)

    sim.run(until=until)

    eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)

    ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=field)

    s = eps_data.shape
    s_y = -simulation_domain[1] / 2 + d_sub + width[0] / 2

    ind_max = np.argmax(ez_data[ez_data.shape[0] // 2, ez_data.shape[1] // 2])
    plt.imshow(eps_data[s[0] // 2].transpose(), interpolation='spline36', cmap='binary', origin='lower')
    plt.imshow(ez_data[s[0] // 2].transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9, origin='lower')
    # plt.axis('off')
    plt.savefig("./meep_yz.png")
    plt.close()
    plt.imshow(eps_data[:, 15].transpose(), interpolation='spline36', cmap='binary', origin='lower')
    plt.imshow(ez_data[:, 15].transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9, origin='lower')
    # plt.axis('off')
    plt.savefig("./meep_xz.png")
    plt.close()
    plt.imshow(eps_data[:, :, s[2]//2].transpose(), interpolation='spline36', cmap='binary', origin='lower')
    plt.imshow(ez_data[:, :, s[2]//2].transpose(), interpolation='spline36', cmap='RdBu', alpha=0.9, origin='lower')
    # plt.axis('off')
    plt.savefig("./meep_xy.png")
    plt.close()
    # plt.imshow(eps_data[s[0] // 2].transpose())
    # # plt.axis('off')
    # plt.savefig("./meep_yz.png")
    # plt.close()
    # plt.imshow(eps_data[:, s[1] // 2].transpose())
    # # plt.axis('off')
    # plt.savefig("./meep_xz.png")
    # plt.close()
    # plt.imshow(eps_data[:, :, ind_max].transpose())
    # # plt.axis('off')
    # plt.savefig("./meep_xy.png")
    # plt.close()
    return ez_data[ind_max]
