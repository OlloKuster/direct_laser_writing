import h5py
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import tidy3d as td
import matplotlib.colors as colors
import pyvista as pv



def polarization_splitter_regular_intermediate_plot():
    """
    Creates the plotting function used for the intermediate evulation of the structures.
    :return: Evaluation Plotter function.
    """

    def plotter(rho_init, rho_final, _, projection, i):
        plt.switch_backend('agg')
        fig, ax = plt.subplots(2, 2, sharex=True)
        ax[0, 0].imshow(rho_init[:, :, rho_init.shape[2] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)
        ax[0, 1].imshow(rho_init[:, rho_init.shape[1] // 2].T, origin='lower', cmap='binary', vmin=0, vmax=1)
        rho_final = projection(rho_final)
        ax[1, 0].imshow(rho_final[:, :, rho_init.shape[2] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)
        ax[1, 1].imshow(rho_final[:, rho_init.shape[2] // 2].T, origin='lower', cmap='binary', vmin=0, vmax=1)
        plt.savefig(f"problems/polarization_splitter/plots/progression/rho_{i:03d}.png")
        plt.close()

        sim_data_te = td.SimulationData.from_file("problems/polarization_splitter/plots/progression/current_simulation_te.hdf5")
        sim_data_tm = td.SimulationData.from_file("problems/polarization_splitter/plots/progression/current_simulation_tm.hdf5")
        fields_y = sim_data_te["Field Monitor"].Ey.squeeze()
        fields_z = sim_data_tm["Field Monitor"].Ez.squeeze()
        eps = sim_data_te["Permittivity Monitor"].eps_xx.real.squeeze()

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(eps.T, origin='lower', cmap='binary')
        ax[0].imshow(np.real(fields_y).T, origin='lower', cmap='RdBu_r', norm=colors.CenteredNorm(), alpha=0.9)
        ax[1].imshow(eps.T, origin='lower', cmap='binary')
        ax[1].imshow(np.real(fields_z).T, origin='lower', cmap='RdBu_r', norm=colors.CenteredNorm(), alpha=0.9)
        plt.savefig(f"problems/polarization_splitter/plots/progression/field_{i:03d}.png")
        plt.close()

        pv.global_theme.allow_empty_mesh = True
        p = pv.Plotter(off_screen=True)
        data = pv.wrap(np.array(rho_final))
        p.add_mesh(data.contour(), cmap='binary')
        p.camera_position = 'yz'
        p.camera.elevation = 30
        p.camera.azimuth = - 45
        p.remove_scalar_bar()
        p.camera.zoom(1.3)
        p.show(screenshot=f"problems/polarization_splitter/plots/progression/eps_{i:03d}.png")
        p.close()

        pv.plot(np.array(rho_final), off_screen=True, screenshot=f"problems/mode_converter/plots/progression/eps_density_{i:03d}.png", cmap='binary')

    return plotter


def polarization_splitter_regular_final_plot():
    """
    Creates the plotting function used for the final evaluation of the structures.
    :return: Final Plotter function.
    """

    def plotter(extent, rho_0=None, loss_hist=None, beta=None, em_loss_hist=None, eps=None, E=None, run_id=0,
                save=True):

        if loss_hist is not None:
            plt.plot(loss_hist)
            # plt.yscale('log')
            plt.savefig(
                f"problems/polarization_splitter/plots/loss_{run_id}_{beta}.png")
            plt.close()

        if beta is not None:
            plt.plot(em_loss_hist)
            plt.savefig(
                f"problems/polarization_splitter/plots/em_loss_{run_id}_{beta}.png")
            plt.close()

        if E is not None:
            plt.imshow(eps[:, :, eps.shape[2] // 4].T, origin='lower', cmap='binary',
                       extent=(0, extent[0], 0, extent[1]))
            plt.imshow(np.abs(E[0][E[0].shape[0] // 4].T), origin='lower', cmap="magma", alpha=0.8,
                       extent=(0, extent[0], 0, extent[1]))
            plt.xlabel(r"x ($\mathrm{\mu}$m)", fontsize=12)
            plt.ylabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
            plt.savefig(
                f"problems/polarization_splitter/plots/eps_and_e_{run_id}_{beta}.png")
            plt.close()
        if eps is not None:
            plt.imshow(eps[:, :, eps.shape[2] // 4].T, origin='lower', cmap='binary',
                       extent=(0, extent[0], 0, extent[1]))
            plt.xlabel(r"x ($\mathrm{\mu}$m)", fontsize=12)
            plt.ylabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
            plt.savefig(
                f"problems/polarization_splitter/plots/eps_{run_id}_{beta}.png")
            plt.close()

        if save:
            with h5py.File(
                    f"problems/polarization_splitter/plots/data_{run_id}_{beta}.h5",
                    'w') as f:
                grp = f.create_group("polarization_splitter")
                grp.create_dataset("eps", data=eps)
                grp.create_dataset("rho", data=rho_0)
                grp.create_dataset("loss", data=loss_hist)
                grp.create_dataset("em_loss", data=em_loss_hist)
                f.close()

    return plotter


def mode_converter_robust_intermediate_plot():
    """
    Creates the plotting function used for the intermediate evaluation of the robust (3) designed structures.
    :return: Evaluation Plotter function.
    """

    def plotter(rho_init, rho_final, projection, i):
        fig, ax = plt.subplots(2, 3, sharex=True)

        rho_final = projection(rho_final)

        rho_f_eroded = rho_final[0]
        rho_f_normal = rho_final[1]
        rho_f_dilated = rho_final[2]

        ax[0, 0].imshow(rho_init[:, :, rho_init.shape[2] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)
        ax[1, 0].imshow(rho_f_eroded[:, :, rho_f_eroded.shape[2] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)

        ax[0, 1].imshow(rho_init[:, :, rho_init.shape[2] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)
        ax[1, 1].imshow(rho_f_normal[:, :, rho_f_normal.shape[2] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)

        ax[0, 2].imshow(rho_init[:, :, rho_init.shape[2] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)
        ax[1, 2].imshow(rho_f_dilated[:, :, rho_f_dilated.shape[2] // 4].T, origin='lower', cmap='binary', vmin=0,
                        vmax=1)

        plt.savefig(f"problems/mode_converter/plots/progression/rho_{i:03d}.png")
        plt.close()

    return plotter


def mode_converter_robust_final_plot():
    """
    Creates the plotting function used for the final evaluation of the robust (3) designed structures.
    :return: Final Plotter function.
    """

    def plotter(extent, rho_0=None, loss_hist=None, beta=None, em_loss_hist=None, grads=None, eps=None, E=None,
                run_id=0,
                save=False):

        if loss_hist is not None:
            plt.plot(loss_hist, label='eroded')
            plt.legend()
            # plt.yscale('log')
            plt.savefig(
                f"problems/mode_converter/plots/loss_{run_id}_{beta}.png")
            plt.close()

        if em_loss_hist is not None:
            em_ero = []
            em_norm = []
            em_dil = []
            for i in range(len(em_loss_hist)):
                em_ero.append(em_loss_hist[i][0])
                em_norm.append(em_loss_hist[i][1])
                em_dil.append(em_loss_hist[i][2])
            plt.plot(em_ero, label='eroded')
            plt.plot(em_norm, label='normal')
            plt.plot(em_dil, label='dilated')
            plt.legend()
            plt.savefig(
                f"problems/mode_converter/plots/em_loss_{run_id}_{beta}.png")
            plt.close()

        if grads is not None:
            plt.plot(grads)
            plt.savefig(
                f"problems/mode_converter/plots/grads_{run_id}_{beta}.png")
            plt.xlabel("Iteration")
            plt.ylabel("Gradient")
            plt.close()

        if eps is not None:
            fig, axs = plt.subplots(1, 3)

            axs[0].imshow(eps[0][:, :, eps[0].shape[2] // 4].T, origin='lower', cmap='binary',
                          extent=(0, extent[0], 0, extent[1]))

            axs[1].imshow(eps[1][:, :, eps[0].shape[2] // 4].T, origin='lower', cmap='binary',
                          extent=(0, extent[0], 0, extent[1]))

            axs[2].imshow(eps[2][:, :, eps[0].shape[2] // 4].T, origin='lower', cmap='binary',
                          extent=(0, extent[0], 0, extent[1]))
            plt.savefig(
                f"problems/mode_converter/plots/eps_{run_id}_{beta}.png")
            plt.close()

        if save:
            with h5py.File(
                    f"problems/mode_converter/plots/data_{run_id}_{beta}.h5",
                    'w') as f:
                grp = f.create_group("mode_converter")
                grp.create_dataset("eps_erosion", data=eps[0])
                grp.create_dataset("eps_normal", data=eps[1])
                grp.create_dataset("eps_final", data=eps[2])
                grp.create_dataset("rho", data=rho_0)
                grp.create_dataset("loss", data=loss_hist)
                grp.create_dataset("em_loss", data=em_loss_hist)
                f.close()

    return plotter
