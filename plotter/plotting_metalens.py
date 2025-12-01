import h5py
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp


def metalens_regular_intermediate_plot():
    def plotter(rho_init, rho_final, projection, i):
        fig, ax = plt.subplots(2, 1, sharex=True)
        rho_init = np.concatenate((rho_init, np.flip(rho_init, axis=0)), axis=0)
        rho_init = np.concatenate((rho_init, np.flip(rho_init, axis=1)), axis=1)
        ax[0].imshow(rho_init[rho_init.shape[0] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)
        ax[0].set_ylabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
        rho_final = np.concatenate((rho_final, np.flip(rho_final, axis=0)), axis=0)
        rho_final = np.concatenate((rho_final, np.flip(rho_final, axis=1)), axis=1)
        rho_final = projection(rho_final)
        ax[1].imshow(rho_final[rho_final.shape[0] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)
        ax[1].set_xlabel(r"x ($\mathrm{\mu}$m)", fontsize=12)
        ax[1].set_ylabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
        plt.savefig(f"problems/metalens/plots/progression/rho_{i:03d}.png")
        plt.show()

    return plotter


def metalens_regular_final_plot():
    def plotter(extent, rho_0=None, loss_hist=None, beta=None, em_loss_hist=None, grads=None, eps=None, E=None,
                save=False):

        if loss_hist is not None:
            plt.plot(loss_hist)
            # plt.yscale('log')
            plt.savefig(
                f"problems/metalens/plots/loss_{beta}.png")
            plt.close()

        if beta is not None:
            plt.plot(em_loss_hist)
            plt.savefig(
                f"problems/metalens/plots/em_loss_{beta}.png")
            plt.close()

        if grads is not None:
            plt.plot(grads)
            plt.savefig(
                f"problems/metalens/plots/grads_{beta}.png")
            plt.xlabel("Iteration")
            plt.ylabel("Gradient")
            plt.close()

        if E is not None:
            plt.imshow(eps[eps.shape[0] // 2].T, origin='lower', cmap='binary',
                       extent=(0, extent[0], 0, extent[1]))
            plt.imshow(np.abs(E[0][E[0].shape[0] // 2].T), origin='lower', cmap="magma", alpha=0.8,
                       extent=(0, extent[0], 0, extent[1]))
            plt.xlabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
            plt.ylabel(r"z ($\mathrm{\mu}$m)", fontsize=12)
            plt.savefig(
                f"problems/metalens/plots/eps_and_e_{beta}.png")
            plt.close()
        if eps is not None:
            plt.imshow(eps[eps.shape[0] // 2].T, origin='lower', cmap='binary',
                       extent=(0, extent[0], 0, extent[1]))
            plt.xlabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
            plt.ylabel(r"z ($\mathrm{\mu}$m)", fontsize=12)
            plt.savefig(
                f"problems/metalens/plots/eps_{beta}.png")
            plt.close()

        if save:
            with h5py.File(
                    f"problems/metalens/plots/data_{beta}.h5",
                    'w') as f:
                grp = f.create_group("lens_3d")
                grp.create_dataset("E", data=E)
                grp.create_dataset("eps", data=eps)
                grp.create_dataset("rho", data=rho_0)
                grp.create_dataset("loss", data=loss_hist)
                grp.create_dataset("em_loss", data=em_loss_hist)
                f.close()

    return plotter


def metalens_robust_intermediate_plot():
    def plotter(rho_init, rho_final, projection, i):
        fig, ax = plt.subplots(2, 3, sharex=True)

        rho_final = projection(rho_final)

        rho_f_eroded = rho_final[0]
        rho_f_normal = rho_final[1]
        rho_f_dilated = rho_final[2]

        rho_eroded = np.concatenate((rho_init, np.flip(rho_init, axis=0)), axis=0)
        rho_eroded = np.concatenate((rho_eroded, np.flip(rho_eroded, axis=1)), axis=1)
        ax[0, 0].imshow(rho_eroded[rho_eroded.shape[0] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)
        rho_f_eroded = np.concatenate((rho_f_eroded, np.flip(rho_f_eroded, axis=0)), axis=0)
        rho_f_eroded = np.concatenate((rho_f_eroded, np.flip(rho_f_eroded, axis=1)), axis=1)
        ax[1, 0].imshow(rho_f_eroded[rho_f_eroded.shape[0] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)

        rho_normal = np.concatenate((rho_init, np.flip(rho_init, axis=0)), axis=0)
        rho_normal = np.concatenate((rho_normal, np.flip(rho_normal, axis=1)), axis=1)
        ax[0, 1].imshow(rho_normal[rho_normal.shape[0] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)
        rho_f_normal = np.concatenate((rho_f_normal, np.flip(rho_f_normal, axis=0)), axis=0)
        rho_f_normal = np.concatenate((rho_f_normal, np.flip(rho_f_normal, axis=1)), axis=1)
        ax[1, 1].imshow(rho_f_normal[rho_f_normal.shape[0] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)

        rho_dilated = np.concatenate((rho_init, np.flip(rho_init, axis=0)), axis=0)
        rho_dilated = np.concatenate((rho_dilated, np.flip(rho_dilated, axis=1)), axis=1)
        ax[0, 2].imshow(rho_dilated[rho_dilated.shape[0] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)
        rho_f_dilated = np.concatenate((rho_f_dilated, np.flip(rho_f_dilated, axis=0)), axis=0)
        rho_f_dilated = np.concatenate((rho_f_dilated, np.flip(rho_f_dilated, axis=1)), axis=1)
        ax[1, 2].imshow(rho_f_dilated[rho_f_dilated.shape[0] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)

        plt.savefig(f"problems/metalens/plots/progression/rho_{i:03d}.png")
        plt.close()

    return plotter


def metalens_robust_final_plot():
    def plotter(extent, rho_0=None, loss_hist=None, beta=None, em_loss_hist=None, grads=None, eps=None, E=None,
                save=False):

        if loss_hist is not None:
            plt.plot(loss_hist, label='eroded')
            plt.legend()
            # plt.yscale('log')
            plt.savefig(
                f"problems/metalens/plots/loss_{beta}.png")
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
                f"problems/metalens/plots/em_loss_{beta}.png")
            plt.close()

        if grads is not None:
            plt.plot(grads)
            plt.savefig(
                f"problems/metalens/plots/grads_{beta}.png")
            plt.xlabel("Iteration")
            plt.ylabel("Gradient")
            plt.close()

        if E is not None:
            fig, axs = plt.subplots(1, 3)

            axs[0].imshow(eps[0][eps[0].shape[0] // 2].T, origin='lower', cmap='binary',
                          extent=(0, extent[0], 0, extent[1]))
            axs[0].imshow(np.abs(E[0][0][E[0][0].shape[0] // 2].T), origin='lower', cmap="magma", alpha=0.8,
                          extent=(0, extent[0], 0, extent[1]))

            axs[1].imshow(eps[1][eps[0].shape[0] // 2].T, origin='lower', cmap='binary',
                          extent=(0, extent[0], 0, extent[1]))
            axs[1].imshow(np.abs(E[1][0][E[0][0].shape[0] // 2].T), origin='lower', cmap="magma", alpha=0.8,
                          extent=(0, extent[0], 0, extent[1]))

            axs[2].imshow(eps[2][eps[0].shape[0] // 2].T, origin='lower', cmap='binary',
                          extent=(0, extent[0], 0, extent[1]))
            axs[2].imshow(np.abs(E[2][0][E[0][0].shape[0] // 2].T), origin='lower', cmap="magma", alpha=0.8,
                          extent=(0, extent[0], 0, extent[1]))

            plt.savefig(
                f"problems/metalens/plots/eps_and_e_{beta}.png")
            plt.close()

        if eps is not None:
            fig, axs = plt.subplots(1, 3)

            axs[0].imshow(eps[0][eps[0].shape[0] // 2].T, origin='lower', cmap='binary',
                          extent=(0, extent[0], 0, extent[1]))

            axs[1].imshow(eps[1][eps[0].shape[0] // 2].T, origin='lower', cmap='binary',
                          extent=(0, extent[0], 0, extent[1]))

            axs[2].imshow(eps[2][eps[0].shape[0] // 2].T, origin='lower', cmap='binary',
                          extent=(0, extent[0], 0, extent[1]))
            plt.savefig(
                f"problems/metalens/plots/eps_{beta}.png")
            plt.close()

        if save:
            with h5py.File(
                    f"problems/metalens/plots/data_{beta}.h5",
                    'w') as f:
                grp = f.create_group("lens_3d")
                grp.create_dataset("E_erosion", data=E[0])
                grp.create_dataset("E_normal", data=E[1])
                grp.create_dataset("E_dilation", data=E[2])
                grp.create_dataset("eps_erosion", data=eps[0])
                grp.create_dataset("eps_normal", data=eps[1])
                grp.create_dataset("eps_final", data=eps[2])
                grp.create_dataset("rho", data=rho_0)
                grp.create_dataset("loss", data=loss_hist)
                grp.create_dataset("em_loss", data=em_loss_hist)
                f.close()

    return plotter
