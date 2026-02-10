import h5py
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp


def multiplexer_regular_intermediate_plot():
    """
    Creates the plotting function used for the intermediate evulation of the structures.
    :return: Evaluation Plotter function.
    """
    def plotter(rho_init, rho_final, projection, i):
        plt.switch_backend('agg')
        fig, ax = plt.subplots(2, 2, sharex=True)
        ax[0, 0].imshow(rho_init[:, :, rho_init.shape[2] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)
        ax[0, 1].imshow(rho_init[:, rho_init.shape[1] // 2].T, origin='lower', cmap='binary', vmin=0, vmax=1)
        rho_final = projection(rho_final)
        ax[1, 0].imshow(rho_final[:, :, rho_init.shape[2] // 4].T, origin='lower', cmap='binary', vmin=0, vmax=1)
        ax[1, 1].imshow(rho_final[:, rho_init.shape[2] // 2].T, origin='lower', cmap='binary', vmin=0, vmax=1)
        plt.savefig(f"problems/multiplexer/plots/progression/rho_{i:03d}.png")
        plt.close()

    return plotter


def multiplexer_regular_final_plot():
    """
    Creates the plotting function used for the final evaluation of the structures.
    :return: Final Plotter function.
    """
    def plotter(extent, rho_0=None, loss_hist=None, beta=None, em_loss_hist=None, eps=None, E=None, run_id=0,
                save=False):

        if loss_hist is not None:
            plt.plot(loss_hist)
            # plt.yscale('log')
            plt.savefig(
                f"problems/multiplexer/plots/loss_{run_id}_{beta}.png")
            plt.close()

        if beta is not None:
            plt.plot(em_loss_hist)
            plt.savefig(
                f"problems/multiplexer/plots/em_loss_{run_id}_{beta}.png")
            plt.close()

        if E is not None:
            plt.imshow(eps[:, :, eps.shape[2] // 4].T, origin='lower', cmap='binary',
                       extent=(0, extent[0], 0, extent[1]))
            plt.imshow(np.abs(E[0][E[0].shape[0] // 4].T), origin='lower', cmap="magma", alpha=0.8,
                       extent=(0, extent[0], 0, extent[1]))
            plt.xlabel(r"x ($\mathrm{\mu}$m)", fontsize=12)
            plt.ylabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
            plt.savefig(
                f"problems/multiplexer/plots/eps_and_e_{run_id}_{beta}.png")
            plt.close()
        if eps is not None:
            plt.imshow(eps[:, :, eps.shape[2] // 4].T, origin='lower', cmap='binary',
                       extent=(0, extent[0], 0, extent[1]))
            plt.xlabel(r"x ($\mathrm{\mu}$m)", fontsize=12)
            plt.ylabel(r"y ($\mathrm{\mu}$m)", fontsize=12)
            plt.savefig(
                f"problems/multiplexer/plots/eps_{run_id}_{beta}.png")
            plt.close()

        if save:
            with h5py.File(
                    f"problems/multiplexer/plots/data_{run_id}_{beta}.h5",
                    'w') as f:
                grp = f.create_group("multiplexer")
                grp.create_dataset("eps", data=eps)
                grp.create_dataset("rho", data=rho_0)
                grp.create_dataset("loss", data=loss_hist)
                grp.create_dataset("em_loss", data=em_loss_hist)
                f.close()

    return plotter

