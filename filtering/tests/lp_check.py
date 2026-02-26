import numpy as np
import scipy.ndimage
import torch
import matplotlib.pyplot as plt
import scipy

from filtering._filter_loader import filter_loader
from filtering.dose_model._dose_filter import dose_filter_f
from filtering.dose_model.config_print import ConfigPrint
from projection.SSP.subpixel_smoothed_projection import ssp_proj_jax_f
from projection._projection_loader import projection_loader


def test(seed):
    np.random.seed(seed)
    resolution = 10
    for factor in np.linspace(0.0072, 0.0076, 1):
        ConfigPrint.lp = factor
        size_lat = int(np.ceil(0.4*resolution))
        size_ax = int(np.ceil(0.9*resolution))
        print(size_ax)
        rho_0 = np.zeros((5*resolution, 5*resolution, 5*resolution))
        rho_0[rho_0.shape[0]//2-size_lat:-rho_0.shape[0]//2+size_lat, rho_0.shape[1]//2-size_lat:-rho_0.shape[1]//2+size_lat,  rho_0.shape[2]//2-size_ax:-rho_0.shape[2]//2+size_ax] = 1
        #
        # rho_0[rho_0.shape[0]//2, rho_0.shape[1]//2-4, rho_0.shape[2]//4:-rho_0.shape[2]//4] = 1
        # # rho_0[rho_0.shape[0]//2, rho_0.shape[1]//2, rho_0.shape[2]//4:-rho_0.shape[2]//4] = 1
        # rho_0[rho_0.shape[0]//2, rho_0.shape[1]//2-2, rho_0.shape[2]//4:-rho_0.shape[2]//4] = 1
        # rho_0[rho_0.shape[0]//2, rho_0.shape[1]//2, rho_0.shape[2]//4:-rho_0.shape[2]//4] = 1
        # rho_0[rho_0.shape[0]//2, rho_0.shape[1]//2+2, rho_0.shape[2]//4:-rho_0.shape[2]//4] = 1
        # # rho_0[rho_0.shape[0]//2, rho_0.shape[1]//2, rho_0.shape[2]//4:-rho_0.shape[2]//4] = 1
        # rho_0[rho_0.shape[0]//2, rho_0.shape[1]//2+4, rho_0.shape[2]//4:-rho_0.shape[2]//4] = 1


        rho_0_torch = torch.tensor(rho_0, device='cuda', requires_grad=True)
        dose_filter = filter_loader("dose_conv", resolution)
        proj = projection_loader("ssp_jax", 0.5, 8, resolution)

        rho_filt = dose_filter(rho_0_torch)
        result = rho_filt.detach().cpu().numpy().squeeze()
        result = proj(result)
        result_bin = np.where(result > ConfigPrint.rho_th_GT, 1, 0)
        f, ax = plt.subplots(1, 3)
        ax[0].imshow(rho_0[rho_0.shape[0]//2].T, vmin=0, vmax=1, origin='lower')
        ax[1].imshow(result[result.shape[0]//2].T, vmin=0, vmax=1, origin='lower')
        ax[2].imshow(result_bin[result_bin.shape[0]//2].T, vmin=0, vmax=1, origin='lower')
        plt.savefig(f"plots_1/lp_{factor:3f}.png")
        plt.close()

        import pyvista as pv
        p = pv.Plotter()
        data = pv.wrap(np.array(result_bin))
        p.add_mesh(data.contour(), cmap='binary')
        p.camera_position = 'yz'
        p.camera.elevation = 30
        p.camera.azimuth = - 45
        p.add_axes()
        # p.remove_scalar_bar()
        p.camera.zoom(1.3)
        p.show()
        if result[result.shape[0]//2, result.shape[1]//2, result.shape[2]//2] >= 0.5:
            print(factor)
            break
        print(f"lp: {factor}")
        print(f"bin_value: {result_bin[result.shape[0]//2, result.shape[1]//2, result.shape[2]//2]}")
        print(f"actual_value: {result[result.shape[0]//2, result.shape[1]//2, result.shape[2]//2]}")
    return

if __name__ == "__main__":
    test(42342)