import numpy as np
import torch
import matplotlib.pyplot as plt

from filtering._filter_loader import filter_loader


def test(seed):
    np.random.seed(seed)
    resolution = 10
    # rho_0_init = np.random.random((10*resolution, 10*resolution, 5*resolution))
    # rho_0_g = gaussian_filter(rho_0_init, 15)
    # rho_0 = np.round(rho_0_g)
    rho_0 = np.zeros((5*resolution, 5*resolution, 5*resolution))
    rho_0[:rho_0.shape[0]//4, :rho_0.shape[1]//4, :rho_0.shape[2]//4] = 1
    rho_0_torch = torch.tensor(rho_0, device='cuda', requires_grad=True)
    dose_filter = filter_loader("dose_conv", resolution)

    rho_filt = dose_filter(rho_0_torch)
    rho_filt_tmp = rho_filt
    # rho_filt_np = rho_filt > 0.5
    rho_filt_np = rho_filt.detach().cpu().numpy()

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(rho_0[rho_0.shape[0]//6].T, origin='lower', cmap='binary', extent=(0, 10, 0, 5))
    ax[0, 1].imshow(rho_filt_np[rho_0.shape[0]//4].T, origin='lower', cmap='binary', extent=(0, 10, 0, 5))

    # factor = 2
    # rho_0 = scipy.ndimage.zoom(rho_0_g, factor)
    # rho_0 = np.round(rho_0)
    # rho_0_torch = torch.tensor(rho_0, device='cuda', requires_grad=True)
    # dose_filter = filter_loader("dose_conv", resolution * factor)
    #
    # rho_filt = dose_filter(rho_0_torch)
    # rho_filt_np = rho_filt > 0.5
    # rho_filt_np = rho_filt_np.detach().cpu().numpy()
    #
    # ax[1, 0].imshow(rho_0[rho_0.shape[0]//2].T, origin='lower', cmap='binary', extent=(0, 10, 0, 5))
    # ax[1, 1].imshow(rho_filt_np[rho_0.shape[0]//2].T, origin='lower', cmap='binary', extent=(0, 10, 0, 5))
    plt.show()

    # sum = torch.sum(rho_filt)
    # sum.backward()
    # print(rho_0_torch.grad.sum())
    # grad = rho_0_torch.grad
    # grad_np = grad.detach().cpu().numpy()
    #
    # plt.imshow(grad_np[grad_np.shape[0]//2].T, origin='lower')
    # plt.show()


if __name__ == "__main__":
    test(152)