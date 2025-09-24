import numpy as np
import matplotlib.pyplot as plt

from filtering._filter_loader import filter_loader


def test(seed):
    sigma = 2
    np.random.seed(seed)
    rho_0 = np.random.random((50, 50, 50))
    unity = filter_loader("None", sigma)

    rho_filt = unity(rho_0)

    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(rho_0[rho_0.shape[0]//2].T, origin='lower', cmap='binary')
    ax1.imshow(rho_filt[rho_0.shape[0]//2].T, origin='lower', cmap='binary')
    plt.show()


if __name__ == "__main__":
    test(123)