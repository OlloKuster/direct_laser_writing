import numpy as np
import matplotlib.pyplot as plt

from filtering.dose_model.config_print import ConfigPrint
from projection._projection_loader import projection_loader


def test():
    xx = np.linspace(0, 1, 100)
    projection = projection_loader("tanh_jax", ConfigPrint.rho_0_GT, 16, 10)

    proj = projection(xx)
    print(projection(0.07188))
    plt.plot(proj)
    plt.show()


if __name__ == "__main__":
    test()