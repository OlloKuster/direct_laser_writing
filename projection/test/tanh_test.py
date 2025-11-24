import numpy as np
import matplotlib.pyplot as plt

from filtering.dose_model.config_print import ConfigPrint
from projection._projection_loader import projection_loader


def test():
    xx = np.linspace(0, 1, 100)
    projection = projection_loader("ssp_jax", 0.5, 100, 10)

    proj = projection(xx)
    print(projection(np.ones_like(xx)*2))
    plt.plot(proj)
    plt.show()


if __name__ == "__main__":
    test()