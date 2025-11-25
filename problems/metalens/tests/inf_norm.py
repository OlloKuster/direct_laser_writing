import numpy as np

for i in np.linspace(1, 50, 50):
    x = np.array([0.3, 0.7, 0.1])
    print(f"i: {i}| {np.sum(x**i)**(1/i)}")