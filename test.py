import matplotlib.pyplot as plt
import numpy as np

plt.figure(dpi=300)

x = np.linspace(0, 1000, 1000)
y = (1/2) * x + 30 * np.sin(x / 60)
test_case = np.array([x, y]).T

plt.scatter(x, y, s=1)

plt.tight_layout()

plt.draw()

plt.pause(0.0001)

plt.clf()