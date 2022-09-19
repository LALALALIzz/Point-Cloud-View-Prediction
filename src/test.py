import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    x = np.arange(1, 101)
    y = 1 / (1 - 14/1500 * x)
    plt.figure()
    plt.title("Net Speedup vs Percent Vectorization")
    plt.plot(x, y)
    plt.xlabel("Percentage Vectorization")
    plt.ylabel("Net Speedup")
    plt.grid()
    plt.show()