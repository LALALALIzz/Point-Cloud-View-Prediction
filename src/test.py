import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss

if __name__ == "__main__":
    '''
    data = pd.read_csv('../NJIT/node1mobility.csv').to_numpy()
    data = data[:5000, :3]
    decimated = []
    for i in range(data.shape[1]):
        decimated.append(ss.decimate(data[:, i], 5))
    decimated = np.stack(decimated, axis=1)
    print(decimated.shape)
    x2 = np.linspace(0, 5000, len(decimated))
    x = np.linspace(0, 5000, 5000)
    plt.subplot(3, 1, 1)
    plt.plot(x, data[:, 0])
    plt.plot(x2, decimated[:, 0], 'r')
    plt.subplot(3, 1, 2)
    plt.plot(x, data[:, 1])
    plt.plot(x2, decimated[:, 1], 'b')
    plt.subplot(3, 1, 3)
    plt.plot(x, data[:, 2])
    plt.plot(x2, decimated[:, 2], 'g')
    plt.show()
    '''
    x = np.linspace(1, 100, 100)
    y = 100/(100 - 0.95 * x)
    # print(y)
    plt.plot(x, y, 'r')
    plt.grid()
    plt.xlabel('Percentage encryption')
    plt.ylabel('Net speedup')
    plt.show()
