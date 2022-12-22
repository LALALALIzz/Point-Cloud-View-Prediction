import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
def hidden_state(h, x):
    return h ** 2 + 3 * x

if __name__ == '__main__':
    index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    # mean_list = []
    stdx_list = []
    stdy_list = []
    stdz_list = []
    csv_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'NJIT_DownSample', 'node%ddownsample.csv'))
    for i in index:
        path = csv_path % i
        dataset = pd.read_csv(path).to_numpy()
        stdx_list.append(np.std(dataset[:, 0]))
        stdy_list.append(np.std(dataset[:, 1]))
        stdz_list.append(np.std(dataset[:, 2]))
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(index, stdx_list, 'o')
    plt.subplot(3, 1, 2)
    plt.plot(index, stdy_list, 'o')
    plt.subplot(3, 1, 3)
    plt.plot(index, stdz_list, 'o')
    plt.show()