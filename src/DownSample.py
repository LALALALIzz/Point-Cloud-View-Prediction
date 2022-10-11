import os
import pandas as pd
import numpy as np
import scipy.signal as ss

def Decimator(dataset):
    decimated = []
    for i in range(dataset.shape[1]):
        decimated.append(ss.decimate(dataset[:, i], 5))
    decimated = np.stack(decimated, axis=1)
    return decimated

for i in range(1, 19):
    load_path = "../NJIT/node%dmobility.csv" %i
    save_path = "../NJIT_DownSample/node%ddownsample.csv" %i
    dataset = pd.read_csv(load_path).to_numpy()
    decimated = Decimator(dataset)
    df = pd.DataFrame(decimated, columns=['x', 'y', 'z', 'rx', 'ry', 'rz'])
    df.to_csv(save_path, index=False)