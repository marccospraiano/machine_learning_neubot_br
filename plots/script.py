import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

arr = np.array([2, 4])
arr2 = np.array([8,9])

plt.figure(figsize=(8,5))
plt.plot(arr, arr2)
plt.savefig(os.path.join('./../plots', 'plot.eps'))
plt.show();