import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

#Load data
os.chdir('../../gpupanda')
python_gpu2_times= [.10, .11, .10] #single precision, alpha=0.1, Tfunction
python_gpu2_time = np.mean(python_gpu2_times)
matlab_gpu2_time = [0.720859] #single precision, alpha=0.1, Tfunction

#Check that (small, alpha=0.1, single precision) networks are identical in gpuPANDA and PANDA in python
singleSmallGPU=pd.read_csv('data/single_gpu_panda.txt',sep=' ', header=None)
singleSmallCPU=pd.read_csv('data/single_cpu_panda.txt',sep=' ', header=None)
np.max(np.max(np.abs(singleSmallGPU-singleSmallCPU))) #3.500000000045134e-05

#Plot figure
plt.rcParams["font.family"] = "Arial"
plt.subplot(122)
plt.plot(singleSmallGPU.values.flatten(), singleSmallCPU.values.flatten(), 'o',
         markeredgewidth=2, alpha=0.5)
plt.text(15, 0, '3.5e-5', fontsize=12)
plt.xlabel('gpuPANDA edge weights')
plt.ylabel('PANDA edge weights')
plt.plot([-6,26],[-6,26])
plt.subplot(121)
plt.bar([1,2],[python_gpu2_time,matlab_gpu2_time])
plt.ylabel('Run time (s)')
plt.xticks([1,2], ('gpuPANDA (Python)', 'gpuPANDA (MATLAB)'))