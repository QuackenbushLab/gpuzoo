import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

#Load data
os.chdir('/Users/mab8354/projects/gpupanda')
python_gpu2_times= [.10, .11, .10] #single precision, alpha=0.1, Tfunction
python_gpu2_time = np.mean(python_gpu2_times)
python_cpu2_times = [1.77, 1.55, 1.66] #single precision, alpha=0.1, Tfunction
python_cpu2_time = np.mean(python_cpu2_times)

#Check that (small, alpha=0.1, single precision) networks are identical in gpuPANDA and PANDA in python
singleSmallGPU=pd.read_csv('data/Python/single_gpu_panda.txt',sep=' ', header=None)
singleSmallCPU=pd.read_csv('data/Python/single_cpu_panda.txt',sep=' ', header=None)
np.max(np.max(np.abs(singleSmallGPU-singleSmallCPU))) #3.500000000045134e-05

#Plot figure
plt.rcParams["font.family"] = "Arial"
plt.subplot(122)
plt.plot(singleSmallGPU.values.flatten(), singleSmallCPU.values.flatten(), 'o',
         markeredgewidth=2, alpha=0.5)
plt.text(5, 0, 'Maxdiff = $3.5*10^{-5}$', fontsize=12)
plt.xlabel('gpuPANDA edge weights')
plt.ylabel('PANDA edge weights')
plt.plot([-6,26],[-6,26])
plt.subplot(121)
plt.bar([1,2],[python_gpu2_time,python_cpu2_time])
plt.ylabel('Run time (s)')
plt.xticks([1,2], ('gpuPANDA (Python)', 'PANDA (Python)'))

#Plot gpuLIONESS
gpu1_lioness=pd.read_csv('data/MATLAB/lioness/LIONESS_gpu_gpu1_resTable.csv')
gpu1_online_lioness=pd.read_csv('data/MATLAB/lioness/LIONESS_gpu_gpu1_online_resTable.csv')
gpu2_lioness=pd.read_csv('data/MATLAB/lioness/LIONESS_gpu_gpu2_resTable.csv')
cpu1_lioness=pd.read_csv('data/MATLAB/lioness/LIONESS_cpu_cpu1_resTable.csv')
cpu2_lioness=pd.read_csv('data/MATLAB/lioness/LIONESS_cpu_cpu2_resTable.csv')
lionessGPU=pd.read_csv('data/MATLAB/lioness/gpuLIONESS.csv',header=None)
lionessCPU=pd.read_csv('data/MATLAB/lioness/cpuLIONESS.csv',header=None)
np.max(np.max(np.abs(lionessGPU-lionessCPU)))
plt.subplot(122)
plt.plot(lionessGPU.values.flatten(), lionessCPU.values.flatten(), 'o',
         markeredgewidth=2, alpha=0.5)
#plt.text(15, -50, 'Maxdiff = 0.015', fontsize=12)
#plt.text(15, -70, 'Avdiff = $6.5ex10^{5}$', fontsize=12)
plt.xlabel('gpuLIONESS edge weights')
plt.ylabel('LIONESS edge weights')
plt.plot([-60,110],[-60,110])
plt.subplot(121)


plt.bar([1,2,3,4,5],[gpu1_lioness.runtime[0],gpu1_online_lioness.runtime[0],gpu2_lioness.runtime[0],cpu1_lioness.runtime[0],cpu2_lioness.runtime[0]])
plt.ylabel('Run time (s)')
plt.xticks([1,2,3,4,5], ('gpuLIONESS \n(GPU1)','gpuLIONESS on-line \n(GPU1)','gpuLIONESS \n(GPU2)', 'LIONESS \n(CPU1)', 'LIONESS \n(CPU2)'))
