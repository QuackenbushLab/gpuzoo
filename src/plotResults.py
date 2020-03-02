import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os

os.chdir('../../gpupanda')
cpu1 = pd.read_csv('data/results/cpu_cpu1_resTable.csv')
cpu2p1 = pd.read_csv('data/results/cpu2/cpu_cpu2_resTable_Part1.csv')
cpu2p1=cpu2p1.iloc[:,5:]
cpu2p2 = pd.read_csv('data/results/cpu2/cpu_cpu2_resTable_Part2.csv')
gpu1 = pd.read_csv('data/results/gpu_gpu1_resTable.csv')
gpu2 = pd.read_csv('data/results/gpu_gpu2_resTable.csv')
cpu1Price = 2.304
cpu2Price = 1.808
gpu1Price = 3.06
gpu2Price = 0.9
alphas = [0.1, 0.2, 0.3]
model = 'medium'
precision = 'single'
distCell = ['Tfunction', 'Euclidean', 'Squared Euclidean',
            'Standardized Euclidean', 'Cityblock', 'Chebychev', 'Cosine', 'Correlation']

# Merge cpu2 part 1 and part2
cpu2=cpu1.copy()
cpu2['runtime']=np.zeros((144,1))
a=np.where(cpu2['model']=='medium')
cpu2p2.index=cpu2.iloc[a[0][0]:,1:].index
assert(cpu2.iloc[a[0][0]:,1:].equals(cpu2p2.iloc[:,1:]))
cpu2['runtime'].iloc[a[0][0]:]=cpu2p2['runtime']
#now part 1
a=cpu2p1.shape[0]
assert(cpu2.iloc[:a,1:].equals(cpu2p1.iloc[:a,1:]))
cpu2['runtime'].iloc[:a]=cpu2p1['runtime']

k=1
N=len(distCell)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]
for alpha in alphas:
    ax = plt.subplot(2, 3, k, polar=True, )
    selInd= np.where((gpu1['alpha']==alpha) & (gpu1['precision']==precision) & (gpu1['model']==model))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], distCell, color='grey', size=8)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([ 5,10, 20, 35], [ "5","10", "20","35"], color="grey", size=7)
    plt.ylim(0, 35)
    values=cpu1['runtime'].iloc[selInd]/gpu1['runtime'].iloc[selInd]
    values=values.tolist()
    values += values[:1]
    values2=cpu2['runtime'].iloc[selInd]/gpu1['runtime'].iloc[selInd]
    values2=values2.tolist()
    values2 += values2[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="group A")
    ax.fill(angles, values, 'b', alpha=0.1)
    ax.plot(angles, values2, linewidth=1, linestyle='solid', label="group B")
    ax.fill(angles, values2, 'r', alpha=0.1)
    # Price
    ax2 = plt.subplot(2, 3, k+3, polar=True, )
    ax2.set_theta_offset(math.pi / 2)
    ax2.set_theta_direction(-1)
    plt.xticks(angles[:-1], distCell, color='grey', size=8)
    # Draw ylabels
    ax2.set_rlabel_position(0)
    plt.yticks([5, 10, 15, 20], ["5", "10", "15", "20"], color="grey", size=7)
    plt.ylim(0, 20)
    values=(cpu1['runtime'].iloc[selInd]*cpu1Price)/(gpu1['runtime'].iloc[selInd]*gpu1Price)
    values=values.tolist()
    values += values[:1]
    values2=(cpu2['runtime'].iloc[selInd]*cpu2Price)/(gpu1['runtime'].iloc[selInd]*gpu1Price)
    values2=values2.tolist()
    values2 += values2[:1]
    ax2.plot(angles, values, linewidth=1, linestyle='solid', label="group A")
    ax2.fill(angles, values, 'b', alpha=0.1)
    ax2.plot(angles, values2, linewidth=1, linestyle='solid', label="group B")
    ax2.fill(angles, values2, 'r', alpha=0.1)
    #increment
    k=k+1

plt.savefig('paper/figures/figure1/figure1b.eps', format='eps')

# plot optimality of cost
selIndOpt= np.where((gpu1['precision']==precision) & (gpu1['model']==model))
X=cpu1['runtime'].iloc[selIndOpt]/gpu1['runtime'].iloc[selIndOpt]
Y=(cpu1['runtime'].iloc[selIndOpt]*cpu1Price)/(gpu1['runtime'].iloc[selIndOpt]*gpu1Price)
ax = plt.subplot(2, 1, 1,)
ax.plot(X,Y,marker='X')
ax.plot([1,15],[1,15])
axes = plt.gca()
axes.set_xlim([0,15])
axes.set_ylim([0,15])
axes.set_aspect('equal', adjustable='box')
ax2 = plt.subplot(2, 1, 2,)
X=cpu2['runtime'].iloc[selIndOpt]/gpu1['runtime'].iloc[selIndOpt]
Y=(cpu2['runtime'].iloc[selIndOpt]*cpu2Price)/(gpu1['runtime'].iloc[selIndOpt]*gpu1Price)
ax2.plot(X,Y,marker='X')
ax2.plot([1,40],[1,40])
axes = plt.gca()
axes.set_xlim([0,40])
axes.set_ylim([0,40])
axes.set_aspect('equal', adjustable='box')
plt.savefig('paper/figures/figure1/figure1c.eps', format='eps')


# figure 2 for small models
model = 'small'
precisions = ['single','double']
styleVec=['solid','--']
k=1
N=len(distCell)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]
for alpha in alphas:
    ax = plt.subplot(2, 3, k, polar=True, )
    colorIter=0
    for precision in precisions:
        selInd= np.where((gpu1['alpha']==alpha) & (gpu1['precision']==precision) & (gpu1['model']==model))
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], distCell, color='grey', size=8)
        # Draw ylabels
        ax.set_rlabel_position(0)
        values=cpu1['runtime'].iloc[selInd]/gpu1['runtime'].iloc[selInd]
        values=values.tolist()
        values += values[:1]
        values2=cpu1['runtime'].iloc[selInd]/gpu2['runtime'].iloc[selInd]
        values2=values2.tolist()
        values2 += values2[:1]
        values3=cpu2['runtime'].iloc[selInd]/gpu1['runtime'].iloc[selInd]
        values3=values3.tolist()
        values3 += values3[:1]
        values4=cpu2['runtime'].iloc[selInd]/gpu2['runtime'].iloc[selInd]
        values4=values4.tolist()
        values4 += values4[:1]
        ax.plot(angles, values, linewidth=1, linestyle=styleVec[colorIter], label="group A",color='r')
        ax.plot(angles, values2, linewidth=1, linestyle=styleVec[colorIter], label="group B",color='b')
        ax.plot(angles, values3, linewidth=1, linestyle=styleVec[colorIter], label="group B",color='g')
        ax.plot(angles, values4, linewidth=1, linestyle=styleVec[colorIter], label="group B",color='k')
        colorIter=colorIter+1
    # Price
    ax2 = plt.subplot(2, 3, k+3, polar=True, )
    ax2.set_theta_offset(math.pi / 2)
    ax2.set_theta_direction(-1)
    plt.xticks(angles[:-1], distCell, color='grey', size=8)
    # Draw ylabels
    ax2.set_rlabel_position(0)
    colorIter=0
    for precision in precisions:
        selInd = np.where((gpu1['alpha'] == alpha) & (gpu1['precision'] == precision) & (gpu1['model'] == model))
        ax2.set_theta_offset(math.pi / 2)
        ax2.set_theta_direction(-1)
        plt.xticks(angles[:-1], distCell, color='grey', size=8)
        # Draw ylabels
        ax2.set_rlabel_position(0)
        # plt.yticks([ 5,10, 20, 35], [ "5","10", "20","35"], color="grey", size=7)
        # plt.ylim(0, 35)
        values=(cpu1['runtime'].iloc[selInd]*cpu1Price)/(gpu1['runtime'].iloc[selInd]*gpu1Price)
        values=values.tolist()
        values += values[:1]
        values2=(cpu1['runtime'].iloc[selInd]*cpu1Price)/(gpu2['runtime'].iloc[selInd]*gpu2Price)
        values2=values2.tolist()
        values2 += values2[:1]
        values3=(cpu2['runtime'].iloc[selInd]*cpu2Price)/(gpu1['runtime'].iloc[selInd]*gpu1Price)
        values3=values3.tolist()
        values3 += values3[:1]
        values4=(cpu2['runtime'].iloc[selInd]*cpu2Price)/(gpu2['runtime'].iloc[selInd]*gpu2Price)
        values4=values4.tolist()
        values4 += values4[:1]
        ax2.plot(angles, values, linewidth=1, linestyle=styleVec[colorIter], label="group A",color='r')
        # ax.fill(angles, values, 'b', alpha=0.1)
        ax2.plot(angles, values2, linewidth=1, linestyle=styleVec[colorIter], label="group B",color='b')
        # ax.fill(angles, values2, 'r', alpha=0.1)
        ax2.plot(angles, values3, linewidth=1, linestyle=styleVec[colorIter], label="group B",color='g')
        # ax.fill(angles, values3, 'r', alpha=0.1)
        ax2.plot(angles, values4, linewidth=1, linestyle=styleVec[colorIter], label="group B",color='k')
        # ax.fill(angles, values4, 'r', alpha=0.1)
        colorIter=colorIter+1
    #increment
    k=k+1

plt.savefig('paper/figures/figure2/figure2a.eps', format='eps')


# plot optimality of cost for small models
model='small'
selIndOpt= np.where( (gpu1['model']==model))
X=cpu1['runtime'].iloc[selIndOpt]/gpu1['runtime'].iloc[selIndOpt]
Y=(cpu1['runtime'].iloc[selIndOpt]*cpu1Price)/(gpu1['runtime'].iloc[selIndOpt]*gpu1Price)
ax = plt.subplot(1, 4, 1,)
ax.plot(X,Y,marker='X')
ax.plot([1,10],[1,10])
axes = plt.gca()
axes.set_xlim([0,10])
axes.set_ylim([0,10])
axes.set_aspect('equal', adjustable='box')
ax2 = plt.subplot(1, 4, 2,)
X=cpu2['runtime'].iloc[selIndOpt]/gpu1['runtime'].iloc[selIndOpt]
Y=(cpu2['runtime'].iloc[selIndOpt]*cpu2Price)/(gpu1['runtime'].iloc[selIndOpt]*gpu1Price)
ax2.plot(X,Y,marker='X')
ax2.plot([1,15],[1,15])
axes = plt.gca()
axes.set_xlim([0,15])
axes.set_ylim([0,15])
axes.set_aspect('equal', adjustable='box')
X=cpu1['runtime'].iloc[selIndOpt]/gpu2['runtime'].iloc[selIndOpt]
Y=(cpu1['runtime'].iloc[selIndOpt]*cpu1Price)/(gpu2['runtime'].iloc[selIndOpt]*gpu2Price)
ax = plt.subplot(1, 4, 3,)
ax.plot(X,Y,marker='X')
ax.plot([1,15],[1,15])
axes = plt.gca()
axes.set_xlim([0,15])
axes.set_ylim([0,15])
axes.set_aspect('equal', adjustable='box')
ax2 = plt.subplot(1, 4, 4,)
X=cpu2['runtime'].iloc[selIndOpt]/gpu2['runtime'].iloc[selIndOpt]
Y=(cpu2['runtime'].iloc[selIndOpt]*cpu2Price)/(gpu2['runtime'].iloc[selIndOpt]*gpu2Price)
ax2.plot(X,Y,marker='X')
ax2.plot([1,20],[1,20])
axes = plt.gca()
axes.set_xlim([0,20])
axes.set_ylim([0,20])
axes.set_aspect('equal', adjustable='box')

plt.savefig('paper/figures/figure2/figure2b.eps', format='eps')