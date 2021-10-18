import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os

# define global parameters
plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 18

os.chdir('/Users/mab8354/projects/gpupanda')

def matchVal(cpu1, cpu2, gpu1, gpu2, selInd, cpu1Price, cpu2Price, gpu1Price, gpu2Price, selIndDouble, angles, how='runtime'):
    values = (cpu1['runtime'].iloc[selInd] * cpu1Price) / (gpu1['runtime'].iloc[selInd] * gpu1Price)
    values = values.tolist()
    values += values[:1]
    values2 = (cpu2['runtime'].iloc[selInd] * cpu2Price) / (gpu1['runtime'].iloc[selInd] * gpu1Price)
    values2 = values2.tolist()
    values2 += values2[:1]
    values3 = (cpu1['runtime'].iloc[selInd] * cpu1Price) / (gpu2['runtime'].iloc[selInd] * gpu2Price)
    values3 = values3.tolist()
    values3 += values3[:1]
    values4 = (cpu2['runtime'].iloc[selInd] * cpu2Price) / (gpu2['runtime'].iloc[selInd] * gpu2Price)
    values4 = values4.tolist()
    values4 += values4[:1]
    values5 = (cpu2['runtime'].iloc[selIndDouble] * cpu2Price) / (gpu1['runtime'].iloc[selIndDouble] * gpu1Price)
    values5 = values5.tolist()
    values5 += values5[:1]
    values6 = (cpu1['runtime'].iloc[selIndDouble] * cpu1Price) / (gpu1['runtime'].iloc[selIndDouble] * gpu1Price)
    values6 = values6.tolist()
    values6 += values6[:1]
    if how == 'cost':
        ax2.plot(angles, values, linewidth=1, linestyle='solid', label="group A")
        ax2.fill(angles, values, 'b', alpha=0.1)
        ax2.plot(angles, values2, linewidth=1, linestyle='solid', label="group B")
        ax2.fill(angles, values2, 'r', alpha=0.1)
        ax2.plot(angles, values3, linewidth=1, linestyle='solid', label="group C")
        ax2.fill(angles, values3, 'r', alpha=0.1)
        ax2.plot(angles, values4, linewidth=1, linestyle='solid', label="group D")
        ax2.fill(angles, values4, 'r', alpha=0.1)
        ax2.plot(angles, values5, linewidth=1, linestyle='solid', label="group E")
        ax2.fill(angles, values5, 'r', alpha=0.1)
        ax2.plot(angles, values6, linewidth=1, linestyle='solid', label="group F")
        ax2.fill(angles, values6, 'r', alpha=0.1)
    else:
        ax.plot(angles, values, linewidth=1, linestyle='solid', label="group A")
        ax.fill(angles, values, 'b', alpha=0.1)
        ax.plot(angles, values2, linewidth=1, linestyle='solid', label="group B")
        ax.fill(angles, values2, 'r', alpha=0.1)
        ax.plot(angles, values3, linewidth=1, linestyle='solid', label="group C")
        ax.fill(angles, values3, 'r', alpha=0.1)
        ax.plot(angles, values4, linewidth=1, linestyle='solid', label="group D")
        ax.fill(angles, values4, 'r', alpha=0.1)
        ax.plot(angles, values5, linewidth=1, linestyle='solid', label="group E")
        ax.fill(angles, values5, 'r', alpha=0.1)
        ax.plot(angles, values6, linewidth=1, linestyle='dashed', label="group F")
        ax.fill(angles, values6, 'r', alpha=0.1)
    return values, values2, values3, values4, values5, values6


cpu1 = pd.read_csv('data/MATLAB/panda/cpu1/cpu_cpu1_resTable.csv')
cpu2p1 = pd.read_csv('data/MATLAB/panda/cpu2/cpu_cpu2_resTable_Part1.csv')
cpu2p1=cpu2p1.iloc[:,5:]
cpu2p2 = pd.read_csv('data/MATLAB/panda/cpu2/cpu_cpu2_resTable_Part2.csv')
gpu1 = pd.read_csv('data/MATLAB/panda/gpu1/gpu_gpu1_resTable.csv')
gpu2 = pd.read_csv('data/MATLAB/panda/gpu2/gpu_gpu2_resTable.csv')
gpu3 = pd.read_csv('data/MATLAB/panda/gpu3/gpu_gpu3_resTable.csv')
cpu1Price = 2.304
cpu2Price = 1.808
gpu1Price = 3.902
gpu2Price = 3.06
gpu3Price = 0.9
alphas = [0.1, 0.2, 0.3]
distCell = ['Tfunction', 'Euclidean', 'Squared Euclidean',
            'Standardized Euclidean', 'Cityblock', 'Chebychev', 'Cosine', 'Correlation']

# Merge cpu2 part 1 and part2
cpu2=cpu1.copy()
cpu2['runtime']=np.zeros((144,1))
a=np.where(cpu2['model']=='coding-genes')
cpu2p2.index=cpu2.iloc[a[0][0]:,1:].index
assert(cpu2.iloc[a[0][0]:,1:].equals(cpu2p2.iloc[:,1:]))
cpu2['runtime'].iloc[a[0][0]:]=cpu2p2['runtime']
#now part 1
a=cpu2p1.shape[0]
assert(cpu2.iloc[:a,1:].equals(cpu2p1.iloc[:a,1:]))
cpu2['runtime'].iloc[:a]=cpu2p1['runtime']

# Produce DFs
# 1. Run time Fold change table
fcTbl = cpu1.copy()
fcTbl.columns = ['cpu2/gpu2','model','precision','alpha','similarity']
fcTbl['cpu1/gpu1'] = cpu1.runtime / gpu1.runtime
fcTbl['cpu2/gpu1'] = cpu2.runtime / gpu1.runtime
fcTbl['cpu2/gpu2'] = cpu2.runtime / gpu2.runtime
fcTbl['cpu1/gpu3'] = cpu1.runtime / gpu3.runtime
fcTbl['cpu2/gpu3'] = cpu2.runtime / gpu3.runtime
fcTbl['cpu1/gpu2'] = cpu1.runtime / gpu2.runtime
# Reorder DF
fcTbl = fcTbl[['cpu1/gpu1','cpu2/gpu1','cpu2/gpu2','cpu1/gpu3','cpu2/gpu3','cpu1/gpu2','model','precision','alpha','similarity']]
fcTbl.to_csv('data/MATLAB/panda/fcRuntimePanda.csv')
# 2. Runtime table
runtimeTbl = cpu1.copy()
runtimeTbl.columns = ['cpu1','model','precision','alpha','similarity']
runtimeTbl['cpu2'] = cpu2.runtime
runtimeTbl['gpu1'] = gpu1.runtime
runtimeTbl['gpu2'] = gpu2.runtime
runtimeTbl['gpu3'] = gpu3.runtime
runtimeDf = runtimeTbl.copy()
# Reorder DF
runtimeTbl = runtimeTbl[['cpu1','cpu2','gpu1','gpu2','gpu3','model','precision','alpha','similarity']]
runtimeTbl.to_csv('data/MATLAB/panda/RuntimePanda.csv')
# 3. Cost table
runtimeTbl.cpu1 = runtimeTbl.cpu1/3600 * cpu1Price
runtimeTbl.cpu2 = runtimeTbl.cpu2/3600 * cpu2Price
runtimeTbl.gpu1 = runtimeTbl.gpu1/3600 * gpu1Price
runtimeTbl.gpu2 = runtimeTbl.gpu2/3600 * gpu2Price
runtimeTbl.gpu3 = runtimeTbl.gpu3/3600 * gpu3Price
runtimeTbl.to_csv('data/MATLAB/panda/CostPanda.csv')
costDf = runtimeTbl.copy()
# 4. Cost Fold Change table
fcTbl['cpu1/gpu1'] = fcTbl['cpu1/gpu1'] * cpu1Price/gpu1Price
fcTbl['cpu2/gpu1'] = fcTbl['cpu2/gpu1'] * cpu2Price/gpu1Price
fcTbl['cpu2/gpu2'] = fcTbl['cpu2/gpu2'] * cpu2Price/gpu2Price
fcTbl['cpu1/gpu3'] = fcTbl['cpu1/gpu3'] * cpu1Price/gpu3Price
fcTbl['cpu2/gpu3'] = fcTbl['cpu2/gpu3'] * cpu2Price/gpu3Price
fcTbl['cpu1/gpu2'] = fcTbl['cpu1/gpu2'] * cpu1Price/gpu2Price
fcTbl.to_csv('data/MATLAB/panda/fcCostPanda.csv')

## plot runtime bars for transcript model
# set width of bars
barWidth = 0.25
# set heights of bars
gpu1data = [runtimeDf['gpu1'][runtimeDf['alpha'] == 0.1][runtimeDf['model'] == 'transcript'].iloc[0], runtimeDf['gpu1'][runtimeDf['alpha'] == 0.2][runtimeDf['model'] == 'transcript'].iloc[0], runtimeDf['gpu1'][runtimeDf['alpha'] == 0.3][runtimeDf['model'] == 'transcript'].iloc[0]]
cpu1data = [runtimeDf['cpu1'][runtimeDf['alpha'] == 0.1][runtimeDf['model'] == 'transcript'].iloc[0], runtimeDf['cpu1'][runtimeDf['alpha'] == 0.2][runtimeDf['model'] == 'transcript'].iloc[0], runtimeDf['cpu1'][runtimeDf['alpha'] == 0.3][runtimeDf['model'] == 'transcript'].iloc[0]]
cpu2data = [runtimeDf['cpu2'][runtimeDf['alpha'] == 0.1][runtimeDf['model'] == 'transcript'].iloc[0], runtimeDf['cpu2'][runtimeDf['alpha'] == 0.2][runtimeDf['model'] == 'transcript'].iloc[0], runtimeDf['cpu2'][runtimeDf['alpha'] == 0.3][runtimeDf['model'] == 'transcript'].iloc[0]]
# Set position of bar on X axis
r1 = np.arange(len(gpu1data))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
# Make the plot
plt.errbar(r1, gpu1data, yerr= ,color='#7f6d5f', width=barWidth, edgecolor='white', label='GPU1')
plt.errbar(r2, cpu1data, yerr=4.19 ,color='#557f2d', width=barWidth, edgecolor='white', label='CPU1')
plt.errbar(r3, cpu2data, yerr= ,color='#2d7f5e', width=barWidth, edgecolor='white', label='CPU2')
# Add xticks on the middle of the group bars
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(gpu1data))], ['0.1', '0.2', '0.3'])
plt.savefig('paper/figures/figure2/figure2c.eps', format='eps')

## plot cost bars for transcript model
# set width of bars
barWidth = 0.25
# set heights of bars
gpu1data = [costDf['gpu1'][costDf['alpha'] == 0.1][costDf['model'] == 'transcript'].iloc[0], costDf['gpu1'][costDf['alpha'] == 0.2][costDf['model'] == 'transcript'].iloc[0], costDf['gpu1'][costDf['alpha'] == 0.3][costDf['model'] == 'transcript'].iloc[0]]
cpu1data = [costDf['cpu1'][costDf['alpha'] == 0.1][costDf['model'] == 'transcript'].iloc[0], costDf['cpu1'][costDf['alpha'] == 0.2][costDf['model'] == 'transcript'].iloc[0], costDf['cpu1'][costDf['alpha'] == 0.3][costDf['model'] == 'transcript'].iloc[0]]
cpu2data = [costDf['cpu2'][costDf['alpha'] == 0.1][costDf['model'] == 'transcript'].iloc[0], costDf['cpu2'][costDf['alpha'] == 0.2][costDf['model'] == 'transcript'].iloc[0], costDf['cpu2'][costDf['alpha'] == 0.3][costDf['model'] == 'transcript'].iloc[0]]
# Set position of bar on X axis
r1 = np.arange(len(gpu1data))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
# Make the plot
plt.bar(r1, gpu1data, color='#7f6d5f', width=barWidth, edgecolor='white', label='GPU1')
plt.bar(r2, cpu1data, color='#557f2d', width=barWidth, edgecolor='white', label='CPU1')
plt.bar(r3, cpu2data, color='#2d7f5e', width=barWidth, edgecolor='white', label='CPU2')
# Add xticks on the middle of the group bars
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(gpu1data))], ['0.1', '0.2', '0.3'])
plt.savefig('paper/figures/figure2/figure2d.eps', format='eps')

## plot radar for coding-genes
model = 'coding-genes'
precision = 'single'
k=1
N=len(distCell)
angles = [n / float(N) * 2 * math.pi for n in range(N)]
angles += angles[:1]
for alpha in alphas:
    ax = plt.subplot(2, 3, k, polar=True, )
    selInd= np.where((gpu1['alpha']==alpha) & (gpu1['precision']==precision) & (gpu1['model']==model))
    selIndDouble = np.where((gpu1['alpha'] == alpha) & (gpu1['precision'] == 'double') & (gpu1['model'] == model))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], distCell, color='grey', size=8)
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([ 10,20, 40, 60], [ "10","20","40", "60"], color="grey", size=7)
    plt.ylim(0, 65)
    cpu1Pr,cpu2Pr,gpu1Pr,gpu2Pr = 1,1,1,1
    matchVal(cpu1,cpu2,gpu1,gpu2,selInd,cpu1Pr,cpu2Pr,gpu1Pr,gpu2Pr,selIndDouble, angles)
    # Price
    ax2 = plt.subplot(2, 3, k+3, polar=True, )
    ax2.set_theta_offset(math.pi / 2)
    ax2.set_theta_direction(-1)
    plt.xticks(angles[:-1], distCell, color='grey', size=8)
    # Draw ylabels
    ax2.set_rlabel_position(0)
    plt.yticks([10, 20, 30], ["10", "20", "30"], color="grey", size=7)
    plt.ylim(0, 35)
    matchVal(cpu1,cpu2,gpu1,gpu2,selInd,cpu1Price,cpu2Price,gpu1Price,gpu2Price,selIndDouble, angles, how='cost')
    #increment
    k=k+1

plt.savefig('paper/figures/figure2/figure2b.eps', format='eps')

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


import pandas as pd
import numpy as np

tfs = ['TF_'+str(e) for e in list(range(100))]
genes = ['Gene_'+str(e) for e in list(range(900))]
comb = tfs + genes

sample1 = ['omic1_sample'+str(e) for e in list(range(100))]
sample2 = ['omic2_sample'+str(e) for e in list(range(500))]

X1pd = pd.DataFrame(X1, index=comb, columns=sample1)
X2pd = pd.DataFrame(X2, index=comb, columns=sample2)
X1pd.to_csv('~/Downloads/omic1.csv')
X2pd.to_csv('~/Downloads/omic2.csv')
