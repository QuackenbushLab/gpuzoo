df = readtable('cpu1/cpu_cpu1_std_resTable.csv');
df.cv = (df.stdruntime./df.runtime)*100;

subplot(2,1,1)
histogram(df.cv)
ax=gca;
ax.FontSize = 16;
xlabel('CV (%)', 'FontSize', 18)
ylabel('Benchmarks', 'FontSize', 18)
title('Coefficient of variation in CPU1 for three trials', 'FontSize', 18)
df2 = readtable('gpu1/gpu_gpu1_std_resTable.csv');
df2.cv = (df2.stdruntime./df2.runtime)*100;
df2.cv(isnan(df2.cv)) = 0; %replace nan by 0
xlim([0 9]);
subplot(2,1,2)
histogram(df2.cv)
ax=gca;
ax.FontSize = 16;
xlabel('CV (%)', 'FontSize', 18)
ylabel('Benchmarks', 'FontSize', 18)
title('Coefficient of variation in GPU1 for three trials', 'FontSize', 18)
xlim([0 9]);

df3 = readtable('cpu2/cpu_cpu2_std_resTable.csv');
df3.cv = (df3.stdruntime./df3.runtime)*100;
mean(df3.cv)

a=[0.86 0.02 0.76];
mean(a)

b=[0.473 0.155 0.26];
mean(b)