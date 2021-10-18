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
subplot(2,1,2)
histogram(df2.cv)
ax=gca;
ax.FontSize = 16;
xlabel('CV (%)', 'FontSize', 18)
ylabel('Benchmarks', 'FontSize', 18)
title('Coefficient of variation in GPU1 for three trials', 'FontSize', 18)

