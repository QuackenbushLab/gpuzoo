exp_file   = 'test_data/expression.txt';
motif_file = 'test_data/motifTest.txt';
ppi_file   = 'test_data/ppi.txt';
tic;[AgNet2,cpuImpl] = panda_run('',exp_file, motif_file, ppi_file, '',...
                    '', 0.1, 0, 'union',0.5,0,'Tfunction','gpu','double');toc;
                
tic;[AgNet2,gpuImpl] = panda_run('',exp_file, motif_file, ppi_file, '',...
                    '', 0.1, 0, 'union',0.5,0,'Tfunction','gpu','double');toc;
                
%%       
cpuMem=[cpuImpl.MemAvailableAllArrays];
cpuRes=cpuMem-cpuMem(1);

gpuMem=[gpuImpl.MemAvailableAllArrays];
gpuRes=gpuMem-gpuMem(1);
figure;
hold on
diffMem=cpuRes./gpuRes;
bar(diffMem(2:end))
ylim([0 2])