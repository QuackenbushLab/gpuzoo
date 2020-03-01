exp_file   = 'test_data/expression.txt';
motif_file = 'test_data/motifTest.txt';
ppi_file   = 'test_data/ppi.txt';
modeProcess='union';
[Exp,RegNet,TFCoop,TFNames,GeneNames]=processData(exp_file,motif_file,ppi_file,modeProcess);

disp('Computing coexpression network:');
tic; GeneCoReg = Coexpression(Exp); toc;

disp('Normalizing Networks:');
tic
    RegNet    = NormalizeNetwork(RegNet);
    GeneCoReg = NormalizeNetwork(GeneCoReg);
    TFCoop    = NormalizeNetwork(TFCoop);
toc

disp('Running PANDA algorithm:');
verbose=0;precision='double';computing='cpu';similarityMetric='Tfunction';
respWeight=0.5;alpha=0.1;saveMemory=1;
[AgNet,gpuImpl] = gpuPANDA(RegNet, GeneCoReg, TFCoop, alpha, respWeight,...
    similarityMetric, computing, precision, verbose, saveMemory);

[AgNet,cpuImpl] = PANDA(RegNet, GeneCoReg, TFCoop, alpha, respWeight,...
    similarityMetric, computing, precision, verbose);
                
%%       
cpuMem=[cpuImpl.MemAvailableAllArrays];
cpuRes=cpuMem-cpuMem(1);

gpuMem=[gpuImpl.MemAvailableAllArrays];
gpuRes=gpuMem-gpuMem(1);
figure;
hold on
diffMem=cpuRes./gpuRes;
diffMem(1)=1;
bar(diffMem)
ylim([0 3])