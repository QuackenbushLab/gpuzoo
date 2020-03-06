%Figure S2 - Differences between PANDA and gpuPANDA
addpath(genpath('../netZooM'))
addpath(genpath('../gibbon'))
% Experimental setup
exp_file   = 'Hugo_exp1_lcl.txt';
motif_file = 'Hugo_motifCellLine_reduced.txt';
ppi_files  = 'ppi2015_freezeCellLine.txt';
similarityMetrics = {'Tfunction','euclidean',...
    'squaredeuclidean','seuclidean','cityblock','chebychev','cosine',...
    'correlation'};
modeProcesses = 'intersection';
alphas = 0.1;
k=0;l=0;m=0; % benchmark iterator
computings = {'cpu','gpu'};
precision  = {'double','single'};
resCell    = cell(length(computings),length(precision),length(similarityMetrics));
%%
fprintf('Starting benchmarks \n');
for computing = computings
    l=l+1;
    for precision = precisions
        m=m+1;
        for similarityMetric = similarityMetrics % loop through distances
                k=k+1;
                [Exp,RegNet,TFCoop,TFNames,GeneNames]=processData(exp_file,motif_file,ppi_file,modeProcess);
                disp('Computing coexpression network:');
                GeneCoReg = Coexpression(Exp);
                %%
                disp('Normalizing Networks:');
                RegNet    = NormalizeNetwork(RegNet);
                GeneCoReg = NormalizeNetwork(GeneCoReg);
                TFCoop    = NormalizeNetwork(TFCoop);
                % run panda and measure runtime
                saveMemory=0;
                tic;AgNet = PANDA(RegNet, GeneCoReg, TFCoop, alpha, 0.5, similarityMetric{1},...
                        computing{1}, precision{1}, 0, saveMemory);runtime=toc;
                resCell{l,m,k}=AgNet;
        end
    end
end