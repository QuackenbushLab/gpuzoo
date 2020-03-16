%Figure S1 - Differences between single and double
addpath(genpath('../netZooM'))
addpath(genpath('../gibbon'))
% Experimental setup
model_alias= {'large','medium','small'};
exp_files  = {'THP-1.tsv','Hugo_exp1_lcl.txt','Hugo_exp1_lcl.txt'};
motif_files= {'motif_complete_reduced.txt','Hugo_motifCellLine.txt','Hugo_motifCellLine_reduced.txt'};
ppi_files  = {'ppi_complete.txt','ppi2015_freezeCellLine.txt','ppi2015_freezeCellLine.txt'};
similarityMetrics = {'Tfunction','euclidean',...
    'squaredeuclidean','seuclidean','cityblock','chebychev','cosine',...
    'correlation'};%took out minkowski
modeProcesses = {'union','union','intersection'};
alphas = [0.1,0.2,0.3];
nExps  = length(exp_files)*length(similarityMetrics)...
    *length(alphas);
k=0; % benchmark iterator
computing = 'cpu';
hardware  = 'cpu1';
%prepare results table
resTable = cell2table(cell(1,4));
resTable.Properties.VariableNames = {'logDiff','model','alpha','similarity'};
%%
fprintf('Starting benchmarks \n');
for i=1:length(exp_files)% loop through models
    for alpha = alphas % loop through alphas
        for similarityMetric = similarityMetrics % loop through distances
            k=k+1;
            % LCL 
            exp_file=exp_files{i};motif_file=motif_files{i};ppi_file=ppi_files{i};
            modeProcess=modeProcesses{i};%similarityMetric=similarityMetric{1};
            %precision=precision{1};
            % Large model (1603,43698)
           [Exp,RegNet,TFCoop,TFNames,GeneNames]=processData(exp_file,motif_file,ppi_file,modeProcess);
            disp('Computing coexpression network:');
            GeneCoReg = Coexpression(Exp);
            % Medium model (652,27149)
            % Small model (652,1000)
            %%
            disp('Normalizing Networks:');
            RegNet    = NormalizeNetwork(RegNet);
            GeneCoReg = NormalizeNetwork(GeneCoReg);
            TFCoop    = NormalizeNetwork(TFCoop);
            % run panda and measure runtime
            saveMemory=0;
            AgNet1 = PANDA(RegNet, GeneCoReg, TFCoop, alpha, 0.5, similarityMetric{1},...
                        computing, 'single', 0, saveMemory);
            AgNet2 = PANDA(RegNet, GeneCoReg, TFCoop, alpha, 0.5, similarityMetric{1},...
                        computing, 'double', 0, saveMemory);
            diffNet=max(abs(double(AgNet1(:))-double(AgNet2(:))));
            resTable.logDiff{k}   = -log10(diffNet);
            resTable.model{k}     = model_alias{i};
            resTable.alpha{k}     = alpha;
            resTable.similarity{k}= similarityMetric{1};     
        end
    end
end
writetable(resTable,[computing '_' hardware '_resTable_logDiff.csv']);
%plot
tbl=readtable('cpu_cpu1_resTable_logDiff.csv');
figure;
subplot(1,3,1)
bar(reshape(tbl.logDiff(1:8*3),[8 3]))
xticklabels(tbl.similarity(1:8))
subplot(1,3,2)
bar(reshape(tbl.logDiff((8*3)+1:8*6),[8 3]))
xticklabels(tbl.similarity(1:8))
subplot(1,3,3)
bar(reshape(tbl.logDiff((8*6)+1:end),[8 3]))
xticklabels(tbl.similarity(1:8))
%%
%Figure S2 - Differences between PANDA and gpuPANDA
addpath(genpath('../netZooM'))
addpath(genpath('../gibbon'))
addpath(genpath('../gpupanda'))
% Experimental setup
exp_file   = 'Hugo_exp1_lcl.txt';
motif_file = 'Hugo_motifCellLine_reduced.txt';
ppi_file   = 'ppi2015_freezeCellLine.txt';
similarityMetrics = {'Tfunction','euclidean',...
    'squaredeuclidean','seuclidean','cityblock','chebychev','cosine',...
    'correlation'};
modeProcess = 'intersection';
alpha = 0.1;
computings = {'cpu','gpu'};
precisions  = {'double','single'};
resCell    = cell(length(computings),length(precision),length(similarityMetrics));
%%
fprintf('Starting benchmarks \n');
l=0; % benchmark iterator
for computing = computings
    l=l+1;m=0;
    for precision = precisions
        m=m+1;k=0;
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
%save
save('resCell.mat','resCell')
%plot results
m=0;k=0;t=0;
vecTols=[];
figure;
for precision = precisions
    m=m+1;k=0;
    for similarityMetric = similarityMetrics % loop through distances
        k=k+1;t=t+1;
        subplot(length(precisions),length(similarityMetrics),t)
        a=resCell{1,m,k};
        b=resCell{2,m,k};
        plot(a(:),b(:),'o')
        max(abs(a(:)-b(:)))
        hold on;
        plot([min(a(:)) max(a(:))],[min(a(:)) max(a(:))],'Linewidth',2)
    end
end
set(gca,'fontname','arial') 
%%
%Figure S2 - Runtime difference between LIONESS and gpuLIONESS
addpath(genpath('../netZooM'))
addpath(genpath('../gibbon'))
addpath(genpath('../gpupanda'))