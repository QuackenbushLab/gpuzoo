addpath(genpath('../netZooM'))
addpath(genpath('../gibbon'))
% Experimental setup
model_alias= {'large','medium','small'};
exp_files  = {'THP-1.tsv','Hugo_exp1_lcl.txt','Hugo_exp1_lcl.txt'};
motif_files= {'motif_complete_reduced.txt','Hugo_motifCellLine.txt','Hugo_motifCellLine_reduced.txt'};
ppi_files  = {'ppi_complete.txt','ppi2015_freezeCellLine.txt','ppi2015_freezeCellLine.txt'};
precisions = {'single','double'};
similarityMetrics = {'Tfunction','euclidean',...
    'squaredeuclidean','seuclidean','cityblock','minkowski','chebychev','cosine',...
    'correlation'};
modeProcesses = {'union','union','intersection'};
alphas = [0.05,0.1,0.2];
nExps  = length(exp_files)*length(precisions)*length(similarityMetrics)...
    *length(alphas);
k=0; % benchmark iterator
computing = 'cpu';
hardware  = 'cpu3';
%%
% dry run to compile 
fprintf('Performing dry run to compile libraries \n');
exp_file   = 'test_data/expression.txt';
motif_file = 'test_data/motifTest.txt';
ppi_file   = 'test_data/ppi.txt';
panda_out  = '';  % optional, leave empty if file output is not required
save_temp  = '';  % optional, leave empty if temp data files will not be needed afterward
lib_path   = '../netZooM';  % path to the folder of PANDA source code
alpha      = 0.1;
save_pairs = 0;%saving in .pairs format
modeProcess= 'intersection';
AgNet = panda_run(lib_path,exp_file, motif_file, ppi_file, panda_out,...
            save_temp, alpha, save_pairs, 'intersection',0.5, 0,...
            'Tfunction', computing, 'signle', 0);
%%
%prepare results table
resTable = table({'runtime','model','precision','alpha','similarity'});
%%
fprintf('Starting benchmarks \n');
for i=1:length(exp_files)% loop through models
    for precision = precisions % loop through precisions
        for alpha = alphas % loop through alphas
            for similarityMetric = similarityMetrics % loop through distances
                k=k+1;
                % LCL 
                exp_file=exp_files{i};motif_file=motif_files{i};ppi_file=ppi_files{i};
                modeProcess=modeProcesses{i};
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
                tic;AgNet = PANDA(RegNet, GeneCoReg, TFCoop, alpha, 0.5, similarityMetric,...
                    computing, precision, 0);runtime=toc;
                resTable.runtime{k}   = runtime;
                resTable.model{k}     = model_alias{i};
                resTable.precision{k} = precision;
                resTable.alpha{k}     = alpha;
                resTable.similarity{k}= similarityMetric;     
            end
        end
    end
end

writetable(resTable,[computing '_' hardware '_resTable.mat']);
