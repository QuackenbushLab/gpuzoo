addpath(genpath('../netZooM'))
addpath(genpath('../gibbon'))
% Experimental setup
model_alias= {'medium','small'};
exp_files  = {'Hugo_exp1_lcl.txt','Hugo_exp1_lcl.txt'};
motif_files= {'Hugo_motifCellLine.txt','Hugo_motifCellLine_reduced.txt'};
ppi_files  = {'ppi2015_freezeCellLine.txt','ppi2015_freezeCellLine.txt'};
precisions = {'single','double'};
similarityMetrics = {'Tfunction','euclidean',...
    'squaredeuclidean','seuclidean','cityblock','chebychev','cosine',...
    'correlation'};%took out minkowski
modeProcesses = {'union','intersection'};
alphas = [0.1,0.2,0.3];
nExps  = length(exp_files)*length(precisions)*length(similarityMetrics)...
    *length(alphas);
k=0; % benchmark iterator
computing = 'cpu';
hardware  = 'cpu2';
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
            'Tfunction', computing, 'single', 0);
%%
%prepare results table
resTable = cell2table(cell(1,5));
resTable.Properties.VariableNames = {'runtime','model','precision','alpha','similarity'};
%%
fprintf('Starting benchmarks \n');
for i=1:length(exp_files)% loop through models
    for precision = precisions % loop through precisions
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
                try
                    saveMemory=0;
                    tic;AgNet = PANDA(RegNet, GeneCoReg, TFCoop, alpha, 0.5, similarityMetric{1},...
                        computing, precision{1}, 0, saveMemory);runtime=toc;
                catch ME
                    try
                        saveMemory=1;
                        tic;AgNet = PANDA(RegNet, GeneCoReg, TFCoop, alpha, 0.5, similarityMetric{1},...
                        computing, precision{1}, 0, saveMemory);runtime=toc; 
                    catch ME
                        resTable.runtime{k}   = NaN;
                        resTable.model{k}     = model_alias{i};
                        resTable.precision{k} = precision{1};
                        resTable.alpha{k}     = alpha;
                        resTable.similarity{k}= similarityMetric{1}; 
                        display('computation failed \n')   
                        continue
                    end
                end
                resTable.runtime{k}   = runtime;
                resTable.model{k}     = model_alias{i};
                resTable.precision{k} = precision{1};
                resTable.alpha{k}     = alpha;
                resTable.similarity{k}= similarityMetric{1};     
            end
        end
    end
end

writetable(resTable,[computing '_' hardware '_resTable.csv']);
