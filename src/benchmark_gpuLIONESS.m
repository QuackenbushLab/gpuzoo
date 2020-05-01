% Comparing gpuLIONESS in GPU1 and LIONESS in CPU2, both in serial runs 
% (meaning that no parallelization over the samples)
addpath(genpath('../netZooM'))
%%
% First, fetch the data from GRANDdb (https://grand.networkmedicine.org)
% In the terminal please type (after installing awscli)
%  cd gpupanda
%  aws s3 cp s3://granddb/gpuPANDA/Hugo_motifCellLine_reduced.txt .
%  aws s3 cp s3://granddb/gpuPANDA/ppi2015_freezeCellLine.txt .
%  aws s3 cp s3://granddb/optPANDA/expression/Hugo_exp1_lcl.txt .
%%
% Experimental setup
model_alias= {'small'};
exp_files  = {'Hugo_exp1_lcl.txt'};
motif_files= {'Hugo_motifCellLine_reduced.txt'};
ppi_files  = {'ppi2015_freezeCellLine.txt'};
precisions = {'single','double'};
similarityMetrics = {'Tfunction'};%took out minkowski
modeProcesses = {'intersection'};
alphas = [0.1];
nExps  = length(exp_files)*length(precisions)*length(similarityMetrics)...
    *length(alphas);
k=0; % benchmark iterator
computing = 'cpu';
hardware  = 'cpu2';
%%
% dry run to compile 
fprintf('Performing dry run to compile libraries \n');
exp_file1   = 'test_data/expression.txt';
motif_file1 = 'test_data/motifTest.txt';
ppi_file1   = 'test_data/ppi.txt';
panda_out  = '';  % optional, leave empty if file output is not required
save_temp  = '';  % optional, leave empty if temp data files will not be needed afterward
lib_path   = '../netZooM';  % path to the folder of PANDA source code
alpha      = 0.1;
save_pairs = 0;%saving in .pairs format
modeProcess= 'intersection';
AgNet = panda_run(lib_path,exp_file1, motif_file1, ppi_file1, panda_out,...
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
                exp_file=exp_files{i};motif_file=motif_files{i};ppi_file=ppi_files{i};
                modeProcess=modeProcesses{i};
                AgNet = panda_run(lib_path,exp_file, motif_file, ppi_file, panda_out,...
                        save_temp, alpha, save_pairs, modeProcess,0.5, 0,...
                        similarityMetric{1}, computing, precision{1}, 0);
                k=k+1;
                [Exp,RegNet,TFCoop,TFNames,GeneNames]=processData(exp_file,motif_file,ppi_file,modeProcess);
                disp('Reading in expression data!');
                [NumConditions, NumGenes] = size(Exp);  % transposed expression
                fprintf('%d genes and %d conditions!\n', NumGenes, NumConditions);
                disp('Reading in motif data!');
                RegNet    = NormalizeNetwork(RegNet);
                disp('Reading in ppi data!');
                TFCoop    = NormalizeNetwork(TFCoop);
                % Small model (652,1000)
                indexes = 1:NumConditions;
                %%
                % run panda and measure runtime
                try
                    saveMemory=1;
                    tic;
                    for i = indexes
                        fprintf('Running LIONESS for sample %d:\n', i);
                        idx = [1:(i-1), (i+1):NumConditions];  % all samples except i

                        disp('Computing coexpresison network:');
                        GeneCoReg = Coexpression(Exp(idx,:));

                        disp('Normalizing Networks:');
                        GeneCoReg = NormalizeNetwork(GeneCoReg);

                        disp('Running PANDA algorithm:');
                        
                        tic;LocNet = PANDA(RegNet, GeneCoReg, TFCoop, alpha, 0.5, similarityMetric{1},...
                        computing, precision{1}, 0, saveMemory);runtime=toc; 
                        PredNet = NumConditions * (AgNet - LocNet) + LocNet;

                        clear idx GeneCoReg LocNet PredNet f; % clean up for next run
                    end
                    runtime=toc;
                catch ME
                    try
                        saveMemory=0;
                        tic;
                        for i = indexes
                            fprintf('Running LIONESS for sample %d:\n', i);
                            idx = [1:(i-1), (i+1):NumConditions];  % all samples except i

                            disp('Computing coexpresison network:');
                            GeneCoReg = Coexpression(Exp(idx,:));

                            disp('Normalizing Networks:');
                            GeneCoReg = NormalizeNetwork(GeneCoReg);

                            disp('Running PANDA algorithm:');
                            LocNet = PANDA(RegNet, GeneCoReg, TFCoop, alpha, 0.5, similarityMetric{1},...
                            computing, precision{1}, 0, saveMemory); 
                            PredNet = NumConditions * (AgNet - LocNet) + LocNet;

                           clear idx GeneCoReg LocNet PredNet f; % clean up for next run
                        end
                        runtime=toc;
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

writetable(resTable,['LIONESS_' computing '_' hardware '_resTable.csv']);
