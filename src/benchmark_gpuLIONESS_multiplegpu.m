% Comparing gpuLIONESS in GPU1 and LIONESS in CPU2, both in serial runs 
% (meaning that no parallelization over the samples)
addpath(genpath('../../netZooM'))
addpath(genpath('../../gpupanda'))
%%
% First, fetch the data from GRANDdb (https://grand.networkmedicine.org)
% In the terminal please type (after installing awscli)
%  cd gpupanda
%  aws s3 cp s3://granddb/gpuPANDA/Hugo_motifCellLine_reduced.txt .
%  aws s3 cp s3://granddb/gpuPANDA/ppi2015_freezeCellLine.txt .
%  aws s3 cp s3://granddb/optPANDA/expression/Hugo_exp1_lcl.txt .
%%
% Experimental setup
model_alias= {'coding-genes'};
exp_files  = {'Hugo_exp1_lcl.txt'};
motif_files= {'Hugo_motifCellLine.txt'};
ppi_files  = {'ppi2015_freezeCellLine.txt'};
precisions = {'single'};
similarityMetrics = {'Tfunction'};
modeProcesses = {'union'};
alphas = [0.1];
nExps  = length(exp_files)*length(precisions)*length(similarityMetrics)...
    *length(alphas);
k=0; % benchmark iterator
computing = 'gpu';
hardware  = 'gpu2';
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
% define parameters for online coexpression
exp_file=exp_files{1};motif_file=motif_files{1};ppi_file=ppi_files{1};
[Exploop,RegNet,TFCoop,TFNames,GeneNames]=processData(exp_file,motif_file,ppi_file,modeProcess);
computeExpression='serial';
[n, NumGenes] = size(Exploop);
mi=mean(Exploop,1);
stdd=std(Exploop,1);
covv=cov(Exploop);
%%
numGPUs = gpuDeviceCount();
parpool(numGPUs);
fprintf(['Number of GPUs is ',num2str(numGPUs),'\n'])
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
                [Exploop,RegNet,TFCoop,TFNames,GeneNames]=processData(exp_file,motif_file,ppi_file,modeProcess);

                disp('Reading in expression data!');
                [NumConditions, NumGenes] = size(Exploop);  % transposed expression
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
                    t0=tic;saveMemory=0;
                    parfor jj = indexes
                        fprintf('Running LIONESS for sample %d:\n', jj);
                        idx = [1:(jj-1), (jj+1):NumConditions];  % all samples except i

                        disp('Computing coexpression network:');
                        if isequal(computing,'gpu')
                            Exp=gpuArray(Exploop);
                            if isequal(computeExpression,'online')
                                si=Exp(idx,:);
                                GeneCoReg=onlineCoexpression(si,n,mi,stdd,covv);
                            else
                                GeneCoReg = Coexpression(Exp(idx,:));
                            end
                            if isequal(hardware,'gpu2')
                                GeneCoReg = gather(GeneCoReg);
                            end
                        else
                            GeneCoReg = Coexpression(Exploop(idx,:));
                        end

                        disp('Normalizing Networks:');
                        GeneCoReg = NormalizeNetwork(GeneCoReg);
                        if isequal(hardware,'gpu1')
                            GeneCoReg = gather(GeneCoReg);
                        end

                        disp('Running PANDA algorithm:');
                        
                        LocNet = PANDA(RegNet, GeneCoReg, TFCoop, alpha, 0.5, similarityMetric{1},...
                        computing, precision{1}, 0, saveMemory);
                        PredNet = NumConditions * (AgNet - LocNet) + LocNet;

                        clear idx GeneCoReg LocNet PredNet f; % clean up for next run
                    end
                    runtime=toc(t0); 
                catch ME
                    fprintf('device error \n');
                    try
                        t0=tic;saveMemory=0;
                        for jj = indexes
                            fprintf('Running LIONESS for sample %d:\n', jj);
                            idx = [1:(jj-1), (jj+1):NumConditions];  % all samples except i

                            disp('Computing coexpression network:');
                            if isequal(computing,'gpu')
                                Exp=gpuArray(Exploop);
                                GeneCoReg = Coexpression(Exp(idx,:));
                                GeneCoReg = gather(GeneCoReg);
                            else
                                GeneCoReg = Coexpression(Exploop(idx,:));
                            end

                            disp('Normalizing Networks:');
                            GeneCoReg = NormalizeNetwork(GeneCoReg);

                            disp('Running PANDA algorithm:');
                            LocNet = PANDA(RegNet, GeneCoReg, TFCoop, alpha, 0.5, similarityMetric{1},...
                            computing, precision{1}, 0, saveMemory); 
                            PredNet = NumConditions * (AgNet - LocNet) + LocNet;

                           clear idx GeneCoReg LocNet PredNet f; % clean up for next run
                        end
                        runtime=toc(t0); 
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
