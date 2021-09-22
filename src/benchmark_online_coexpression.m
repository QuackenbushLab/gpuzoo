nsamples= [5,500,1000,2000,3000];
resVec = zeros(1,length(nsamples));

for jj =1:length(nsamples)
    x1 =rand(nsamples(jj),1000); %sample-by-gene matrix
    c1 = corr(x1);
    cc1 = cov(x1);
    m1 = mean(x1);
    std1 = std(x1);
    [n,m]=size(x1); % n is number of samples
    allsamples=1:n;

    a=tic;
    for i=1:n
        % 2. correlation without one sample
        samples = setdiff(allsamples,i);
        x2  = x1(samples,:);
        c2  = corr(x2);
    end
    b=toc(a);

    aa=tic;
    for i=1:n
        s1 = x1(i,:); % this is sample i, (it is a row vector)
        % We compute the covariance
        c3=onlineCoexpression(s1,n,m1,std1,cc1);
    end
    bb=toc(aa);
    resVec(jj) = b/bb;
end

figure;
set(gca,'fontname','arial')
plot(resVec)
xticks([1 2 3 4 5])
xticklabels({'5','50','100','200','300'})