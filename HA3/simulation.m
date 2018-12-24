clc;clear

RFS = multiobjectProcess();

%Poisson RFS
inputArg1{1} = [-1 1;-1 1];

RFS = spatialDistribution(RFS,inputArg1);
lambda = 10;
PoissonInstance = PoissonRFSs(RFS,lambda);

%Bernoulli RFS
inputArg1{1} = [-1 1;-1 1];

RFS = spatialDistribution(RFS,inputArg1);
r = 0.5;
BernoulliInstance = BernoulliRFSs(RFS,r);

%multi-Bernoulli RFS
inputArg1{1} = [-1 1;-1 1];
inputArg1{2} = [-1 1;-1 1];

RFS = spatialDistribution(RFS,inputArg1);
M = 2;
r = ones(M,1)*0.5;
multiBernoulliInstance = multiBernoulliRFSs(RFS,M,r);

%multi-Bernoulli mixture RFS
inputArg1{1} = [-1 1;-1 1];
inputArg1{2} = [-1 1;-1 1];
inputArg1{3} = [-1 1;-1 1];

RFS = spatialDistribution(RFS,inputArg1);
M = [2,3];
p = [0.5 0.5];
r = cell(length(M),1);
r{1} = ones(M(1),1)*0.5;
r{2} = ones(M(2),1)*0.5;
multiBernoulliMixtureInstance = multiBernoulliMixtureRFSs(RFS,M,r,p);