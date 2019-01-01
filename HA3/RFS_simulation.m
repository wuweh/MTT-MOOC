% A script used to simulate how to draw samples from Poisson, Bernoulli,
% multi-Bernoulli and multi-Bernoulli mixture RFSs with possible uniform,
% Gaussian or Gaussian mixture spatial distribution.
clc;clear;close all

%% uniform distribution
RFS = multiobjectProcess();

%Poisson RFS
%2D 
inputArg1{1} = [-1 1;-1 1];

RFS = spatialDistribution(RFS,inputArg1);
lambda = 2;
[RFS, PoissonInstance] = PoissonRFSs(RFS,lambda);
cardStemPlot(RFS, 0:10);
multiobjectProcess.instance2Dplot(PoissonInstance);

%Bernoulli RFS
%2D
inputArg1{1} = [-1 1;-1 1];

RFS = spatialDistribution(RFS,inputArg1);
r = 0.5;
[RFS, BernoulliInstance] = BernoulliRFSs(RFS,r);
cardStemPlot(RFS, 0:10);
multiobjectProcess.instance2Dplot(BernoulliInstance);

%multi-Bernoulli RFS
%2D
inputArg1{1} = [-1 1;-1 1];
inputArg1{2} = [-1 1;-1 1];

RFS = spatialDistribution(RFS,inputArg1);
M = 2;
r = ones(M,1)*0.5;
[RFS, multiBernoulliInstance] = multiBernoulliRFSs(RFS,r);
cardStemPlot(RFS, 0:10);
multiobjectProcess.instance2Dplot(multiBernoulliInstance);

%multi-Bernoulli mixture RFS
%2D
inputArg1{1} = [-1 1;-1 1];
inputArg1{2} = [-1 1;-1 1];
inputArg1{3} = [-1 1;-1 1];

RFS = spatialDistribution(RFS,inputArg1);
M = [2,3];
p = [0.5 0.5];
r = cell(length(M),1);
r{1} = ones(M(1),1)*0.5;
r{2} = ones(M(2),1)*0.5;
[RFS, multiBernoulliMixtureInstance] = multiBernoulliMixtureRFSs(RFS,r,p);
cardStemPlot(RFS, 0:10);
multiobjectProcess.instance2Dplot(multiBernoulliMixtureInstance);


%% Gaussian distribution
RFS = multiobjectProcess();

%Poisson RFS
%2D
inputArg1{1} = [0;0];
inputArg2{1} = eye(2);

RFS = spatialDistribution(RFS,inputArg1,inputArg2);
lambda = 2;
[RFS, PoissonInstance] = PoissonRFSs(RFS,lambda);
cardStemPlot(RFS, 0:10);
multiobjectProcess.instance2Dplot(PoissonInstance);

%Bernoulli RFS
%2D
inputArg1{1} = [0;0];
inputArg2{1} = eye(2);

RFS = spatialDistribution(RFS,inputArg1,inputArg2);
r = 0.5;
[RFS, BernoulliInstance] = BernoulliRFSs(RFS,r);
cardStemPlot(RFS, 0:10);
multiobjectProcess.instance2Dplot(BernoulliInstance);

%multi-Bernoulli RFS
%2D
inputArg1{1} = [0;0];
inputArg2{1} = eye(2);
inputArg1{2} = [0;0];
inputArg2{2} = eye(2);

RFS = spatialDistribution(RFS,inputArg1,inputArg2);
M = 2;
r = ones(M,1)*0.5;
[RFS, multiBernoulliInstance] = multiBernoulliRFSs(RFS,r);
cardStemPlot(RFS, 0:10);
multiobjectProcess.instance2Dplot(multiBernoulliInstance);

%multi-Bernoulli mixture RFS
%2D
inputArg1{1} = [0;0];
inputArg2{1} = eye(2);
inputArg1{2} = [0;0];
inputArg2{2} = eye(2);
inputArg1{3} = [0;0];
inputArg2{3} = eye(2);

RFS = spatialDistribution(RFS,inputArg1,inputArg2);
M = [2,3];
p = [0.5 0.5];
r = cell(length(M),1);
r{1} = ones(M(1),1)*0.5;
r{2} = ones(M(2),1)*0.5;
[RFS, multiBernoulliMixtureInstance] = multiBernoulliMixtureRFSs(RFS,r,p);
cardStemPlot(RFS, 0:10);
multiobjectProcess.instance2Dplot(multiBernoulliMixtureInstance);


%% Gaussian mixture distribution
%covaraince can be the same across components
RFS = multiobjectProcess();

%Poisson RFS
%2D
inputArg1{1} = [0 0;1 1];
inputArg2{1} = eye(2);      
inputArg3{1} = [0.5 0.5];

RFS = spatialDistribution(RFS,inputArg1,inputArg2,inputArg3);
lambda = 2;
[RFS, PoissonInstance] = PoissonRFSs(RFS,lambda);
cardStemPlot(RFS, 0:10);
multiobjectProcess.instance2Dplot(PoissonInstance);

%Bernoulli RFS
%2D
inputArg1{1} = [0 0;1 1];
inputArg2{1} = eye(2);      
inputArg3{1} = [0.5 0.5];

RFS = spatialDistribution(RFS,inputArg1,inputArg2,inputArg3);
r = 0.5;
[RFS, BernoulliInstance] = BernoulliRFSs(RFS,r);
cardStemPlot(RFS, 0:10);
multiobjectProcess.instance2Dplot(BernoulliInstance);

%multi-Bernoulli RFS
%2D
inputArg1{1} = [0 0;1 1];
inputArg2{1} = eye(2);      
inputArg3{1} = [0.5 0.5];
inputArg1{2} = [0 0;1 1];
inputArg2{2} = eye(2);      
inputArg3{2} = [0.5 0.5];

RFS = spatialDistribution(RFS,inputArg1,inputArg2,inputArg3);
M = 2;
r = ones(M,1)*0.5;
[RFS, multiBernoulliInstance] = multiBernoulliRFSs(RFS,r);
cardStemPlot(RFS, 0:10);
multiobjectProcess.instance2Dplot(multiBernoulliInstance);

%multi-Bernoulli mixture RFS
%2D
inputArg1{1} = [0 0;1 1];
inputArg2{1} = eye(2);      
inputArg3{1} = [0.5 0.5];
inputArg1{2} = [0 0;1 1];
inputArg2{2} = eye(2);      
inputArg3{2} = [0.5 0.5];
inputArg1{3} = [0 0;1 1];
inputArg2{3} = eye(2);      
inputArg3{3} = [0.5 0.5];

RFS = spatialDistribution(RFS,inputArg1,inputArg2,inputArg3);
M = [2,3];
p = [0.5 0.5];
r = cell(length(M),1);
r{1} = ones(M(1),1)*0.5;
r{2} = ones(M(2),1)*0.5;
[RFS, multiBernoulliMixtureInstance] = multiBernoulliMixtureRFSs(RFS,r,p);
cardStemPlot(RFS, 0:10);
multiobjectProcess.instance2Dplot(multiBernoulliMixtureInstance);
