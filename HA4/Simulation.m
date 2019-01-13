clear; close all; clc

dbstop if error

%Choose object detection probability
P_D = 0.98;
%Choose clutter rate
lambda_c = 5;
%Choose object survival probability
P_S = 0.99;

%Choose linear or nonlinear scenario
scenario_type = 'linear';

%% Create tracking scenario
switch(scenario_type)
    case 'linear'
        %Create sensor model
        range_c = [-1000 1000;-1000 1000];
        sensor_model = modelgen.sensormodel(P_S,P_D,lambda_c,range_c);
        
        %Create linear motion model
        T = 1;
        sigma_q = 5;
        motion_model = motionmodel.cvmodel(T,sigma_q);
        
        %Create linear measurement model
        sigma_r = 10;
        meas_model = measmodel.cvmeasmodel(sigma_r);
        
        %Create ground truth model
        nbirths = 12;
        K = 100;
        tbirth = zeros(nbirths,1);
        tdeath = zeros(nbirths,1);
        
        birth_model = repmat(struct('w',0.03,'x',[],'P',100*eye(motion_model.d)),[1,nbirths]);
        
        birth_model(1).x  = [ 0; 0; 0; -10 ];            tbirth(1)  = 1;     tdeath(1)  = 70;
        birth_model(2).x  = [ 400; -600; -10; 5 ];       tbirth(2)  = 1;     tdeath(2)  = K+1;
        birth_model(3).x  = [ -800; -200; 20; -5 ];      tbirth(3)  = 1;     tdeath(3)  = 70;
        birth_model(4).x  = [ 400; -600; -7; -4 ];       tbirth(4)  = 20;    tdeath(4)  = K+1;
        birth_model(5).x  = [ 400; -600; -2.5; 10 ];     tbirth(5)  = 20;    tdeath(5)  = K+1;
        birth_model(6).x  = [ 0; 0; 7.5; -5 ];           tbirth(6)  = 20;    tdeath(6)  = K+1;
        birth_model(7).x  = [ -800; -200; 12; 7 ];       tbirth(7)  = 40;    tdeath(7)  = K+1;
        birth_model(8).x  = [ -200; 800; 15; -10 ];      tbirth(8)  = 40;    tdeath(8)  = K+1;
        birth_model(9).x  = [ -800; -200; 3; 15 ];       tbirth(9)   = 60;   tdeath(9)  = K+1;
        birth_model(10).x  = [ -200; 800; -3; -15 ];     tbirth(10)  = 60;   tdeath(10) = K+1;
        birth_model(11).x  = [ 0; 0; -20; -15 ];         tbirth(11)  = 80;   tdeath(11) = K+1;
        birth_model(12).x  = [ -200; 800; 15; -5 ];      tbirth(12)  = 80;   tdeath(12) = K+1;
        
    case 'nonlinear'
        %Create sensor model
        %Range/bearing measurement range
        range_c = [-1000 1000;-pi pi];
        sensor_model = modelgen.sensormodel(P_S,P_D,lambda_c,range_c);
        
        %Create nonlinear motion model (coordinate turn)
        T = 1;
        sigmaV = 1;
        sigmaOmega = pi/180;
        motion_model = motionmodel.ctmodel(T,sigmaV,sigmaOmega);
        
        %Create nonlinear measurement model (range/bearing)
        sigma_r = 5;
        sigma_b = pi/180;
        s = [300;400];
        meas_model = measmodel.rangebearingmeasmodel(sigma_r, sigma_b, s);
        
        %Creat ground truth model
        nbirths = 4;
        K = 100;
        
        birth_model = repmat(struct('w',0.03,'x',[],'P',100*diag([1 1 1 1*pi/90 1*pi/90].^2)),[1,nbirths]);
        
        birth_model(1).x = [0; 0; 5; 0; pi/180];       tbirth(1) = 1;   tdeath(1) = 50;
        birth_model(2).x = [20; 20; -20; 0; pi/90];    tbirth(2) = 20;  tdeath(2) = 70;
        birth_model(3).x = [-20; 10; -10; 0; pi/360];  tbirth(3) = 40;  tdeath(3) = 90;
        birth_model(4).x = [-10; -10; 8; 0; pi/270];   tbirth(4) = 60;  tdeath(4) = K;

end

%% Generate true object data (noisy or noiseless) and measurement data
ground_truth = modelgen.groundtruth(nbirths,[birth_model.x],tbirth,tdeath,K);
ifnoisy = 0;
objectdata = objectdatagen(ground_truth,motion_model,ifnoisy);
measdata = measdatagen(objectdata,sensor_model,meas_model);

%% Object tracker parameter setting
P_G = 0.999;            %gating size in percentage
wmin = 1e-3;            %hypothesis pruning threshold
merging_threshold = 4;  %hypothesis merging threshold
M = 100;                %maximum number of hypotheses kept
rmin = 1e-3;            %Bernoulli component pruning threshold
r_recycle = 1e-1;       %Bernoulli component recycling threshold
density_class_handle = @GaussianDensity;    %density class handle
tracker = multiobjectracker();
tracker = tracker.initialize(density_class_handle,P_G,meas_model.d,wmin,merging_threshold,M,rmin,r_recycle);

%% GM-PHD filter
% GMPHDestimates = GMPHDtracker(tracker, birth_model, measdata, sensor_model, motion_model, meas_model);

%% PMBM filter
PMBMestimates = PMBMtracker(tracker, birth_model, measdata, sensor_model, motion_model, meas_model);

%% Ploting
%Trajectory plot
figure
hold on

true_state = cell2mat(objectdata.X');
% GMPHD_estimated_state = cell2mat(GMPHDestimates');
PMBM_estimated_state = cell2mat(PMBMestimates');

h1 = plot(true_state(1,:), true_state(2,:), 'bo');
% h2 = plot(GMPHD_estimated_state(1,:), GMPHD_estimated_state(2,:),'r+');
h2 = plot(PMBM_estimated_state(1,:), PMBM_estimated_state(2,:),'r+');

xlabel('x'); ylabel('y')

legend([h1 h2],'Ground Truth','PHD', 'Location', 'best')

%Cardinality plot
figure
grid on
hold on
true_cardinality = objectdata.N;
% GMPHD_estimated_cardinality = cellfun(@(x) size(x,2), GMPHDestimates);
PMBM_estimated_cardinality = cellfun(@(x) size(x,2), PMBMestimates);

h1 = plot(1:length(true_cardinality),true_cardinality,'bo','linewidth',2);
% h2 = plot(1:length(GMPHD_estimated_cardinality),GMPHD_estimated_cardinality,'r+','linewidth',2);
h2 = plot(1:length(PMBM_estimated_cardinality),PMBM_estimated_cardinality,'r+','linewidth',2);

xlabel('Time step')
ylabel('Cardinality')
% legend([h1 h2],'Ground Truth','PHD', 'Location', 'best')
legend([h1 h2],'Ground Truth','PMBM', 'Location', 'best')

%GOSPA plot
c = 100;
p = 1;
gospa = zeros(K,4);
for k = 1:K
    %Evaluate kinematics estimation performance using GOSPA metric
%     gospa(k,:) = GOSPAmetric(objectdata.X{k},GMPHDestimates{k},c,p);
    gospa(k,:) = GOSPAmetric(objectdata.X{k},PMBMestimates{k},c,p);
end

figure
subplot(4,1,1)
plot(1:K,gospa(:,1),'linewidth',2)
ylabel('GOSPA')
subplot(4,1,2)
plot(1:K,gospa(:,2),'linewidth',2)
ylabel('Kinematics')
subplot(4,1,3)
plot(1:K,gospa(:,3),'linewidth',2)
ylabel('# Miss')
subplot(4,1,4)
plot(1:K,gospa(:,4),'linewidth',2)
xlabel('Time Step'); ylabel('# False')
