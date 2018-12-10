% This home assignment is about tracking a single target in clutter and missed
% detections. For simplicity, track initiation and deletion are not covered here.
% We consider a simple scenario where a single target moves in a 2D region
% [-1000 1000;-1000 1000] with nearly constant velocity.
clear; close all; clc

dbstop if error

%Choose object detection probability
P_D = 0.7;
%Choose clutter rate
lambda_c = 60;

%Choose linear or nonlinear scenario
scenario_type = 'nonlinear';

%% Create tracking scenario
switch(scenario_type)
    case 'linear'
        %Creat sensor model
        range_c = [-1000 1000;-1000 1000];
        sensor_model = modelgen.sensormodel(P_D,lambda_c,range_c);
        
        %Creat linear motion model
        T = 1;
        sigma_q = 5;
        motion_model = motionmodel.cvmodel(T,sigma_q);
        
        %Create linear measurement model
        sigma_r = 10;
        meas_model = measmodel.cvmeasmodel(sigma_r);
        
        %Creat ground truth model
        nbirths = 5;
        K = 100;
        tbirth = zeros(nbirths,1);
        tdeath = zeros(nbirths,1);
        
        initial_state = repmat(struct('x',[],'P',eye(motion_model.d)),[1,nbirths]);
        
        initial_state(1).x = [0; 0; 0; -10];        tbirth(1) = 1;   tdeath(1) = K;
        initial_state(2).x = [400; -600; -10; 5];   tbirth(2) = 1;   tdeath(2) = K;
        initial_state(3).x = [-800; -200; 20; -5];  tbirth(3) = 1;   tdeath(3) = K;
        initial_state(4).x = [0; 0; 7.5; -5];       tbirth(4) = 1;   tdeath(4) = K;
        initial_state(5).x = [-200; 800; -3; -15];  tbirth(5) = 1;   tdeath(5) = K;
          
    case 'nonlinear'
        %Create sensor model
        %Range/bearing measurement range
        range_c = [-1000 1000;-pi pi];
        sensor_model = modelgen.sensormodel(P_D,lambda_c,range_c);
        
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
        
        initial_state = repmat(struct('x',[],'P',diag([1 1 1 1*pi/90 1*pi/90].^2)),[1,nbirths]);
        
        initial_state(1).x = [0; 0; 5; 0; pi/180];       tbirth(1) = 1;   tdeath(1) = K;
        initial_state(2).x = [20; 20; -20; 0; pi/90];      tbirth(2) = 1;   tdeath(2) = K;
        initial_state(3).x = [-20; 10; -10; 0; pi/360];    tbirth(3) = 1;   tdeath(3) = K;
        initial_state(4).x = [-10; -10; 8; 0; pi/270];    tbirth(4) = 1;   tdeath(4) = K;

end

%% Generate true object data (noisy or noiseless) and measurement data
ground_truth = modelgen.groundtruth(nbirths,[initial_state.x],tbirth,tdeath,K);
ifnoisy = 0;
objectdata = objectdatagen(ground_truth,motion_model,ifnoisy);
measdata = measdatagen(objectdata,sensor_model,meas_model);

%% Single object tracker parameter setting
P_G = 0.999;            %gating size in percentage
wmin = 1e-3;            %hypothesis pruning threshold
merging_threshold = 4;  %hypothesis merging threshold
M = 20;                %maximum number of hypotheses kept in MHT
density_class_handle = @GaussianDensity;    %density class handle
tracker = n_objectracker();
tracker = tracker.initialize(density_class_handle,P_G,meas_model.d,wmin,merging_threshold,M);

%% NN tracker
GNNestimates = GNNtracker(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);

%% Multi-hypothesis tracker
TOMHTestimates = TOMHT(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);

%% Ploting
figure
hold on

true_state = cell2mat(objectdata.X');
GNN_estimated_state = cell2mat(GNNestimates');
TOMHT_estimated_state = cell2mat(TOMHTestimates');
h1 = plot(true_state(1,:), true_state(2,:), 'bo');
h2 = plot(GNN_estimated_state(1,:), GNN_estimated_state(2,:),'r+');
h3 = plot(TOMHT_estimated_state(1,:), TOMHT_estimated_state(2,:),'g*');

legend([h1 h2 h3],'Ground Truth','GNN','TOMHT', 'Location', 'best')