% This home assignment is about tracking a single target in clutter and missed
% detections. For simplicity, track initiation and deletion are not covered here.
% We consider a simple scenario where a single target moves in a 2D region
% [-1000 1000;-1000 1000] with nearly constant velocity.
clear; close all; clc

dbstop if error

%Choose object detection probability
P_D = 0.9;
%Choose clutter rate
lambda_c = 10;

%Choose linear or nonlinear scenario
scenario_type = 'linear';

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
        
        ground_truth = modelgen.groundtruth(nbirths,[initial_state.x],tbirth,tdeath,K);
         
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
        
        ground_truth = modelgen.groundtruth(nbirths,[initial_state.x],tbirth,tdeath,K);
        
end


%% Generate true object data (noisy or noiseless) and measurement data
ifnoisy = 0;
objectdata = objectdatagen(ground_truth,motion_model,ifnoisy);
measdata = measdatagen(objectdata,sensor_model,meas_model);

%% Single object tracker parameter setting
P_G = 0.999;            %gating size in percentage
wmin = 1e-4;            %hypothesis pruning threshold
merging_threshold = 4;  %hypothesis merging threshold
M = 100;                %maximum number of hypotheses kept in MHT
density_class_handle = @GaussianDensity;    %density class handle
tracker = n_objectracker();
tracker = tracker.initialize(density_class_handle,P_G,meas_model.d,wmin,merging_threshold,M);

%% NN tracker
GNNestimates = GNNtracker(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);

%% Multi-hypothesis tracker
% TOMHTestimates = TOMHT(tracker, initial_state, measdata, sensor_model, motion_model, meas_model);

%% Ploting
figure
hold on

true_state = cell2mat(objectdata.X');
GNN_estimated_state = cell2mat(GNNestimates');
plot(true_state(1,:), true_state(2,:), 'bo','Linewidth', 2)
plot(GNN_estimated_state(1,:), GNN_estimated_state(2,:),'r+','Linewidth', 2)

% legend('Ground Truth','Nearest Neighbor', 'Multi-hypotheses Data Association', 'Location', 'best')

%% True target data generation
% In this part, you are going to create the groundtruth data. For this
% purpose consider a 2D (nearly) constant velocity model, (some mathematical
% expressions). The sampling time is $T=1s$, and the acceleration noise is
% $\sigma_a = 5$. At time step t_birth, the target is born with initial
% state (mean) (0, 0, 0, 0). Generate
% (t_death - t_birth + 1) seconds of the state trajectory for this target.
% Generate 100 seconds of the state trajectory for this target and observe
% the position and velocity components.

%% Target-generated measurement generation
% In this part, you are going to create the target-generated measurement.
% Assume that the observations of the target location are collected using an
% imperfect sensor. For this purpose consider a linear measurement model with
% measurement noise $\sigma_r = 10$. The target detection probability is assumed
% to be a constant $P_D = 0.9$. Generate the target-generated measurements
% for each target state.

% Hint:Note that the detection process can be simulated by generating a
% uniform random number u ~ U(0,1) for each measurement time and then by
% checking whether u ≶ PD. If u ≤ PD, this means that the target is detected
% and if not, the target is not detected.

%% Clutter measurement generation
% In this part, you are going to generate the clutter measurements. We
% assume that the number of clutter measurements per time step is Poisson
% distributed with rate $\lambda_F = 10$, and that the spatial distribution
% of clutter is uniform in the square region -1000 <= x,y <= 1000. Generate
% a sets of such clutter for each measurement time. Now, for each time step,
% add the target-generated measurements to the set of clutter.

%% Single target tracker
% Implement single target trackers that use 1) nearest neighbor; 2)
% probabilistic data association; 3)multi-hypotheses solutions
% Perform MC runs and try to measure average RMS position and velocity estimation errors of different trackers.
% Plot the position and velocity estimates of different trackers together
% with the groundtruth data. Compare the results. Try different detection
% probability and clutter rate. Can you observe any difference in the
% performance of different trackers?

% 1. Implement ellipsoidal gating given a single target state and
% measurements. Only keep measurements inside the gate. We choose the
% gating size in percentage $P_G = 0.999$. Actual gating size can be
% calculated as chi2inv(P_G,measurement dimension). Then, calculate the
% effective detection probability and missed detection probablity.

% 2. Nearest neighbor: 1. Choose the closest measurement to the measurement
% prediction in the gate in the sense that, $mathematical expression$ 2.
% Update the target using the chosen measurement

% 3. NN is a hard decision mechanism. Soft version of it is to not make a
% hard decision but use all of the measurements in the gate to the extent
% that they suit the prediction. 1. Calculate the missed detection
% hypothesis. 2. For each measurement inside the gate, calculate the
% measurement update likelihood. 3. Normalise the likelihoods. 4. Gaussian
% mixture reduction.

% 4. Instead of merging of the possible hypotheses into one. We can
% propagate and maintain multiple hypotheses over time. Gating is performed
% for each hypothesis. This results in the multi-hypotheses solution.
% Hypothesis weight can be calculate as w_{k+1} = w_k*l_k.
% However, number of hypotheses will grow exponentially over time.
% In order to reduce computational complexity, we can 1. prune hypotheses
% with very small weights. 2. Only keep M hypotheses with the highest
% weights. 3. Merge similar hypotheses in the sense of small Mahalanobis
% distance