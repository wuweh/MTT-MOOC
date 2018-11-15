% This home assignment is about tracking a single target in clutter and missed
% detections. For simplicity, track initiation and deletion are not covered here. 
% We consider a simple scenario where a single target moves in a 2D region 
% [-1000 1000;-1000 1000] with nearly constant velocity.
clear; close all; clc

dbstop if error

P_D = 0.7;
lambda_c = 30;
%Linear measurement range
% range_c = [-1000 1000;-1000 1000];
%Bearing measurement range
% range_c = [-pi pi];
%Range/bearing measurement range
range_c = [-1000 1000;-pi pi];
sensor_model = modelgen.sensormodel(P_D,lambda_c,range_c);

nbirths = 1;
K = 100;
%CV model initialization
% xstart = [0; 0; 10; 10];
% Pstart = eye(4);

%CT model initialization
xstart = [0; 0; 10; 0; pi/180];
Pstart = diag([1 1 1 1*pi/180 1*pi/180].^2);

ground_truth = modelgen.groundtruth(nbirths,xstart,1,K+1,K);

T = 1;
% CV model parameter
% sigma_q = 5;
% motion_model = motionmodel.cvmodel(T,sigma_q);
% sigma_r = 10;
% meas_model = measmodel.cvmeasmodel(sigma_r);

%CT model parameter
sigmaV = 1;
sigmaOmega = pi/180;
motion_model = motionmodel.ctmodel(T,sigmaV,sigmaOmega);
%Linear measurement model
sigma_r = 5;
% meas_model = measmodel.ctmeasmodel(sigma_r);
%Bearing measurement model
sigma_b = pi/180;
% s = [100;100];
% meas_model = measmodel.bearingmeasmodel(sigma_b, s);
% s1 = [200;400];
% s2 = [100;200];
% meas_model = measmodel.dualbearingmeasmodel(sigma_r, s1, s2);
s = [100;100];
meas_model = measmodel.rangebearingmeasmodel(sigma_r, sigma_b, s);

ifnoisy = 1;
targetdata = targetdatagen(ground_truth,motion_model,ifnoisy);
measdata = measdatagen(targetdata,sensor_model,meas_model);

%Parameters
P_G = 0.999;
wmin = 1e-4;
merging_threshold = 4;
M = 100;

%% NN tracker
tracker = singletargetracker();
tracker = tracker.initiator(P_G,meas_model.d,wmin,merging_threshold,M,xstart,Pstart);
nearestNeighborEstimates = cell(K,1);
for k = 1:K
    tracker = nearestNeighborTracker(tracker, measdata{k}, motion_model, meas_model);
    nearestNeighborEstimates{k} = tracker.x;
end
nearestNeighborRMSE = RMSE(cell2mat(nearestNeighborEstimates'),cell2mat(targetdata.X'));

%% PDA tracker
tracker = tracker.initiator(P_G,meas_model.d,wmin,merging_threshold,M,xstart,Pstart);
probDataAssocEstimates = cell(K,1);
for k = 1:K
    tracker = probDataAssocTracker(tracker, measdata{k}, sensor_model, motion_model, meas_model);
    probDataAssocEstimates{k} = tracker.x;
end
probalisticDataAssocRMSE = RMSE(cell2mat(probDataAssocEstimates'),cell2mat(targetdata.X'));

%% Multi-hypothesis tracker
tracker = tracker.initiator(P_G,meas_model.d,wmin,merging_threshold,M,xstart,Pstart);
multiHypothesesEstimates = cell(K,1);
%Initialize hypothesesWeight and multiHypotheses struct
hypothesesWeight = 1;
multiHypotheses = struct('x',tracker.x,'P',tracker.P);
for k = 1:K
    [tracker, hypothesesWeight, multiHypotheses] = ...
        multiHypothesesTracker(tracker, hypothesesWeight, multiHypotheses, ...
        measdata{k}, sensor_model, motion_model, meas_model);
    multiHypothesesEstimates{k} = tracker.x;
end
multiHypothesesRMSE = RMSE(cell2mat(multiHypothesesEstimates'),cell2mat(targetdata.X'));

%% Ploting
figure
hold on
true_state = cell2mat(targetdata.X');
NN_estimated_state = cell2mat(nearestNeighborEstimates');
PDA_estimated_state = cell2mat(probDataAssocEstimates');
MH_estimated_state = cell2mat(multiHypothesesEstimates');
plot(true_state(1,:), true_state(2,:), '-o','Linewidth', 2)
plot(NN_estimated_state(1,:), NN_estimated_state(2,:), 'Linewidth', 2)
plot(PDA_estimated_state(1,:), PDA_estimated_state(2,:), 'Linewidth', 2)
plot(MH_estimated_state(1,:), MH_estimated_state(2,:), 'Linewidth', 2)
legend('Ground Truth','Nearest Neighbor', 'Probalistic Data Association', 'Multi-hypotheses Data Association', 'Location', 'best')

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