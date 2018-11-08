% This home assignment is about tracking a single target in missed
% detection and clutter. We consider a simple scenario where a single target
% moves in a 2D region with nearly constant velocity.
clear; close all; clc

dbstop if error

P_D = 0.7;
lambda_c = 30;
range_c = [-100 100;-100 100];
sensor_model = modelgen.sensormodel(P_D,lambda_c,range_c);
P_G = 0.999;

nbirths = 1;
K = 100;
xstart = [0;1;0;1];
Pstart = eye(4);

ground_truth = modelgen.groundtruth(nbirths,xstart,1,K+1,K);

T = 1;
sigma_q = 0.01;
motion_model = motionmodel.cv2Dmodel(T,sigma_q);
sigma_r = 0.01;
meas_model = measmodel.cv2Dmeasmodel(sigma_r);

ifnoisy = 0;
targetdata = targetdatagen(ground_truth,motion_model,ifnoisy);
measdata = measdatagen(targetdata,sensor_model,meas_model);

%Initiate class
tracker = singletargetracker();

%NN tracker
tracker = tracker.initiator(P_G,meas_model.d,xstart,Pstart);
nearestNeighborEstimates = cell(K,1);
for k = 1:K
    tracker = nearestNeighborAssocTracker(tracker, measdata{k}, motion_model, meas_model);
    nearestNeighborEstimates{k} = tracker.x;
end
nearestNeighborRMSE = RMSE(cell2mat(nearestNeighborEstimates'),cell2mat(targetdata.X'));

%PDA tracker
tracker = tracker.initiator(P_G,meas_model.d,xstart,Pstart);
probDataAssocEstimates = cell(K,1);
for k = 1:K
    tracker = probDataAssocTracker(tracker, measdata{k}, sensor_model, motion_model, meas_model);
    probDataAssocEstimates{k} = tracker.x;
end
probalisticDataAssocRMSE = RMSE(cell2mat(probDataAssocEstimates'),cell2mat(targetdata.X'));

%Multi-hypothesis tracker
tracker = tracker.initiator(P_G,meas_model.d,xstart,Pstart);
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

figure
hold on
true_state = cell2mat(targetdata.X');
NN_estimated_state = cell2mat(nearestNeighborEstimates');
PDA_estimated_state = cell2mat(probDataAssocEstimates');
MH_estimated_state = cell2mat(multiHypothesesEstimates');
plot(true_state(1,:), true_state(3,:), '-o','Linewidth', 2)
plot(NN_estimated_state(1,:), NN_estimated_state(3,:), 'Linewidth', 2)
plot(PDA_estimated_state(1,:), PDA_estimated_state(3,:), 'Linewidth', 2)
plot(MH_estimated_state(1,:), MH_estimated_state(3,:), 'Linewidth', 2)
legend('Ground Truth','Nearest Neighbor', 'Probalistic Data Association', 'Multi-hypotheses Data Association', 'Location', 'best')
%% True target data generation
% In this part, you are going to create the groundtruth data. For this
% purpose consider a 2D (nearly) constant velocity model, (some mathematical
% expressions). The sampling time is $T=1s$, and the acceleration noise is
% $\sigma_a = 0.1$. The target is guaranteed to be born and has expected initial
% state (mean) (0, 0, 0, 0) and uncertainty (covariance) eye(4). Generate 
% (t_death - t_birth + 1) seconds of the state trajectory for this target.

%% Target-generated measurement generation
% In this part, you are going to create the target-generated measurement.
% For this purpose consider a linear measurement model with measurement noise
% $\sigma_r = 0.5$. The target detection probability is assumed to be constant 
% $P_D = 0.9$. Generate the target-generated measurements for each target state.

%% Clutter measurement generation
% In this part, you are going to generate the clutter measurements. We
% assume that the number of clutter measurements per time step is Poisson
% distributed with rate $\lambda_F = 10$, and that the spatial distribution
% of clutter is uniform in the square region -200 <= x,y <= 200. Generate 
% 100 sets of such clutter representing the clutter sets we are going to 
% receive at integer time instants in [0, 99s]. Now add the target-generated
% measurements to the 100 sets of clutter by selecting t_birth as the
% starting time
