% This home assignment is about tracking a single target in missed
% detection and clutter. We consider a simple scenario where a single target
% moves in a 2D region with nearly constant velocity.
clear; close all; clc

dbstop if error

P_D = 0.9;
lambda_c = 10;
range_c = [-100 100;-100 100];
sensor_model = modelgen.sensormodel(P_D,lambda_c,range_c);

nbirths = 1;
K = 100;
xstart = [0;0;0;0];

ground_truth = modelgen.groundtruth(nbirths,xstart,1,K+1,K);

T = 1;
sigma_q = 0.1;
motion_model = motionmodel.cv2Dmodel(T,sigma_q);
sigma_r = 0.1;
meas_model = measmodel.cv2Dmeasmodel(sigma_r);

targetdata = targetdatagen(ground_truth,motion_model);
measdata = measdatagen(targetdata,sensor_model,meas_model);

tracker = singletargetracker();
P_G = 0.999;

Pstart = 0.1*eye(4);
tracker = tracker.initiator(P_G,meas_model.d,xstart,Pstart);
nearestNeighborEstimates = cell(K,1);
squareError = zeros(K,motion_model.d);
for k = 1:K
    tracker = nearestNeighborAssocTracker(tracker, measdata.Z{k}, motion_model, meas_model);
    squareError(k,:) = (tracker.x - targetdata.X{k}).^2;
    nearestNeighborEstimates{k}.x = tracker.x;
    nearestNeighborEstimates{k}.P = tracker.P;
end
nearestNeighborRMSE = sqrt(mean(squareError));

tracker = tracker.initiator(P_G,meas_model.d,xstart,Pstart);
probDataAssocEstimates = cell(K,1);
squareError = zeros(K,motion_model.d);
for k = 1:K
    tracker = probDataAssocTracker(tracker, measdata.Z{k}, sensor_model, motion_model, meas_model);
    squareError(k,:) = (tracker.x - targetdata.X{k}).^2;
    probDataAssocEstimates{k}.x = tracker.x;
    probDataAssocEstimates{k}.P = tracker.P;
end
probDataAssocLinearRMSE = sqrt(mean(squareError));

tracker = tracker.initiator(P_G,meas_model.d,xstart,Pstart);
multiHypothesesEstimates = cell(K,1);
squareError = zeros(K,motion_model.d);

hypothesesWeight = 1;
multiHypotheses = struct('x',tracker.x,'P',tracker.P);
for k = 1:K
    [tracker, hypothesesWeight, multiHypotheses] = ...
        multiHypothesesTracker(tracker, hypothesesWeight, multiHypotheses, measdata.Z{k}, sensor_model, motion_model, meas_model);
    squareError(k,:) = (tracker.x - targetdata.X{k}).^2;
    multiHypothesesEstimates{k}.x = tracker.x;
    multiHypothesesEstimates{k}.P = tracker.P;
end
multiHypothesesRMSE = sqrt(mean(squareError));


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
