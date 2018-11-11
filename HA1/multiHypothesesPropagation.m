function [hypothesesWeightUpdate, multiHypothesesUpdate] = ...
    multiHypothesesPropagation(hypothesesWeight, multiHypotheses, z, gating, sensormodel, motionmodel, measmodel)
%MULTIHYPOTHESESPROPAGATION does one time step prediction and update of
%multiple hypotheses.
%INPUT: hypothesesWeight: the weights of different hypotheses --- 
%                       (number of hypotheses) x 1 vector
%       multiHypotheses: (number of hypotheses) x 1 structure
%                       with two fields: x: target state mean; 
%                       P: target state covariance
%       z: measurements --- (measurement dimension) x (number
%           of measurements) matrix
%       gating: a structure with two fields: P_G: gating size in
%               percentage; size: gating size
%       sensormodel: a structure specifies the sensor parameters
%           P_D: target detection probability --- scalar
%           lambda_c: average number of clutter measurements
%                   per time scan, Poisson distributed --- scalar
%           pdf_c: clutter (Poisson) intensity --- scalar
%       motionmodel: a structure specifies the motion model parameters
%           d: target state dimension --- scalar
%           A: motion transition matrix --- (target state dimension) x
%               (target state dimension) matrix
%           Q: motion noise covariance --- (target state dimension) x
%               (target state dimension) matrix
%       measmodel: a structure specifies the measurement model parameters
%           d: measurement dimension --- scalar
%           H: observation matrix --- (measurement dimension) x
%               (target state dimension) matrix
%           R: measurement noise covariance --- (measurement dimension) x
%               (measurement dimension) matrix
%OUTPUT:hypothesesWeightUpdate: updated hypotheses weights --- 
%                               (number of new hypotheses) x 1 vector
%       multiHypothesesUpdate: (number of new hypotheses) x 1 structure

num_hypotheses = length(multiHypotheses);

%Prediction
for i = 1:num_hypotheses
    [multiHypotheses(i).x, multiHypotheses(i).P] = ...
        KalmanPredict(multiHypotheses(i).x, multiHypotheses(i).P, motionmodel);
end

hypothesesWeightUpdate = zeros(num_hypotheses,1);

%For each hypothesis, generate missed detection hypotheses
multiHypothesesUpdate = struct('x',0,'P',0);
for i = 1:num_hypotheses
    [hypothesesWeightUpdate(i,1), multiHypothesesUpdate(i).x, multiHypothesesUpdate(i).P] = ...
        missDetectHypothesis(multiHypotheses(i).x, multiHypotheses(i).P, sensormodel.P_D, gating.P_G);
    hypothesesWeightUpdate(i,1) = hypothesesWeight(i)*hypothesesWeightUpdate(i,1)*(sensormodel.lambda_c)*(sensormodel.pdf_c);
end

%For each hypothesis, generate measurement update hypotheses
idx = num_hypotheses;
for i = 1:num_hypotheses
    %Perform gating for each hypothesis, only generate hypotheses for
    %measurements in the gate
    z_ingate = Gating(multiHypotheses(i).x, multiHypotheses(i).P, z, measmodel, gating.size);
    if ~isempty(z_ingate)
        [wupd, xupd, Pupd] = measUpdateHypothesis(multiHypotheses(i).x, multiHypotheses(i).P, z_ingate, measmodel);
        for j = 1:length(wupd)
            idx = idx + 1;
            hypothesesWeightUpdate(idx,1) = wupd(j)*hypothesesWeight(i);
            multiHypothesesUpdate(idx).x = xupd(:,j);
            multiHypothesesUpdate(idx).P = Pupd(:,:,j);
        end
    end
end

%Normalize hypotheses weights
hypothesesWeightUpdate = hypothesesWeightUpdate/sum(hypothesesWeightUpdate);

end