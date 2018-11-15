function [w_upd, x_upd, P_upd] = measUpdateHypothesis(x, P, z, measmodel, P_D)
%MEASUPDATEHYPOTHESIS creates a measurement update hypothesis for each
%measurement
%INPUT: x: target state --- (target state dimension) x 1 vector
%       P: target state uncertainty --- (target state dimension) x
%           (target state dimension) matrix
%       z: measurements --- (measurement dimension) x (number of
%           measurements) matrix
%       measmodel: a structure specifies the measurement model parameters
%           d: measurement dimension --- scalar
%           H: function handle return transition/Jacobian matrix
%           h: function handle return the observation of the target
%                   state
%           R: measurement noise covariance matrix
%       P_D: target detection probability --- scalar
%OUTPUT: w_upd: measurement update hypothesis weight --- (number of measurements) x 1 vector
%        x_upd: updated mean(s) --- (target state dimension) x (number of measurements) matrix
%        P_upd: updated covariance(s) --- (target state dimension) x
%        (target state dimension) x (number of measurements) matrix

s_d = size(x,1);
num_meas = size(z,2);
x_upd = zeros(s_d,num_meas);
P_upd = zeros(s_d,s_d,num_meas);

%Calculate measurement update likelihood
meas_likelihood = measLikelihood(x, P, z, measmodel);
%For each measurment in the gate, perform Kalman update
for i = 1:num_meas
    [x_upd(:,i), P_upd(:,:,i)] = KalmanUpdate(x, P, z(:,i), measmodel);
end
w_upd = meas_likelihood*P_D;

end