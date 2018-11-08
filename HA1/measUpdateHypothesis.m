function [meas_likelihood, x_upd, P_upd] = measUpdateHypothesis(x, P, z, measmodel)
%MEASUPDATEHYPOTHESIS calculates the measurement update likelihood
%INPUT: x: single target state --- (target state dimension) x 1 vector
%       P: single target state uncertainty --- (target state dimension) x
%           (target state dimension) matrix
%       z: measurements --- (measurement dimension) x (number of
%           measurements) matrix
%       measmodel: a structure specifies the measurement model parameters
%           d: measurement dimension --- scalar
%           H: observation matrix --- (measurement dimension) x
%               (target state dimension) matrix
%           R: measurement noise covariance --- (measurement dimension) x
%               (measurement dimension) matrix
%OUTPUT: meas_likelihood: likelihood of measurement update --- scalar
%        x_upd: updated mean --- (target state dimension) x (number of measurements) matrix
%        P_upd: updated covariance --- (target state dimension) x
%        (target state dimension) x (number of measurements) matrix

s_d = size(x,1);
num_meas = size(z,2);
x_upd = zeros(s_d,num_meas);
P_upd = zeros(s_d,s_d,num_meas);

%Calculate measurement update likelihood
meas_likelihood = measLikelihood(x, P, z, measmodel);
%For each measurment in the gate, perform Kalman update
for i = 1:num_meas
    [x_upd(:,i), P_upd(:,:,i)] = linearKalmanUpdate(x, P, z(:,i), measmodel);
end

end