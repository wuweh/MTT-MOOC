function meas_likelihood = measLikelihood(x, P, z, measmodel)
%MEASLIKELIHOOD calculates the measurement update likelihood,
%i.e., N(y;\hat{y},S).
%INPUT:  z: measurements --- (measurement dimension) x (number
%           of measurements) matrix
%        x: target state mean --- (target state dimension) x 1 vector
%        P: %target state covariance --- (target state dimension) x (target state dimension) matrix
%        measmodel: a structure specifies the measurement model parameters
%           d: measurement dimension --- scalar
%           H: observation matrix --- (measurement dimension) x
%               (target state dimension) matrix
%           R: measurement noise covariance --- (measurement dimension) x
%               (measurement dimension) matrix
%OUTPUT: meas_likelihood: measurement update likelihood for
%       each measurement --- (number of measurements) x 1 vector
num_meas = size(z,2);

S = measmodel.H(x)*P*measmodel.H(x)' + measmodel.R;
S = (S+S')/2;
y_hat = measmodel.h(x);

[Vs,~] = chol(S);
det_S = prod(diag(Vs))^2;
inv_sqrt_S = inv(Vs);
iS = inv_sqrt_S*inv_sqrt_S';
temp = repmat(y_hat,[1 num_meas]);
meas_likelihood = exp(-0.5*size(z,measmodel.d)*log(2*pi) - 0.5*log(det_S) - 0.5*dot(z-temp,iS*(z-temp)))';
%Check if S is positive finite
% if p == 1
%     %If S is ill-conditioned, set measurement update likelihoods to zero
%     meas_likelihood = zeros(num_meas,1);
% else
%     det_S = prod(diag(Vs))^2;
%     inv_sqrt_S = inv(Vs);
%     iS = inv_sqrt_S*inv_sqrt_S';
%     temp = repmat(y_hat,[1 num_meas]);
%     meas_likelihood = exp(-0.5*size(z,m_d)*log(2*pi) - 0.5*log(det_S) - 0.5*dot(z-temp,iS*(z-temp)))';
% end
% meas_likelihood = mvnpdf(z',(measmodel.H*x)',S);

end