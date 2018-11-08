function meas_likelihood = measLikelihood(x, P, z, measmodel)
%MEASLIKELIHOOD calculates the measurement update likelihood,
%i.e., N(y;\hat{y},S).
%INPUT:  z: measurements --- (measurement dimension) x (number
%       of measurements) matrix
%OUTPUT: meas_likelihood: measurement update likelihood for
%       each measurement --- (number of measurements) x 1 vector
[m_d, num_meas] = size(z);
S = measmodel.H*P*measmodel.H' + measmodel.R;
y_hat = measmodel.H*x;
[Vs,p] = chol(S);
if p == 1
    meas_likelihood = zeros(num_meas,1);
else
    det_S = prod(diag(Vs))^2;
    inv_sqrt_S = inv(Vs);
    iS = inv_sqrt_S*inv_sqrt_S';
    temp = repmat(y_hat,[1 num_meas]);
    meas_likelihood = exp(-0.5*size(z,m_d)*log(2*pi) - 0.5*log(det_S) - 0.5*dot(z-temp,iS*(z-temp)))';
end
% meas_likelihood = mvnpdf(z',(measmodel.H*x)',S);

end