function meas_likelihood = measLikelihood(x, P, z, measmodel)
%MEASLIKELIHOOD calculates the measurement update likelihood,
%i.e., N(y;\hat{y},S).
%INPUT:  z: measurements --- (measurement dimension) x (number
%of measurements) matrix
%OUTPUT: meas_likelihood: measurement update likelihood for
%each measurement --- (measurement dimension) x 1 vector
S = measmodel.H*P*measmodel.H' + measmodel.R;
meas_likelihood = mvnpdf(z',(measmodel.H*x)',S);
end