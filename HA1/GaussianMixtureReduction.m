function [x_hat, P_hat] = GaussianMixtureReduction(w, x, P)
%GAUSSIANMIXTUREREDUCTION: approximate a Gaussian mixture
%density as a single Gaussian
%INPUT: w: normalised weight of Gaussian components --- (number
%of Gaussians) x 1 vector
%       x: means of Gaussian components --- (variable dimension)
%       x (number of Gaussians) matrix
%       P: variances of Gaussian components --- (variable
%       dimension) x (variable dimension) x (number of Gaussians) matrix
%OUTPUT:x_hat: approximated mean --- (variable dimension) x 1
%vector
%       P_hat: approximated covariance --- (variable dimension)
%       x (variable dimension) matrix
x_hat = x*w;
numGaussian = length(w);
P_hat = zeros(size(P(:,:,1)));
for i = 1:numGaussian
    x_diff = x(:,i) - x_hat;
    P_hat = w(i).*(P(:,:,i) + x_diff*x_diff');
end
end