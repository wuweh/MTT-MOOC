function [x_pred, P_pred] = KalmanPredict(x, P, motionmodel)
%KALMANPREDICT performs linear/nonlinear (Extended) Kalman prediction step
%INPUT: x: target state --- (state dimension) x 1 vector
%       P: target state covariance --- (state dimension) x (state
%           dimension) matrix
%       motionmodel: a structure specifies the motion model parameters:
%                   F: function handle return transition/Jacobian matrix
%                   f: function handle return predicted targe state
%                   Q: motion noise covariance matrix
%OUTPUT:x_pred: predicted target state --- (state dimension) x 1 vector
%       P_pred: predicted target state covariance --- (state dimension) x (state
%           dimension) matrix

x_pred = motionmodel.f(x);
P_pred = motionmodel.F(x)*P*motionmodel.F(x)'+motionmodel.Q;
%Make sure P is symmetric
P_pred = (P_pred+P_pred')/2;

end