function [x_upd, P_upd] = KalmanUpdate(x, P, z, measmodel)
%KALMANUPDATE performs linear/nonlinear (Extended) Kalman update step
%INPUT: z: measurements --- (measurement dimension) x 1 vector
%       x: target state --- (state dimension) x 1 vector
%       P: target state covariance --- (state dimension) x (state
%           dimension) matrix
%       measmodel: a structure specifies the measurement model parameters:
%                   H: function handle return transition/Jacobian matrix
%                   h: function handle return the observation of the target state
%                   R: measurement noise covariance matrix
%OUTPUT:x_upd: updated target state --- (state dimension) x 1 vector
%       P_upd: updated target state covariance --- (state dimension) x (state
%           dimension) matrix

%Measurement model Jacobian
Hx = measmodel.H(x);
%Innovation covariance
S = Hx*P*Hx' + measmodel.R;
%Make sure matrix S is positive definite
S = (S+S')/2;

%Use choleskey decomposition to speed up matrix inversion
Vs = chol(S); 
inv_sqrt_S = inv(Vs); 
iS= inv_sqrt_S*inv_sqrt_S';
K  = P*Hx'*iS;
% K = (P*Hx')/S;

%State update
x_upd = x + K*(z - measmodel.h(x));
%Covariance update
P_upd = (eye(size(x,1)) - K*Hx)*P;
%Make sure P_upd is symmetric
P_upd = 0.5*(P_upd + P_upd');


end