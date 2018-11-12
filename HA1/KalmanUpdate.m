function [x_upd, P_upd] = KalmanUpdate(x, P, z, measmodel)
%KALMANUPDATE performs linear/nonlinear Kalman update step
%INPUT:  z: measurements --- (measurement dimension) x 1 vector

%Measurement model Jacobian
Hx = measmodel.H(x);
%Innovation covariance
S = Hx*P*Hx' + measmodel.R;
%Kalman gain
K = (P*Hx')/S;
%State update
x_upd = x + K*(z - measmodel.h(x));
%Covariance update
P_upd = (eye(size(x,1)) - K*Hx)*P;
%Make sure P_upd is symmetric
P_upd = 0.5*(P_upd + P_upd');
% end


end