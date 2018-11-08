function [x_upd, P_upd] = linearKalmanUpdate(x, P, z, measmodel)
%LINEARKALMANUPDATE performs linear Kalman update step
%INPUT:  z: measurements --- (measurement dimension) x 1 vector
S = measmodel.H*P*measmodel.H' + measmodel.R;
K = P*measmodel.H'/S;
x_upd = x + K*(z - measmodel.H*x);
P_upd = (eye(size(x,1)) - K*measmodel.H)*P;
end