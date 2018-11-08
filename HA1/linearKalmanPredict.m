function [x_pred, P_pred] = linearKalmanPredict(x, P, motionmodel)
%LINEARKALMANPREDICT performs linear Kalman prediction step
x_pred = motionmodel.A*x;
P_pred = motionmodel.Q+motionmodel.A*P*motionmodel.A';
end

