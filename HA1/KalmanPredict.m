function [x_pred, P_pred] = KalmanPredict(x, P, motionmodel)
%KALMANPREDICT performs linear/nonlinear Kalman prediction step

x_pred = motionmodel.f(x);
P_pred = motionmodel.F(x)*P*motionmodel.F(x)'+motionmodel.Q;
%Make sure P is symmetric
P_pred = (P_pred+P_pred')/2;

end