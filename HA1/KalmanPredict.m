function [x_pred, P_pred] = KalmanPredict(x, P, motionmodel)
%KALMANPREDICT performs linear/nonlinear Kalman prediction step

if motionmodel.d == 4
    x_pred = motionmodel.A*x;
    P_pred = motionmodel.Q+motionmodel.A*P*motionmodel.A';
elseif motionmodel.d == 5
    x_pred = motionmodel.f(x);
    P_pred = motionmodel.F(x)*P*motionmodel.F(x)'+motionmodel.Q;
    %Make sure P is symmetric
    P_pred = (P_pred+P_pred')/2;
end

end