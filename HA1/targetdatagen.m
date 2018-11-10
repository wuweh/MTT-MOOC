function targetdata = targetdatagen(groundtruth,motionmodel,ifnoisy)
%TARGETDATA generates groundtruth target data
%INPUT:  groundtruth specifies the parameters to generate groundtruth
%           nbirths: number of targets to be tracked --- scalar
%           xstart: target initial states --- (target state
%                   dimension) x nbirths matrix
%           tbirth: time step when targets are born --- (target state
%                   dimension) x 1 vector
%           tdeath: time step when targets die --- (target state
%                   dimension) x 1 vector
%           K: total tracking time --- scalar
%        motionmodel: a structure specifies the motion model parameters
%           d: target state dimension --- scalar
%           A: motion transition matrix --- (target state dimension) x
%               (target state dimension) matrix
%           Q: motion noise covariance --- (target state dimension) x
%               (target state dimension) matrix
%        ifnoisy: boolean value indicating whether to generate noisy target
%        trajectory or not
%OUTPUT: targetdata.X:  (K x 1) cell array, each cell stores target states
%                       of size (target state dimension) x (number of targets
%                       at corresponding time step)
%        targetdata.N:  (K x 1) cell array, each cell stores the number of
%                       targets at corresponding time step

%Generate the tracks
K = groundtruth.K;
targetdata.X = cell(K,1);
targetdata.N = zeros(K,1);

for i = 1:groundtruth.nbirths
    targetstate = groundtruth.xstart(:,i);
    for k = groundtruth.tbirth(i):min(groundtruth.tdeath(i),K)
        if ifnoisy
            targetstate = motionmodel.A*targetstate + ...
                motionmodel.B*randn(size(motionmodel.B,2),1);
        else
            targetstate = motionmodel.A*targetstate;
        end
        targetdata.X{k} = [targetdata.X{k} targetstate];
        targetdata.N(k) = targetdata.N(k) + 1;
    end
end

end
