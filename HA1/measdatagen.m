function measdata = measdatagen(targetdata, sensormodel, measmodel)
%MEASDATAGEN generates target-generated measurements and clutter
%INPUT: targetdata.X:  (K x 1) cell array, each cell stores target states
%                       of size (target state dimension) x (number of targets
%                       at corresponding time step)
%       targetdata.N:  (K x 1) cell array, each cell stores the number of
%                       targets at corresponding time step
%       sensormodel: a structure specifies sensor model parameters
%           P_D: target detection probability --- scalar
%           lambda_c: average number of clutter measurements
%                   per time scan, Poisson distributed --- scalar
%           range_c: range of surveillance area --- 2 x 2
%                   matrix of the form [xmin xmax;ymin ymax] or 1 x 2
%                   vector for bearing only measurement
%       measmodel: a structure specifies the measurement model parameters
%           d: measurement dimension --- scalar
%           H: function handle return transition/Jacobian matrix
%           h: function handle return the observation of the target
%                   state
%           R: measurement noise covariance matrix
%OUTPUT:measdata: cell array of size (total tracking time, 1), each cell 
%                   stores measurements of size (measurement dimension) x
%                   (number of measurements at corresponding time step)
measdata = cell(length(targetdata.X),1);

%Generate measurements
for k = 1:length(targetdata.X)
    if targetdata.N(k) > 0
        idx = rand(targetdata.N(k),1) <= sensormodel.P_D;
        %Only generate target-generated observations for detected targets
        if isempty(targetdata.X{k}(:,idx))
            measdata{k} = [];
        else
            measdata{k} = mvnrnd(measmodel.h(targetdata.X{k}(:,idx)), measmodel.R)';
        end
    end
    %Number of clutter measurements
    N_c = poissrnd(sensormodel.lambda_c);
    %Generate clutter
    if measmodel.d == 2
        C = repmat(sensormodel.range_c(:,1),[1 N_c])+ diag(sensormodel.range_c*[-1; 1])*rand(measmodel.d,N_c);
    elseif measmodel.d == 1
        C = (sensormodel.range_c(1,2)-sensormodel.range_c(1,1))*rand(measmodel.d,N_c)-sensormodel.range_c(1,2);
    end
    %Total measurements are the union of target detections and clutter
    measdata{k}= [measdata{k} C];                                                                  
end

