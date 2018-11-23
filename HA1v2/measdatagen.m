function measdata = measdatagen(objectdata, sensormodel, measmodel)
%MEASDATAGEN generates target-generated measurements and clutter
%INPUT:     objectdata: a structure contains object data
%           sensormodel: a structure specifies sensor model parameters
%           measmodel: a structure specifies the measurement model parameters
%OUTPUT:    measdata: cell array of size (total tracking time, 1), each cell stores measurements 
%                     of size (measurement dimension) x (number of measurements at corresponding time step)

%Initialize memory
measdata = cell(length(objectdata.X),1);

%Generate measurements
for k = 1:length(objectdata.X)
    if objectdata.N(k) > 0
        idx = rand(objectdata.N(k),1) <= sensormodel.P_D;
        %Only generate object-originated observations for detected targets
        if isempty(objectdata.X{k}(:,idx))
            measdata{k} = [];
        else
            measdata{k} = mvnrnd(measmodel.h(objectdata.X{k}(:,idx)), measmodel.R)';
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

end