function measdata = measdatagen(targetdata, sensormodel, measmodel)

measdata.K = targetdata.K;
measdata.Z = cell(targetdata.K,1);

%generate measurements
for k = 1:targetdata.K
    if targetdata.N(k) > 0
        %detected target indices
        idx = rand(targetdata.N(k),1) <= sensormodel.P_D;
        %single target observations if detected
        if isempty(targetdata.X{k}(:,idx))
            measdata.Z{k} = [];
        else
            measdata.Z{k} = mvnrnd(measmodel.H*targetdata.X{k}(:,idx), measmodel.R)';
        end
    end
    %number of clutter points
    N_c = poissrnd(sensormodel.lambda_c);
    %clutter generation
    C = repmat(sensormodel.range_c(:,1),[1 N_c])+ diag(sensormodel.range_c*[-1; 1])*rand(measmodel.d,N_c);
    %measurement is union of detections and clutter
    measdata.Z{k}= [measdata.Z{k} C];                                                                  
end

